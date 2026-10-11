"""搬运流水线：检索 → 预筛 → 艾特率判定 → 下载 → 投递 → 记账。

一次 run 只搬一条。候选漏斗的顺序是刻意的：
先用搜索接口自带的 video_review（评论数）和屏蔽词做零成本过滤，
再对少量候选花请求算艾特率，最后只对通过的那一条下载视频。
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .bili_api import BiliClient, BiliError
from .deliver import Deliverer, append_context
from .judge import Candidate, Judge
from .state import StateStore


class Pipeline:
    """一次搬运流程。"""

    def __init__(
        self,
        ctx: Any,
        client: BiliClient,
        judge: Judge,
        store: StateStore,
        logger: Any,
        work_dir: Path,
    ) -> None:
        self._ctx = ctx
        self._client = client
        self._judge = judge
        self._store = store
        self._logger = logger
        self._work_dir = work_dir
        self._deliverer = Deliverer(ctx, logger)

    def set_judge(self, judge: Judge) -> None:
        """热重载配置后换掉判据（阈值可能变了）。"""
        self._judge = judge

    # ===== 检索 =====

    async def _collect_candidates(self, cfg: Any, extra_keywords: Sequence[str] = ()) -> List[Dict[str, Any]]:
        """跑几个关键词，收集落在播放区间、评论数够、没查过的候选。

        extra_keywords 来自自然语言请求里的主题词（比如「来点鬼畜的」），
        它们优先占用本轮的关键词名额，保证点播的口味真的被搜到。
        """
        pool = [k.strip() for k in extra_keywords if k and k.strip()]
        rest = [k for k in cfg.keywords if k.strip() and k not in pool]
        random.shuffle(rest)
        keywords = pool + rest
        now = int(time.time())
        begin = now - int(cfg.window_hours[1]) * 3600
        end = now - int(cfg.window_hours[0]) * 3600
        found: Dict[str, Dict[str, Any]] = {}
        for keyword in keywords[: max(1, cfg.keywords_per_run)]:
            order = random.choice(list(cfg.orders) or ["totalrank"])
            page = random.randint(1, max(1, cfg.page_max))
            try:
                items = await self._client.search_videos(
                    keyword=keyword,
                    order=order,
                    page=page,
                    window_begin=begin,
                    window_end=end,
                    play_min=cfg.play_min,
                    play_max=cfg.play_max,
                    video_review_min=cfg.min_video_review,
                )
            except BiliError as exc:
                self._logger.warning(f"搜索「{keyword}」失败: {exc}")
                continue
            for item in items:
                bvid = str(item.get("bvid") or "")
                if bvid and bvid not in found:
                    found[bvid] = item
            self._logger.debug(f"搜索「{keyword}」order={order} page={page} 命中 {len(items)} 条")
        return list(found.values())

    async def _pick(self, cfg: Any, extra_keywords: Sequence[str] = ()) -> Optional[Candidate]:
        """从候选池里挑出第一条通过判据的。"""
        raw = await self._collect_candidates(cfg, extra_keywords)
        fresh: List[Dict[str, Any]] = []
        for item in raw:
            bvid = str(item.get("bvid") or "")
            if not bvid or self._store.is_checked(bvid) or self._store.is_sent(bvid):
                continue
            # 先做零成本的预筛（时长/屏蔽词），再排序——否则排序会把长合集排到前面，
            # 白白占掉检查名额又被时长上限否掉
            reason = self._judge.prefilter(item)
            if reason:
                self._store.mark_checked(bvid)
                self._logger.info(f"跳过 {bvid}：{reason}")
                continue
            fresh.append(item)
        self._logger.info(f"本轮候选 {len(raw)} 条，可检查 {len(fresh)} 条")
        # 评论数多的优先：分母足够，艾特率才可信；但封顶 300，避免长合集靠评论数霸榜
        fresh.sort(key=lambda item: min(int(item.get("video_review") or 0), 300), reverse=True)
        checked = 0
        for item in fresh:
            if checked >= cfg.max_check_per_run:
                self._logger.info(f"本轮检查数达上限 {cfg.max_check_per_run}，留到下一轮")
                break
            bvid = str(item.get("bvid") or "")
            checked += 1
            candidate = await self._judge.evaluate(item)
            self._store.mark_checked(bvid)
            if candidate.reject_reason:
                self._logger.info(f"跳过 {bvid}（{candidate.title[:20]}）：{candidate.reject_reason}")
                continue
            self._logger.info(
                f"命中 {bvid} {candidate.title[:28]} 艾特率={candidate.at_ratio:.0%}"
                f"（{candidate.at_count}/{candidate.comment_count}）播放={candidate.play}"
            )
            return candidate
        return None

    # ===== 下载 =====

    async def _download(self, candidate: Candidate, cfg: Any) -> Optional[Path]:
        """取 720P 播放地址并下载（免登录最高就是 720P，不需要转码）。"""
        try:
            info = await self._client.playurl(candidate.bvid, candidate.cid, list(cfg.quality_preference))
        except BiliError as exc:
            self._logger.warning(f"{candidate.bvid} 取播放地址失败: {exc}")
            return None
        segments = info.get("segments") or []
        if not segments:
            self._logger.warning(f"{candidate.bvid} 没有可下载的分段")
            return None
        if len(segments) > 1:
            # 多分段要拼接，本机没有可用的 ffmpeg CLI，直接放弃这一条
            self._logger.info(f"{candidate.bvid} 是多分段视频（{len(segments)} 段），跳过")
            return None
        self._work_dir.mkdir(parents=True, exist_ok=True)
        dest = self._work_dir / f"{candidate.bvid}_{info.get('quality')}.mp4"
        try:
            written = await self._client.download(
                str(segments[0].get("url") or ""), dest, int(cfg.max_size_mb) * 1024 * 1024
            )
        except BiliError as exc:
            self._logger.warning(f"{candidate.bvid} 下载失败: {exc}")
            dest.unlink(missing_ok=True)
            return None
        self._logger.info(f"{candidate.bvid} 下载完成 {written / 1024 / 1024:.1f}MB → {dest.name}")
        return dest

    # ===== 主流程 =====

    async def run_once(
        self,
        cfg: Any,
        override_stream: str = "",
        extra_keywords: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """跑一轮：挑一条、下下来、发出去。返回本轮报告。

        override_stream 非空时只发到该会话（命令与自然语言触发用），
        否则发到配置里的所有群。extra_keywords 是点播的口味词。
        """
        report: Dict[str, Any] = {"sent": False, "reason": "", "candidate": None, "targets": []}
        candidate = await self._pick(cfg, extra_keywords)
        if candidate is None:
            report["reason"] = "本轮没有通过判据的候选"
            return report

        video_path = await self._download(candidate, cfg)
        if video_path is None:
            report["reason"] = "候选下载失败"
            report["candidate"] = self._candidate_brief(candidate)
            return report

        if override_stream:
            targets: List[Tuple[str, str]] = [(override_stream, "override")]
        else:
            targets = [(group_id, group_id) for group_id in cfg.groups]

        for stream_id, label in targets:
            if not override_stream:
                stream_id = await self._resolve_stream(stream_id) or ""
                if not stream_id:
                    self._logger.warning(f"找不到群 {label} 的聊天流，跳过")
                    report["targets"].append({"group": label, "sent": False, "reason": "no_stream"})
                    continue
            result = await self._deliverer.send(stream_id, candidate, video_path, cfg)
            sent = bool(result.get("sent") or result.get("method") == "forward")
            report["targets"].append({"group": label, "sent": sent, "method": result.get("method")})
            if sent:
                await append_context(self._ctx, stream_id, candidate, self._logger)
            else:
                self._logger.error(f"{label} 投递失败: {result}")

        if any(item.get("sent") for item in report["targets"]):
            report["sent"] = True
            self._store.record_sent(
                {
                    "bvid": candidate.bvid,
                    "title": candidate.title,
                    "author": candidate.author,
                    "play": candidate.play,
                    "at_ratio": round(candidate.at_ratio, 3),
                    "comment_count": candidate.comment_count,
                    "sent_at": int(time.time()),
                    "groups": [item["group"] for item in report["targets"] if item.get("sent")],
                }
            )
            self._cleanup_old_videos(cfg.keep_files)
        else:
            report["reason"] = "投递失败"
        report["candidate"] = self._candidate_brief(candidate)
        return report

    async def _resolve_stream(self, group_id: str) -> Optional[str]:
        """把群号解析成真实聊天流 ID。"""
        try:
            stream = await self._ctx.chat.get_stream_by_group_id(str(group_id), platform="qq")
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(f"查询群 {group_id} 的聊天流异常: {exc!r}")
            return None
        if isinstance(stream, dict):
            return str(stream.get("session_id") or stream.get("stream_id") or "") or None
        return None

    def _cleanup_old_videos(self, keep: int) -> None:
        """只保留最近几个视频文件，其余删掉，避免插件数据目录无限膨胀。"""
        files = sorted(self._work_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in files[max(1, keep):]:
            path.unlink(missing_ok=True)

    @staticmethod
    def _candidate_brief(candidate: Candidate) -> Dict[str, Any]:
        data = asdict(candidate)
        data.pop("tags", None)
        return data
