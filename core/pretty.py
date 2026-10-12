"""「来点好看的」频道：短时长 + 高收藏率 + 只要二次元。

与搬史频道（看评论区艾特率）完全不同的判据：
  1. **短**：时长 ≤ max_duration_seconds（默认 120s）；
  2. **收藏率高**：收藏/播放 ≥ min_favorite_play（默认 0.05）；
  3. **只要二次元**：子分区（typename）必须落在白名单里，标题/标签命中真人词直接否掉。

为什么用「子分区白名单」而不是关键词判断二次元：搜索接口返回的 typename 是**子分区**，
实测能精确区分——二次元的落在 MMD·3D / 同人·手书 / MAD·AMV / 人力VOCALOID / 绘画 / 动画综合，
而真人向的落在 仿妆cos / 颜值·网红舞 / 舞蹈综合 / 特摄。
在一份 170 条的采样里，用这套白名单筛掉 44 条后，剩下的**没有一条是真人出镜**。

为什么收藏率用「收藏/播放」而不是「收藏/点赞」：抽样分位数显示 fav/play 区分度更干净
（中位 0.039、70% 位 0.071、90% 位 0.130、最高 0.402），而 fav/like 在二次元内容上
大量落在 0.3~2.1 之间、混在一起没有明确界线。
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from .bili_api import BiliClient, BiliError
from .judge import Candidate, parse_duration
from .state import StateStore


# 剧集/期数标记：substring 列表表达不了「第 4 集」这种带数字的，所以单独用正则
_SERIES_RE = re.compile(r"第\s*\d{1,4}\s*[集话期季]|EP\s*\d{1,4}|第[一二三四五六七八九十]{1,3}[集话期]")

# LLM 复核提示词：让模型判断「画面向好看内容」还是「低创剧集/剧情向」
_REVIEW_TEMPLATE = """下面是B站上一条短视频的元信息。请判断它是不是「好看的二次元画面向内容」。

要的是这三类（画面或动态本身就是看点）：
1. MMD/3D 纸片人舞蹈；
2. 动漫画面剪辑（MAD/AMV、踩点混剪）；
3. 手书/同人动画（画师自己画的短片）。

不要的是：低创自制动画剧集（例如"某某新番 第4集"这种连载）、剧情向/配音向短片、
杂谈解说、教程讲解、静态图文/壁纸合集、真人出镜内容。
注意：虚拟主播本体不算问题（Hololive 之类的 MMD 可以），但直播切片/会限/杂谈不要。

标题：{title}
UP：{author}
分区：{typename}
标签：{tags}
时长：{duration}秒　播放：{play}　点赞率：{like_play}　收藏/点赞：{fav_like}

只回一行：符合就回「好看」，不符合就回「不要」，后面跟不超过 20 字的理由。"""


class PrettyPicker:
    """「来点好看的」候选挑选器（不需要评论请求，纯靠搜索接口自带的数据判断）。"""

    def __init__(self, client: BiliClient, store: StateStore, logger: Any, ctx: Any = None) -> None:
        self._client = client
        self._store = store
        self._logger = logger
        # ctx 用于 LLM 复核（ctx.llm.generate）；没有 ctx 时自动跳过复核
        self._ctx = ctx
        self._llm_available = ctx is not None


    @staticmethod
    def _hit_real_people(text: str, words: Sequence[str]) -> Optional[str]:
        lowered = (text or "").lower()
        for word in words:
            word = (word or "").strip()
            if word and word.lower() in lowered:
                return word
        return None

    def judge(self, item: Dict[str, Any], cfg: Any) -> Optional[str]:
        """返回拒绝原因；通过则返回 None。"""
        duration = parse_duration(str(item.get("duration") or ""))
        if duration <= 0 or duration > cfg.pretty_max_duration_seconds:
            return f"时长不合适（{duration}s）"
        typename = str(item.get("typename") or "")
        if not any(token in typename for token in cfg.pretty_typename_whitelist):
            return f"分区不在二次元白名单（{typename or '未知'}）"
        title = str(item.get("title") or "").replace('<em class="keyword">', "").replace("</em>", "")
        tag = str(item.get("tag") or "")
        hit = self._hit_real_people(f"{title} {tag}", cfg.pretty_real_blacklist)
        if hit:
            return f"疑似真人内容（命中「{hit}」）"
        hit = self._hit_real_people(f"{title} {tag}", cfg.pretty_block_words)
        if hit:
            return f"屏蔽词命中「{hit}」"
        if _SERIES_RE.search(title):
            return "疑似自制剧集/连续投稿"
        ratio = self.rate_of(item, cfg)
        if ratio is None:
            return "缺收藏/点赞/播放数据"
        label = "收藏/点赞" if cfg.pretty_rate_metric == "fav_like" else "收藏/播放"
        if ratio < cfg.pretty_min_favorite_rate:
            return (
                f"收藏率不够（{label} {ratio:.0%} < {cfg.pretty_min_favorite_rate:.0%}）"
            )
        return None

    @staticmethod
    def rate_of(item: Dict[str, Any], cfg: Any) -> Optional[float]:
        """按配置的口径算收藏率；分母缺数据时返回 None。"""
        play = int(item.get("play") or 0)
        like = int(item.get("like") or 0)
        favorite = int(item.get("favorites") or 0)
        if not favorite:
            return None
        if cfg.pretty_rate_metric == "fav_like":
            return favorite / like if like else None
        return favorite / play if play else None

    async def _llm_verdict(self, item: Dict[str, Any], cfg: Any) -> Optional[bool]:
        """让模型判断「是不是好看的画面向内容」。返回 True/False；无法判断时返回 None。"""
        if not getattr(cfg, "pretty_llm_review", False) or not self._llm_available:
            return None
        try:
            play = int(item.get("play") or 0)
            like = int(item.get("like") or 0)
            fav = int(item.get("favorites") or 0)
            prompt = _REVIEW_TEMPLATE.format(
                title=str(item.get("title") or ""),
                author=str(item.get("author") or ""),
                typename=str(item.get("typename") or ""),
                tags=str(item.get("tag") or "")[:80],
                duration=parse_duration(str(item.get("duration") or "")),
                play=play,
                like_play=f"{like / play:.1%}" if play else "-",
                fav_like=f"{fav / like:.2f}" if like else "-",
            )
            result = await self._ctx.llm.generate(
                prompt=prompt,
                model=str(getattr(cfg, "pretty_llm_model", "") or ""),
                temperature=0.2,
                max_tokens=120,
            )
        except Exception as exc:  # noqa: BLE001 - 复核失败不能拖垮搬运
            self._logger.warning(f"[好看] LLM 复核异常，跳过复核: {exc!r}")
            self._llm_available = False
            return None
        if not (isinstance(result, dict) and result.get("success")):
            self._logger.warning(f"[好看] LLM 复核失败（{result.get('error') if isinstance(result, dict) else result}），跳过复核")
            self._llm_available = False
            return None
        verdict = str(result.get("response") or "").strip()
        self._logger.info(f"[好看] LLM 复核：{verdict[:60]}")
        return "不要" not in verdict

    async def pick(self, cfg: Any, extra_keywords: Sequence[str] = ()) -> Optional[Candidate]:
        """从搜索结果里挑一条最好的（收藏率最高的那条，同分随机）。"""
        pool = [k.strip() for k in extra_keywords if k and k.strip()]
        rest = [k for k in cfg.pretty_keywords if k.strip() and k not in pool]
        random.shuffle(rest)
        keywords = pool + rest
        now = int(time.time())
        begin = now - int(cfg.pretty_window_hours[1]) * 3600
        end = now - int(cfg.pretty_window_hours[0]) * 3600

        found: Dict[str, Dict[str, Any]] = {}
        for keyword in keywords[: max(1, cfg.pretty_keywords_per_run)]:
            order = random.choice(list(cfg.pretty_orders) or ["totalrank"])
            page = random.randint(1, max(1, cfg.pretty_page_max))
            try:
                items = await self._client.search_videos(
                    keyword=keyword,
                    order=order,
                    page=page,
                    window_begin=begin,
                    window_end=end,
                    play_min=cfg.pretty_play_min,
                    play_max=cfg.pretty_play_max,
                    video_review_min=0,
                )
            except BiliError as exc:
                self._logger.warning(f"[好看] 搜索「{keyword}」失败: {exc}")
                continue
            for item in items:
                bvid = str(item.get("bvid") or "")
                if bvid and bvid not in found:
                    found[bvid] = item
            self._logger.debug(f"[好看] 搜索「{keyword}」order={order} page={page} 命中 {len(items)} 条")

        ready: List[tuple] = []
        for item in found.values():
            bvid = str(item.get("bvid") or "")
            if not bvid or self._store.is_checked(bvid) or self._store.is_sent(bvid):
                continue
            reason = self.judge(item, cfg)
            if reason:
                self._logger.debug(f"[好看] 跳过 {bvid}：{reason}")
                continue
            duration = parse_duration(str(item.get("duration") or ""))
            ratio = self.rate_of(item, cfg) or 0.0
            # 排序键里混一个随机数：收藏率相同时随机挑，避免每次都搬同一条
            ready.append((ratio, random.random(), item, duration))
        self._logger.info(f"[好看] 候选 {len(found)} 条，合格 {len(ready)} 条")
        if not ready:
            return None
        ready.sort(key=lambda row: (row[0], row[1]), reverse=True)
        # 搜索结果里没有 cid，下载播放地址必须要它，所以选中的候选要再拉一次详情补齐。
        # 顺手用详情里的数据覆盖一遍（搜索接口的统计有时是缓存的），并按真实时长再校验一次。
        attempts = max(3, int(getattr(cfg, "pretty_llm_review_attempts", 3) or 3))
        for _ratio, _, item, _ in ready[:attempts]:
            bvid = str(item.get("bvid") or "")
            verdict = await self._llm_verdict(item, cfg)
            if verdict is False:
                # 判为不要：标记已检查，避免每轮都重新问一次
                self._store.mark_checked(bvid)
                self._logger.info(f"[好看] LLM 判为不要，换下一条：{str(item.get('title') or '')[:26]}")
                continue
            try:
                detail = await self._client.video_detail(bvid)
            except BiliError as exc:
                self._logger.warning(f"[好看] {bvid} 取详情失败: {exc}")
                continue
            cid = int((detail or {}).get("cid") or 0)
            if not cid:
                self._logger.warning(f"[好看] {bvid} 没有 cid，跳过")
                continue
            duration = int((detail or {}).get("duration") or 0)
            if duration <= 0 or duration > cfg.pretty_max_duration_seconds:
                self._logger.info(f"[好看] {bvid} 真实时长 {duration}s 超上限，换下一条")
                continue
            candidate = self._build(item, duration)
            candidate.cid = cid
            candidate.aid = int((detail or {}).get("aid") or candidate.aid)
            stat = (detail or {}).get("stat") or {}
            candidate.play = int(stat.get("view") or candidate.play)
            candidate.like = int(stat.get("like") or candidate.like)
            candidate.favorite = int(stat.get("favorite") or candidate.favorite)
            candidate.typename = str((detail or {}).get("tname") or candidate.typename)
            candidate.author = str(((detail or {}).get("owner") or {}).get("name") or candidate.author)
            candidate.title = str((detail or {}).get("title") or candidate.title)
            real_ratio = self.rate_of(
                {"play": candidate.play, "like": candidate.like, "favorites": candidate.favorite}, cfg
            ) or 0.0
            candidate.favorite_ratio = real_ratio
            candidate.favorite_metric = cfg.pretty_rate_metric
            metric_label = "收藏/点赞" if cfg.pretty_rate_metric == "fav_like" else "收藏/播放"
            self._logger.info(
                f"[好看] 选中 {candidate.bvid} 收藏率={real_ratio:.0%}（{metric_label}）"
                f" 时长={candidate.duration}s 分区={candidate.typename} {candidate.title[:24]}"
            )
            return candidate
        self._logger.info("[好看] 候选都拿不到可用播放信息")
        return None

    def _build(self, item: Dict[str, Any], duration: int) -> Candidate:
        return Candidate(
            bvid=str(item.get("bvid") or ""),
            aid=int(item.get("aid") or 0),
            cid=0,
            title=str(item.get("title") or "").replace('<em class="keyword">', "").replace("</em>", ""),
            author=str(item.get("author") or ""),
            author_mid=int(item.get("mid") or 0),
            play=int(item.get("play") or 0),
            like=int(item.get("like") or 0),
            favorite=int(item.get("favorites") or 0),
            duration=duration,
            typename=str(item.get("typename") or ""),
            tags=[t for t in str(item.get("tag") or "").split(",") if t],
            channel="pretty",
        )
