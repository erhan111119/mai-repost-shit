"""投递：把视频以「聊天记录」形式发进群。

**依赖宿主内部行为的说明（重要）**
宿主没有 VideoComponent，转发节点里的视频段只能以「未知类型 → DictComponent」的形态落到
适配器，再由适配器的 `_build_dict_component_segment` 认领成视频段。也就是说：

    {"type": "video", "file": "C:/x.mp4"}   ← 顶层 file 键，宿主会原样透传

这个写法不在公开 API 里，是照宿主的归一化/装配代码试出来的。宿主一旦改动
`_normalize_plugin_segments` / `_component_from_dict` 的处理，视频会变成空文本段或
`[unsupported:...]` 占位——而且**不会报错**，只会安静地发出一条没有视频的消息。

所以这里做了三件事：
  1. 三级回退：转发带视频 → 直接发视频 → 按「文件」发（FileComponent 是宿主正式建模的组件，
     最不依赖内部行为，代价是群里显示成文件而不是视频播放器）；
  2. `[target].delivery_mode` 可手动指定路线，宿主更新后一条配置就能绕开坏掉的那条；
  3. 全部失败时**大声失败**：记 error 日志并按 `notify_on_failure` 往群里说一句。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .judge import Candidate


def build_forward_messages(
    candidate: Candidate,
    video_path: Path,
    bot_qq: str,
    bot_name: str,
    include_header: bool = True,
) -> List[Dict[str, Any]]:
    """构造合并转发节点：一段说明文字 + 视频本体。"""
    segments: List[Dict[str, Any]] = []
    if include_header:
        segments.append({"type": "text", "content": candidate.header_text()})
    segments.append({"type": "video", "file": video_path.as_posix()})
    return [{"user_id": bot_qq, "nickname": bot_name, "segments": segments}]


class Deliverer:
    """按配置的投递模式发送，失败按顺序回退。"""

    # 超过这个体积就别把视频塞进转发节点了：本地文件大视频在转发节点里最容易失败
    _FORWARD_SIZE_HINT = 10 * 1024 * 1024

    def __init__(self, ctx: Any, logger: Any) -> None:
        self._ctx = ctx
        self._logger = logger

    def _chain(self, mode: str, video_size: int) -> List[str]:
        """决定投递顺序。

        小文件优先走「聊天记录里带视频」（观感最好）；文件偏大时优先直接发视频——
        实测 37MB 的视频放进转发节点、直接发、按文件发三条路全被 NapCat 判定失败，
        所以大文件不适合先试最脆的那条。
        """
        if mode == "video":
            return ["video", "file", "forward"]
        if mode == "file":
            return ["file", "video", "forward"]
        if video_size > self._FORWARD_SIZE_HINT:
            return ["video", "forward", "file"]
        return ["forward", "video", "file"]

    async def send(self, stream_id: str, candidate: Candidate, video_path: Path, cfg: Any) -> Dict[str, Any]:
        """发送一条候选；返回 {"sent": bool, "method": str, ...}。"""
        bot_qq = str(await self._ctx.config.get("bot.qq_account", "") or "")
        bot_name = str(await self._ctx.config.get("bot.nickname", "") or "麦麦")
        mode = str(getattr(cfg, "delivery_mode", "forward") or "forward").lower()
        try:
            video_size = video_path.stat().st_size
        except OSError:
            video_size = 0
        chain = self._chain(mode, video_size)

        errors: List[str] = []
        for method in chain:
            try:
                if method == "forward":
                    messages = build_forward_messages(
                        candidate, video_path, bot_qq, bot_name, cfg.include_header_text
                    )
                    result = await self._ctx.send.forward(messages, stream_id, return_details=True)
                    if isinstance(result, dict) and result.get("sent"):
                        return {"sent": True, "method": "forward", "detail": result}
                    errors.append(f"forward 未成功: {result}")
                elif method == "video":
                    result = await self._ctx.send.custom(
                        "video", {"file": video_path.as_posix()}, stream_id, return_details=True
                    )
                    if isinstance(result, dict) and result.get("sent"):
                        # 先确认视频发出去再补说明文字，避免失败时留下没有视频的孤儿文字
                        if cfg.include_header_text:
                            await self._ctx.send.text(candidate.header_text(), stream_id)
                        return {"sent": True, "method": "video", "detail": result}
                    errors.append(f"video 段未成功: {result}")
                else:
                    # FileComponent 是宿主正式建模的组件，最不依赖内部段格式；群里显示成文件
                    result = await self._ctx.send.custom(
                        "file",
                        {"url": video_path.as_posix(), "name": f"{candidate.bvid}.mp4"},
                        stream_id,
                        return_details=True,
                    )
                    if isinstance(result, dict) and result.get("sent"):
                        if cfg.include_header_text:
                            await self._ctx.send.text(candidate.header_text(), stream_id)
                        return {"sent": True, "method": "file", "detail": result}
                    errors.append(f"file 段未成功: {result}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{method} 异常: {exc!r}")

        # 三条路都失败。注意：宿主的返回值来自平台批量回执，实测出现过「上传其实落到了群里、
        # 回执却是 FAILED 且 error 为空」的情况，所以这里只陈述事实，不替平台下结论。
        self._logger.error(
            f"投递失败（{video_size / 1024 / 1024:.1f}MB，三条路都返回失败；"
            f"体积过大时 NapCat 上传会失败或回报失败，可调小 [download].max_size_mb）："
            + "；".join(errors)
        )
        if getattr(cfg, "notify_on_failure", True):
            try:
                await self._ctx.send.text(
                    "这次没搬成：平台报投递失败。如果群里已经出现视频，忽略这条即可"
                    "（平台偶尔会上传成功却回报失败）。",
                    stream_id,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(f"失败提示也发不出去: {exc!r}")
        return {"sent": False, "method": "none", "errors": errors}


async def append_context(ctx: Any, stream_id: str, candidate: Candidate, logger: Any) -> None:
    """把搬运记录写进聊天流上下文，让 bot 之后能接上这个话题。"""
    text = (
        f"[我刚刚在群里搬了一条B站视频] {candidate.title}"
        f"（UP：{candidate.author}，播放 {candidate.play:,}，评论区艾特率 {candidate.at_ratio:.0%}）"
        f" https://www.bilibili.com/video/{candidate.bvid}"
    )
    try:
        result = await ctx.maisaka.context.append(
            stream_id,
            [{"type": "text", "content": text}],
            visible_text=text,
            source_kind="bilibili_share",
        )
        if not (isinstance(result, dict) and result.get("success")):
            logger.warning(f"搬运记录注入上下文失败: {result}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"搬运记录注入上下文异常: {exc!r}")
