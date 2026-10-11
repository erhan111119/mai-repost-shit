"""投递：把视频以「聊天记录」（合并转发）形式发进群。

**只有这一种投递方式**：转发节点 = 一段说明文字 + 视频本体。不走「直接发视频」或
「按文件发」——那两种会在群里多出消息，也与"搬史"想要的观感不符。

**依赖宿主内部行为的说明（重要）**
宿主没有 VideoComponent，转发节点里的视频段只能以「未知类型 → DictComponent」的形态落到
适配器，再由适配器的 `_build_dict_component_segment` 认领成视频段。也就是说：

    {"type": "video", "file": "C:/x.mp4"}   ← 顶层 file 键，宿主会原样透传

这个写法不在公开 API 里，是照宿主的归一化/装配代码试出来的。宿主一旦改动
`_normalize_plugin_segments` / `_component_from_dict` 的处理，视频会变成空文本段或
`[unsupported:...]` 占位——而且**不会报错**，只会安静地发出一条没有视频的消息。

失败时**只在本地记日志，不往群里发任何提示**（群里只看得到搬出来的视频）。
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
    """把一条候选以合并转发的形式发出去。"""

    def __init__(self, ctx: Any, logger: Any) -> None:
        self._ctx = ctx
        self._logger = logger

    async def send(self, stream_id: str, candidate: Candidate, video_path: Path, cfg: Any) -> Dict[str, Any]:
        """发送一条候选；返回 {"sent": bool, "method": "forward", ...}。"""
        bot_qq = str(await self._ctx.config.get("bot.qq_account", "") or "")
        bot_name = str(await self._ctx.config.get("bot.nickname", "") or "麦麦")
        try:
            size_mb = video_path.stat().st_size / 1024 / 1024
        except OSError:
            size_mb = 0.0
        messages = build_forward_messages(
            candidate, video_path, bot_qq, bot_name, cfg.include_header_text
        )
        try:
            result = await self._ctx.send.forward(messages, stream_id, return_details=True)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(f"投递异常（合并转发调用失败，视频 {size_mb:.1f}MB）: {exc!r}")
            return {"sent": False, "method": "forward", "errors": [repr(exc)]}

        if isinstance(result, dict) and result.get("sent"):
            return {"sent": True, "method": "forward", "detail": result}

        # 失败不往群里说话（群里只会看到搬出来的视频），但日志要写清楚便于排查：
        # 宿主的 result 来自平台批量回执，实测体积过大时 NapCat 会判 FAILED 且 error 为空。
        self._logger.error(
            f"投递失败（合并转发未成功，视频 {size_mb:.1f}MB）：{result}。"
            "体积过大时 NapCat 会上传失败，可调小 [download].max_size_mb"
        )
        return {"sent": False, "method": "forward", "detail": result}


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
