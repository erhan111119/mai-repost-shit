"""投递：把视频以「聊天记录」形式发进群。

宿主没有 VideoComponent，视频段只能以适配器兜底的 dict 段形态进转发节点。
实测确认可用的写法是「顶层 file 键」：
    {"type": "video", "file": "C:/path/x.mp4"}
不能写成 {"type": "video", "data": {...}}（type 会被剥掉，适配器认不出），
也不能走 base64——插件↔宿主 RPC 帧上限 16MB，base64 膨胀 4/3，稍大一点就炸。
所以这里一律用本地路径引用（NapCat 与插件同机，能直接读文件）。
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
    """负责把一条候选发出去，并在合并转发失败时回退。"""

    def __init__(self, ctx: Any, logger: Any) -> None:
        self._ctx = ctx
        self._logger = logger

    async def send(self, stream_id: str, candidate: Candidate, video_path: Path, cfg: Any) -> Dict[str, Any]:
        """主路径走合并转发；失败则退回「单独发视频 + 补一条文字」。"""
        bot_qq = str(await self._ctx.config.get("bot.qq_account", "") or "")
        bot_name = str(await self._ctx.config.get("bot.nickname", "") or "麦麦")
        messages = build_forward_messages(candidate, video_path, bot_qq, bot_name, cfg.include_header_text)
        try:
            result = await self._ctx.send.forward(messages, stream_id, return_details=True)
            if isinstance(result, dict) and result.get("sent"):
                return {"method": "forward", "detail": result}
            self._logger.warning(f"合并转发未成功，回退为直接发视频: {result}")
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(f"合并转发异常，回退为直接发视频: {exc!r}")

        fallback: Dict[str, Any] = {"method": "fallback"}
        try:
            if cfg.include_header_text:
                await self._ctx.send.text(candidate.header_text(), stream_id)
            result = await self._ctx.send.custom(
                "video", {"file": video_path.as_posix()}, stream_id, return_details=True
            )
            fallback["detail"] = result
            fallback["sent"] = bool(isinstance(result, dict) and result.get("sent"))
        except Exception as exc:  # noqa: BLE001
            self._logger.error(f"直接发视频也失败: {exc!r}")
            fallback["sent"] = False
            fallback["error"] = repr(exc)
        return fallback


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
