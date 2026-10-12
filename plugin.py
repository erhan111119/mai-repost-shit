"""mai repost shit —— 定时把「评论区全是艾特」的B站视频压成 720p 以聊天记录搬进群。

设计要点（都是实测结论，改之前先读 docs/NOTES.md）：
  * 判据只有一条：评论含 @ 比例 ≥ 阈值，且评论样本数达标（否则比值是噪声）。
  * 只搬一条/轮：一天两三次，靠 StateStore 生成随机时刻，重启不会补发一堆。
  * 不转码：免登录 playurl 最高就是 720P，直接按档位下载即可，本机也不需要 ffmpeg。
  * 投递用合并转发 + 本地文件路径引用（base64 会撞 RPC 16MB 帧上限）。
  * 除了命令，还注册了一个 Tool 让群友用自然语言点播（「搬个史」「来点鬼畜的」）。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from maibot_sdk import Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder, ToolParameterInfo, ToolParamType

from .core.bili_api import BiliClient
from .core.defaults import (
    DEFAULT_BLACKLIST,
    DEFAULT_KEYWORDS,
    DEFAULT_PRETTY_KEYWORDS,
    DEFAULT_PRETTY_REAL_BLOCK,
    DEFAULT_PRETTY_TYPENAME,
)
from .core.judge import Judge
from .core.pipeline import Pipeline
from .core.pretty import PrettyPicker
from .core.speaker import SpeakerCache, normalize_qq
from .core.state import StateStore

TOOL_REPOST = "repost_shi"
TOOL_PRETTY = "repost_pretty"
# 入站钩子：群聊里 Tool 回调拿不到发言者 QQ，只能在这里按会话缓存真实发言者
RECEIVE_HOOK = "chat.receive.after_process"
# 回复后处理钩子：把 response 置空即可取消这次回复（宿主会记「返回了空回复」并不发送）
REPLY_HOOK = "maisaka.reply.before_post_process"

# 口语触发的命令正则。宿主用 re.compile(pattern).search(text) 匹配、**非锚定**，
# 所以「憨憨，给我来点好看的」这种带称呼的说法也能命中。
# 为什么做成命令而不是只靠 LLM 工具：实测群里说「来点好康的/涩的/好看的」时，
# 模型只是聊天式回绝（「好看的没有了」），根本不调用工具；命令链是确定性的，
# 而且 intercept 置 1 之后消息不进 LLM、机器人也不会回话。
_ASK_WORDS = r"(?:来点|来几个|再来点|整点|搞点|发点|给点|找点|搬点|搬个|搬几个|搬几)"
_SHI_WORDS = r"(?:史|屎|抽象|鬼畜|答辩|难绷|乐子)"
_PRETTY_WORDS = r"(?:涩|(?<!颜)色|黄|好看|好康|二次元|纸片人|mmd)"
_GAP = r"[^，。！？!?,、]{0,6}"
SHI_TRIGGER_PATTERN = rf"(?i){_ASK_WORDS}{_GAP}{_SHI_WORDS}"
PRETTY_TRIGGER_PATTERN = rf"(?i){_ASK_WORDS}{_GAP}{_PRETTY_WORDS}"


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=False, description="是否启用插件（同时控制定时搬运）")
    config_version: str = Field(default="1.1.1", description="配置版本")


class TargetSectionConfig(PluginConfigBase):
    """投递目标与频率。"""

    __ui_label__ = "投递"
    __ui_icon__ = "send"
    __ui_order__ = 1

    groups: List[str] = Field(default_factory=list, description="目标群号列表（填群号，不是流 ID）")
    daily_times: int = Field(default=3, description="每天搬几次")
    active_hours: List[int] = Field(default_factory=lambda: [9, 23], description="允许发送的时间范围 [起, 止]（24 小时制）")
    catch_up_minutes: int = Field(default=30, description="错过计划时刻后多久内仍补发，超时则跳过")
    include_header_text: bool = Field(default=True, description="聊天记录里视频上方是否带标题/数据说明")


class SourceSectionConfig(PluginConfigBase):
    """候选检索。"""

    __ui_label__ = "检索"
    __ui_icon__ = "search"
    __ui_order__ = 2

    keywords: List[str] = Field(default_factory=lambda: list(DEFAULT_KEYWORDS), description="关键词池，每轮随机抽几个")
    keywords_per_run: int = Field(default=8, description="每轮用几个关键词搜")
    orders: List[str] = Field(default_factory=lambda: ["totalrank", "pubdate"], description="搜索排序方式")
    page_max: int = Field(default=3, description="每个关键词随机翻到第几页（上限）")
    window_hours: List[int] = Field(default_factory=lambda: [72, 720], description="发布时间窗 [最近, 最远]（小时）")
    play_min: int = Field(default=50000, description="播放量下限")
    play_max: int = Field(default=300000, description="播放量上限")
    min_video_review: int = Field(default=40, description="评论数下限（搜索接口自带，用来保证艾特率有分母）")
    max_video_review: int = Field(
        default=600,
        description="评论数上限（0 表示不限）。实测评论越多艾特率越低（40~150 条中位 33%、600+ 只 10%），"
        "所以把解说/剧集类大热门筛掉",
    )
    max_check_per_run: int = Field(
        default=16,
        description="每轮最多实际检查几条候选（每条约 4 个请求）。"
        "实测能过 50% 艾特率的候选约占一成，检查越多越容易出片",
    )


class JudgeSectionConfig(PluginConfigBase):
    """艾特率判据。"""

    __ui_label__ = "判据"
    __ui_icon__ = "filter"
    __ui_order__ = 3

    at_ratio_threshold: float = Field(default=0.5, description="评论含 @ 比例阈值")
    min_comments: int = Field(default=30, description="评论样本下限，不足则判样本不足")
    comment_pages: int = Field(default=3, description="取几页评论（每页 20 条）")


class DownloadSectionConfig(PluginConfigBase):
    """下载与体积控制。"""

    __ui_label__ = "下载"
    __ui_icon__ = "download"
    __ui_order__ = 4

    max_duration_seconds: int = Field(default=240, description="视频时长上限（秒）")
    max_size_mb: int = Field(
        default=15,
        description="体积上限（MB）。超限会自动降一档清晰度；实测 37MB 的视频三条投递路线都会被"
        "平台判失败，所以别放太大",
    )
    quality_preference: List[int] = Field(
        default_factory=lambda: [64, 32, 16], description="清晰度优先级（64=720P, 32=480P, 16=360P）"
    )
    keep_files: int = Field(default=2, description="本地保留最近几个视频文件")


class FilterSectionConfig(PluginConfigBase):
    """屏蔽词与内容安全。"""

    __ui_label__ = "屏蔽"
    __ui_icon__ = "ban"
    __ui_order__ = 5

    title_blacklist: List[str] = Field(
        default_factory=lambda: list(DEFAULT_BLACKLIST), description="标题屏蔽词（子串匹配，命中即跳过）"
    )
    tag_blacklist: List[str] = Field(
        default_factory=list, description="标签屏蔽词（子串匹配，命中即跳过）"
    )
    up_blacklist: List[str] = Field(
        default_factory=list, description="UP 主屏蔽词（子串匹配，命中即跳过）"
    )


class ToolSectionConfig(PluginConfigBase):
    """自然语言点播（LLM Tool）。"""

    __ui_label__ = "自然语言"
    __ui_icon__ = "message-square"
    __ui_order__ = 6

    enabled: bool = Field(default=True, description="是否允许用自然语言叫它搬史")
    permission: str = Field(
        default="anyone",
        description="谁能触发：anyone=群成员都可以 / whitelist=只有下白名单里的 QQ 能触发。"
        "注意工具回调拿不到发言者 QQ，白名单是借助入站钩子按会话回查的",
    )
    whitelist: List[str] = Field(default_factory=list, description="permission=whitelist 时允许触发的 QQ 号列表")
    speaker_cache_ttl_seconds: int = Field(default=600, description="发言者缓存有效期（秒），超时后白名单模式会拒绝")
    cooldown_minutes: int = Field(default=10, description="同一个群两次点播的最小间隔（分钟）")
    daily_limit_per_group: int = Field(default=3, description="每个群每天点播的次数上限（0 表示不限）")
    max_count: int = Field(default=3, description="单次点播最多搬几条")
    suppress_reply: bool = Field(
        default=True,
        description="点播后是否吞掉机器人这次的回话（只调用搬史工具、不在群里多说话）",
    )
    suppress_reply_window_seconds: int = Field(
        default=120, description="吞回话的有效窗口（秒）：点播后这段时间内的第一条回复会被取消"
    )


class PrettySectionConfig(PluginConfigBase):
    """「来点好看的」频道：短时长 + 高收藏率 + 只二次元。"""

    __ui_label__ = "来点好看的"
    __ui_icon__ = "sparkles"
    __ui_order__ = 7

    enabled: bool = Field(default=True, description="是否允许用自然语言点播「好看的」（二次元向）")
    keywords: List[str] = Field(
        default_factory=lambda: list(DEFAULT_PRETTY_KEYWORDS), description="关键词池，每轮随机抽几个"
    )
    keywords_per_run: int = Field(default=6, description="每轮用几个关键词搜")
    orders: List[str] = Field(default_factory=lambda: ["totalrank", "pubdate"], description="搜索排序方式")
    page_max: int = Field(default=3, description="每个关键词随机翻到第几页（上限）")
    window_hours: List[int] = Field(default_factory=lambda: [72, 720], description="发布时间窗 [最近, 最远]（小时）")
    play_min: int = Field(default=10000, description="播放量下限")
    play_max: int = Field(default=500000, description="播放量上限")
    max_duration_seconds: int = Field(default=120, description="时长上限（秒）——「时长不长」就是这个")
    min_favorite_play: float = Field(
        default=0.05,
        description="收藏率（收藏/播放）下限。实测这批候选：中位 0.039、70% 位 0.071、90% 位 0.130，"
        "0.05 大约留下四成",
    )
    typename_whitelist: List[str] = Field(
        default_factory=lambda: list(DEFAULT_PRETTY_TYPENAME),
        description="子分区白名单（子串匹配）：只放二次元向分区，真人向（仿妆cos/颜值/舞蹈综合/特摄）不在里面",
    )
    real_blacklist: List[str] = Field(
        default_factory=lambda: list(DEFAULT_PRETTY_REAL_BLOCK),
        description="真人向屏蔽词：标题或标签命中即跳过",
    )
    cooldown_minutes: int = Field(default=10, description="同一个群两次点播的最小间隔（分钟）")
    daily_limit_per_group: int = Field(default=3, description="每个群每天点播的次数上限（0 表示不限）")


class ShitpostConfig(PluginConfigBase):
    """插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    target: TargetSectionConfig = Field(default_factory=TargetSectionConfig)
    source: SourceSectionConfig = Field(default_factory=SourceSectionConfig)
    judge: JudgeSectionConfig = Field(default_factory=JudgeSectionConfig)
    download: DownloadSectionConfig = Field(default_factory=DownloadSectionConfig)
    filters: FilterSectionConfig = Field(default_factory=FilterSectionConfig)
    tool: ToolSectionConfig = Field(default_factory=ToolSectionConfig)
    pretty: PrettySectionConfig = Field(default_factory=PrettySectionConfig)


@dataclass
class RunConfig:
    """一次搬运所需的扁平配置，供 core 层的模块使用。"""

    groups: List[str]
    include_header_text: bool
    keywords: List[str]
    keywords_per_run: int
    orders: List[str]
    page_max: int
    window_hours: List[int]
    play_min: int
    play_max: int
    min_video_review: int
    max_video_review: int
    max_check_per_run: int
    at_ratio_threshold: float
    min_comments: int
    comment_pages: int
    max_duration_seconds: int
    max_size_mb: int
    quality_preference: List[int]
    keep_files: int
    title_blacklist: List[str]
    tag_blacklist: List[str]
    up_blacklist: List[str]
    # 「来点好看的」频道
    pretty_keywords: List[str]
    pretty_keywords_per_run: int
    pretty_orders: List[str]
    pretty_page_max: int
    pretty_window_hours: List[int]
    pretty_play_min: int
    pretty_play_max: int
    pretty_max_duration_seconds: int
    pretty_min_favorite_play: float
    pretty_typename_whitelist: List[str]
    pretty_real_blacklist: List[str]


class BilibiliShitpostPlugin(MaiBotPlugin):
    """定时搬运B站「艾特率超标」视频的插件。"""

    config_model = ShitpostConfig

    def __init__(self) -> None:
        super().__init__()
        self._client: Optional[BiliClient] = None
        self._store: Optional[StateStore] = None
        self._pipeline: Optional[Pipeline] = None
        self._loop_task: Optional[asyncio.Task[None]] = None
        self._wake = asyncio.Event()
        self._running = False
        # 点播任务在创建时同步置位，避免连续两次点播都通过重入检查
        self._manual_busy = False
        # 群聊里工具回调拿不到发言者 QQ，靠入站钩子按会话缓存，供白名单校验
        self._speakers: Optional[SpeakerCache] = None
        # 会话 -> 吞回话的截止时间：点播后这段时间内的第一条回复会被取消
        self._reply_suppress: Dict[str, float] = {}

    # ===== 生命周期 =====

    async def on_load(self) -> None:
        """装载状态、建客户端、起定时循环。"""
        data_dir = self.ctx.paths.data_dir
        work_dir = self.ctx.paths.runtime_dir / "clips"
        data_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        self._store = StateStore(data_dir / "state.json", self.ctx.logger)
        self._store.load()
        self._client = BiliClient(self.ctx.logger)
        self._speakers = SpeakerCache(self.config.tool.speaker_cache_ttl_seconds, self.ctx.logger)
        self._pipeline = Pipeline(
            ctx=self.ctx,
            client=self._client,
            judge=Judge(self._client, self._run_config(), self.ctx.logger),
            store=self._store,
            logger=self.ctx.logger,
            work_dir=work_dir,
            pretty=PrettyPicker(self._client, self._store, self.ctx.logger),
        )
        self._loop_task = asyncio.create_task(self._schedule_loop(), name="mai-repost-shit-scheduler")
        mode = self._permission_mode()
        self.ctx.logger.info(
            f"搬史机已装载：目标群 {self.config.target.groups or '（未配置）'}，"
            f"每天 {self.config.target.daily_times} 次，播放区间 "
            f"{self.config.source.play_min}~{self.config.source.play_max}，"
            f"投递方式 聊天记录（合并转发）"
        )
        self.ctx.logger.info(f"自然语言点播权限：{self._describe_permission(mode)}")
        self.ctx.logger.info(
            "频道：搬史（评论区艾特率"
            f" ≥ {self.config.judge.at_ratio_threshold:.0%}）"
            f" + 来点好看的（收藏率 ≥ {self.config.pretty.min_favorite_play:.0%}、时长 ≤ "
            f"{self.config.pretty.max_duration_seconds}s、只二次元，"
            f"{'开' if self.config.pretty.enabled else '关'}）"
        )
        # 权限模式单独喊一遍：默认全员可触发是个有安全含义的默认值，不能只混在装载行里
        if mode == "anyone" and self.config.plugin.enabled and self.config.tool.enabled:
            self.ctx.logger.warning(
                "自然语言点播当前为「全员可触发」：群里任何人说一句就能让它检索并往群里发视频。"
                '要收紧就把 [tool].permission 设为 "whitelist" 并填 [tool].whitelist，'
                "或把 [tool].enabled 设为 false 只留管理员命令 /搬史。"
            )
        if self.config.plugin.enabled:
            await self._check_targets()

    async def _check_targets(self) -> None:
        """装载时把目标解析一遍：解析不到就当场告警，别等定时轮到点才静默跳过。

        注意两点：
          - 聊天流要等该群/该人产生过消息才存在，所以这里只能告警、不能禁用功能；
          - 这是**诊断**步骤，任何异常都必须被吞掉——on_load 抛异常会导致整个插件加载失败。
        """
        if self._pipeline is None:
            return
        try:
            resolved: List[str] = []
            failed: List[str] = []
            for target in self._run_config().groups:
                result = await self._pipeline.resolve_target(target)
                if result is None:
                    failed.append(target)
                else:
                    resolved.append(f"{target}（{result[1]}）")
            if resolved:
                self.ctx.logger.info(f"目标解析成功：{'、'.join(resolved)}")
            if failed:
                self.ctx.logger.warning(
                    f"目标解析不到聊天流：{'、'.join(failed)}"
                    "——定时搬运到这些目标时会跳过。群号写错了？或者该群/该人还没产生过消息？"
                    "（目标既可以是群号，也可以是 QQ 号发私聊）"
                )
        except Exception as exc:  # noqa: BLE001 - 诊断失败不能拖垮加载
            self.ctx.logger.warning(f"目标自检失败（不影响插件加载）: {exc!r}")

    async def on_unload(self) -> None:
        """停循环、落盘、关连接。宿主只给 5 秒，全部操作都要快。"""
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
        self._loop_task = None
        if self._store is not None:
            await self._store.save()
        if self._client is not None:
            await self._client.aclose()

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        """配置热重载：判据的阈值要跟着换。"""
        self.ctx.logger.info(f"配置已更新（scope={scope} version={version}），下一轮生效")
        if self._client is not None and self._pipeline is not None:
            self._pipeline.set_judge(
                Judge(self._client, self._run_config(), self.ctx.logger),
                PrettyPicker(self._client, self._store, self.ctx.logger) if self._store else None,
            )

    # ===== 配置快照 =====

    def _run_config(self) -> RunConfig:
        """把分层配置拍平成 core 层要的一次性配置。"""
        cfg = self.config
        window = list(cfg.source.window_hours)
        while len(window) < 2:
            window.append(720)
        return RunConfig(
            groups=[str(g).strip() for g in cfg.target.groups if str(g).strip()],
            include_header_text=cfg.target.include_header_text,
            keywords=list(cfg.source.keywords),
            keywords_per_run=cfg.source.keywords_per_run,
            orders=list(cfg.source.orders),
            page_max=cfg.source.page_max,
            window_hours=window[:2],
            play_min=cfg.source.play_min,
            play_max=cfg.source.play_max,
            min_video_review=cfg.source.min_video_review,
            max_video_review=cfg.source.max_video_review,
            max_check_per_run=cfg.source.max_check_per_run,
            at_ratio_threshold=cfg.judge.at_ratio_threshold,
            min_comments=cfg.judge.min_comments,
            comment_pages=cfg.judge.comment_pages,
            max_duration_seconds=cfg.download.max_duration_seconds,
            max_size_mb=cfg.download.max_size_mb,
            quality_preference=list(cfg.download.quality_preference),
            keep_files=cfg.download.keep_files,
            title_blacklist=list(cfg.filters.title_blacklist),
            tag_blacklist=list(cfg.filters.tag_blacklist),
            up_blacklist=list(cfg.filters.up_blacklist),
            pretty_keywords=list(cfg.pretty.keywords),
            pretty_keywords_per_run=cfg.pretty.keywords_per_run,
            pretty_orders=list(cfg.pretty.orders),
            pretty_page_max=cfg.pretty.page_max,
            pretty_window_hours=list(cfg.pretty.window_hours)[:2] or [72, 720],
            pretty_play_min=cfg.pretty.play_min,
            pretty_play_max=cfg.pretty.play_max,
            pretty_max_duration_seconds=cfg.pretty.max_duration_seconds,
            pretty_min_favorite_play=cfg.pretty.min_favorite_play,
            pretty_typename_whitelist=list(cfg.pretty.typename_whitelist),
            pretty_real_blacklist=list(cfg.pretty.real_blacklist),
        )

    # ===== 发言者缓存与点播权限 =====

    @HookHandler(
        RECEIVE_HOOK,
        name="repost_shi_cache_speaker",
        description="缓存各会话最近一位真人发言者，供自然语言点播做白名单校验",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        error_policy=ErrorPolicy.SKIP,  # 缓存失败绝不影响入站处理
    )
    async def cache_speaker(self, message: Dict[str, Any] | None = None, **kwargs: Any) -> None:
        """只读观察：记录发言者后原样放行（返回 None）。"""
        del kwargs
        try:
            if self._speakers is None or not isinstance(message, dict):
                return None
            session_id = str(message.get("session_id") or "").strip()
            user_info = (message.get("message_info") or {}).get("user_info") or {}
            speaker = normalize_qq(user_info.get("user_id"))
            if not session_id or not speaker:
                return None  # 系统/通知类消息没有发言者，跳过
            self._speakers.remember(
                session_id,
                speaker,
                str(user_info.get("user_cardname") or user_info.get("user_nickname") or ""),
            )
        except Exception as exc:  # noqa: BLE001 - 缓存失败不影响入站
            self.ctx.logger.debug(f"缓存发言者失败（已忽略）: {exc!r}")
        return None

    # ===== 点播后不回话 =====

    @HookHandler(
        REPLY_HOOK,
        name="repost_shi_suppress_reply",
        description="点播搬史后吞掉机器人这一次的回话（只让工具把视频发出去）",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        error_policy=ErrorPolicy.SKIP,  # 抑制失败不该影响正常回复链路
    )
    async def suppress_reply_after_tool(self, **kwargs: Any) -> Dict[str, Any]:
        """把这一次的回复文本置空以取消发送。

        为什么要用钩子而不是只靠提示词：模型经常在调完工具后顺口说一句「搬着呢」
        「刚给你下了几个」，而用户要的是**群里只出现视频**。宿主对空回复的处理是
        「不发送并记为 failure」，所以置空就能可靠地吞掉这次回话。
        只吞「点播之后的第一条」，避免把随后正常的聊天回复也吃掉。
        """
        try:
            if not self.config.tool.suppress_reply:
                return {"action": "continue"}
            session_id = str(kwargs.get("session_id") or "")
            deadline = self._reply_suppress.get(session_id)
            if not deadline:
                return {"action": "continue"}
            if time.time() > deadline:
                self._reply_suppress.pop(session_id, None)
                return {"action": "continue"}
            self._reply_suppress.pop(session_id, None)
            self.ctx.logger.info("点播搬史后吞掉本次回话（只发视频，不在群里多说话）")
            modified = dict(kwargs)
            modified["response"] = ""
            return {"action": "continue", "modified_kwargs": modified}
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.debug(f"吞回话失败（已忽略）: {exc!r}")
            return {"action": "continue"}

    def _permission_mode(self) -> str:
        """规范化权限模式，未知值一律按最严格的 whitelist 处理。"""
        mode = str(self.config.tool.permission or "anyone").strip().lower()
        return mode if mode in {"anyone", "whitelist"} else "whitelist"

    def _describe_permission(self, mode: str) -> str:
        if mode == "anyone":
            return "全员可触发（仅按群冷却限流）"
        count = len([q for q in (normalize_qq(x) for x in self.config.tool.whitelist) if q])
        return f"白名单（{count} 个 QQ）" if count else "白名单（名单为空，实际无人可触发）"

    def _permission_denied_reason(self, stream_id: str) -> str:
        """返回拒绝原因；允许则返回空串。"""
        mode = self._permission_mode()
        if mode == "anyone":
            return ""
        allowed = {q for q in (normalize_qq(x) for x in self.config.tool.whitelist) if q}
        if not allowed:
            self.ctx.logger.warning("点播权限设为 whitelist 但白名单为空，已拒绝本次触发")
            return "这个功能当前只对白名单开放，但白名单还没配"
        speaker = self._speakers.resolve(stream_id) if self._speakers is not None else None
        if speaker is None:
            # 工具回调拿不到发言者，只能靠入站钩子的缓存；拿不到就拒绝，不能放过
            return "没认出你是谁（缓存里没有你的发言），先在群里说句话再叫我"
        if speaker["user_id"] not in allowed:
            # 缓存的是"该会话最近发言者"，可能不是本次调用者：拒绝的同时丢掉这条缓存，
            # 逼下一次触发重新经过入站钩子，缩短旧条目可能被蹭到的窗口
            if self._speakers is not None:
                self._speakers.forget(stream_id)
            return f"这个功能只对白名单开放（{speaker['user_id']} 不在名单里）"
        return ""

    # ===== 定时循环 =====

    async def _schedule_loop(self) -> None:
        """每分钟醒一次看是否到点；被 cancel 时立刻退出，配合宿主 5 秒卸载预算。"""
        while True:
            try:
                self._wake.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), timeout=60)
            except asyncio.CancelledError:
                break
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                self.ctx.logger.error(f"定时搬运异常: {exc!r}", exc_info=True)

    async def _tick(self) -> None:
        """到点就跑一轮。"""
        if self._pipeline is None or self._store is None:
            return
        cfg = self.config
        if not cfg.plugin.enabled:
            return
        now = datetime.now()
        store = self._store
        store.ensure_plan(
            now,
            times_per_day=cfg.target.daily_times,
            hour_start=int(cfg.target.active_hours[0]),
            hour_end=int(cfg.target.active_hours[1]),
        )
        due = store.next_due(now)
        if due is None:
            await store.save()
            return
        if now.timestamp() < due:
            await store.save()
            return
        # 过期太久（比如插件刚被重启）就跳过，避免开服瞬间补发三条
        if now.timestamp() - due > max(1, cfg.target.catch_up_minutes) * 60:
            store.mark_plan_done(due)
            await store.save()
            self.ctx.logger.info("错过的发送时刻已跳过（超出补发窗口）")
            return
        store.mark_plan_done(due)
        await store.save()
        await self._run_and_report(self._run_config())

    async def _run_and_report(
        self,
        cfg: RunConfig,
        override_stream: str = "",
        extra_keywords: Sequence[str] = (),
        channel: str = "shi",
    ) -> Dict[str, Any]:
        """跑一轮并汇报结果，带重入保护。"""
        if self._pipeline is None:
            return {"sent": False, "reason": "插件尚未就绪"}
        if self._running:
            return {"sent": False, "reason": "上一轮还在进行中"}
        self._running = True
        try:
            report = await self._pipeline.run_once(
                cfg,
                override_stream=override_stream,
                extra_keywords=extra_keywords,
                channel=channel,
            )
        finally:
            self._running = False
        if self._store is not None:
            await self._store.save()
        if report.get("sent"):
            candidate = report.get("candidate") or {}
            self.ctx.logger.info(f"本轮已搬：{candidate.get('title')}")
        else:
            self.ctx.logger.info(f"本轮未搬：{report.get('reason')}")
        return report

    # ===== 口语触发（命令链：确定性、不经过模型、不回话）=====

    @Command(
        "repost_by_words",
        description="口语搬史：群里说「来点史/搬点屎/来点抽象的」就自动搬一条（不走模型、机器人不回话）",
        pattern=SHI_TRIGGER_PATTERN,
    )
    async def cmd_repost_by_words(
        self, text: str = "", stream_id: str = "", user_id: str = "", **kwargs: Any
    ) -> Tuple[bool, str, int]:
        """按口语触发搬史。返回 intercept=1，消息不再进 LLM。"""
        del text, kwargs
        return await self._trigger_by_words("shi", stream_id, user_id)

    @Command(
        "pretty_by_words",
        description="口语点播好看的：群里说「来点涩的/来点黄的/来点好看的/来点好康的」就自动搬一条（不走模型、不回话）",
        pattern=PRETTY_TRIGGER_PATTERN,
    )
    async def cmd_pretty_by_words(
        self, text: str = "", stream_id: str = "", user_id: str = "", **kwargs: Any
    ) -> Tuple[bool, str, int]:
        """按口语触发好看频道。返回 intercept=1，消息不再进 LLM。"""
        del text, kwargs
        return await self._trigger_by_words("pretty", stream_id, user_id)

    def _permission_denied_for_user(self, user_id: str) -> str:
        """用真实 QQ 号判权限（命令链能拿到真实发言者，比工具那条缓存路径更准）。"""
        mode = self._permission_mode()
        if mode == "anyone":
            return ""
        allowed = {q for q in (normalize_qq(x) for x in self.config.tool.whitelist) if q}
        if not allowed:
            return "白名单还没配"
        speaker = normalize_qq(user_id)
        if not speaker:
            return "没认出你是谁"
        if speaker not in allowed:
            return f"只对白名单开放（{speaker} 不在名单里）"
        return ""

    async def _trigger_by_words(self, channel: str, stream_id: str, user_id: str) -> Tuple[bool, str, int]:
        """口语触发的公共实现（两个频道共用）。"""
        # 命令链的返回值不会发到群里（宿主只写日志），所以这里的文字只是给日志看的
        if not self.config.plugin.enabled:
            return False, "插件未启用", 0
        if not stream_id:
            return False, "拿不到当前会话", 0
        if channel == "pretty" and not self.config.pretty.enabled:
            return False, "好看频道未启用", 0
        denied = self._permission_denied_for_user(user_id)
        if denied:
            return False, f"权限不足：{denied}", 1
        if self._store is None:
            return False, "插件未就绪", 0
        if self._manual_busy or self._running:
            return False, "正在搬上一条", 1

        key = f"{stream_id}#pretty" if channel == "pretty" else stream_id
        cooldown = (
            self.config.pretty.cooldown_minutes if channel == "pretty" else self.config.tool.cooldown_minutes
        )
        daily = (
            self.config.pretty.daily_limit_per_group
            if channel == "pretty"
            else self.config.tool.daily_limit_per_group
        )
        block = self._store.manual_block_reason(key, cooldown_minutes=cooldown, daily_limit=daily)
        if block:
            return False, f"被限流：{block}", 1

        self._store.record_manual(key)
        await self._store.save()
        self._manual_busy = True
        task = asyncio.create_task(
            self._manual_share(stream_id, 1, [], channel=channel), name=f"mai-repost-{channel}-words"
        )
        task.add_done_callback(self._on_manual_done)
        # intercept=1：消息不进 LLM，机器人不会为这条消息说话
        return True, f"已触发{channel}频道搬运", 1

    # ===== 自然语言点播（LLM Tool，模型主动调用时走这条）=====

    @Tool(
        TOOL_REPOST,
        brief_description=(
            "从B站搬一条「史」到当前群：专挑评论区艾特率超过 50% 的抽象视频，"
            "下载成 720p 后以聊天记录的形式发出来。"
            "当有人想搬史、要来点抽象的、想看乐子、点名要某类整活/鬼畜视频时调用。"
            "调用后立刻返回，视频会在后台下载并自动以聊天记录形式发出。"
            "**调用之后不要再调用 reply、也不要回任何文字**——群里只会看到搬出来的视频；"
            "不要追问「要不要」「确定吗」，也不要应一声「搬着呢」，插件会吞掉这次回话。"
        ),
        parameters=[
            ToolParameterInfo(
                name="topic",
                param_type=ToolParamType.STRING,
                description="对方点名的口味或主题，例如「鬼畜」「抽象」「答辩」「游戏」；没点名就留空",
                required=False,
            ),
            ToolParameterInfo(
                name="count",
                param_type=ToolParamType.INTEGER,
                description="要几条，默认 1；对方说「来几个」「多来点」时给 2 或 3",
                required=False,
                default=1,
            ),
        ],
        visibility="visible",
        chat_scope="all",
    )
    async def tool_repost(self, topic: str = "", count: int = 1, **kwargs: Any) -> Dict[str, Any]:
        """群友用自然语言点播一条（或几条）史。"""
        stream_id = str(kwargs.get("stream_id") or "")
        if not stream_id:
            return self._tool_result("拿不到当前会话，没法搬。")
        if not self.config.plugin.enabled:
            return self._tool_result("搬史功能没开（插件配置里 [plugin].enabled 还是 false）。")
        if not self.config.tool.enabled:
            return self._tool_result("自然语言点播被关掉了，要用的话让管理员开 [tool].enabled。")
        # 权限：命令是 operator 专属，工具默认全员可用但可收紧成白名单
        denied = self._permission_denied_reason(stream_id)
        if denied:
            return self._tool_result(f"先不搬：{denied}。管理员可以用 /搬史 指令搬。")
        if self._store is None:
            return self._tool_result("插件还没就绪，稍后再试。")
        if self._manual_busy or self._running:
            return self._tool_result("正在搬上一条，等它发出来再说。")

        limit = max(1, self.config.tool.max_count)
        wanted = max(1, min(int(count or 1), limit))
        # 手动触发不占定时配额，但有独立的冷却与每日上限（按会话记账）
        block = self._store.manual_block_reason(
            stream_id,
            cooldown_minutes=self.config.tool.cooldown_minutes,
            daily_limit=self.config.tool.daily_limit_per_group,
        )
        if block:
            return self._tool_result(f"先不搬：{block}。")

        self._store.record_manual(stream_id)
        await self._store.save()
        extra = [topic.strip()] if topic and topic.strip() else []
        self._manual_busy = True
        task = asyncio.create_task(self._manual_share(stream_id, wanted, extra), name="mai-repost-shit-manual")
        task.add_done_callback(self._on_manual_done)
        # 点播后吞掉机器人这一次的回话：只把视频发出去，不在群里多说话
        if self.config.tool.suppress_reply:
            self._reply_suppress[stream_id] = time.time() + max(
                10, self.config.tool.suppress_reply_window_seconds
            )
        return self._tool_result(
            "已提交搬运，视频会在后台下载并以聊天记录形式自动发到这个群，约一两分钟。"
            "不要再调用 reply，也不要回复任何文字。"
        )

    @Tool(
        TOOL_PRETTY,
        brief_description=(
            "从B站搬一条「好看的」二次元视频到当前群：只要纸片人（MMD/手书/MAD/AMV/插画/动画），"
            "不要真人出镜；挑时长 2 分钟以内、收藏率高的。"
            "当有人说「来点涩的/来点色的/来点黄的/来点好看的/来点好康的/来点二次元」这类话时调用。"
            "**调用之后不要再调用 reply、也不要回任何文字**——插件会吞掉这次回话，群里只出现视频。"
            "不要追问「要哪种」，不要应一声「找着呢」。"
        ),
        parameters=[
            ToolParameterInfo(
                name="topic",
                param_type=ToolParamType.STRING,
                description="对方点名的类型，例如「MMD」「手书」「插画」「壁纸」「初音」；没点名就留空",
                required=False,
            ),
            ToolParameterInfo(
                name="count",
                param_type=ToolParamType.INTEGER,
                description="要几条，默认 1；对方说「来几个」时给 2 或 3",
                required=False,
                default=1,
            ),
        ],
        visibility="visible",
        chat_scope="all",
    )
    async def tool_pretty(self, topic: str = "", count: int = 1, **kwargs: Any) -> Dict[str, Any]:
        """用自然语言点播「好看的」二次元视频。"""
        tunnel = "pretty"
        stream_id = str(kwargs.get("stream_id") or "")
        if not stream_id:
            return self._tool_result("拿不到当前会话，没法找。", tunnel)
        if not self.config.plugin.enabled:
            return self._tool_result("搬视频功能没开（插件配置里 [plugin].enabled 还是 false）。", tunnel)
        if not self.config.pretty.enabled:
            return self._tool_result("「好看的」频道被关掉了，要用的话让管理员开 [pretty].enabled。", tunnel)
        denied = self._permission_denied_reason(stream_id)
        if denied:
            return self._tool_result(f"先不搬：{denied}。管理员可以用 /搬好看 指令。", tunnel)
        if self._store is None:
            return self._tool_result("插件还没就绪，稍后再试。", tunnel)
        if self._manual_busy or self._running:
            return self._tool_result("正在搬上一条，等它发出来再说。", tunnel)

        limit = max(1, self.config.tool.max_count)
        wanted = max(1, min(int(count or 1), limit))
        # 冷却与配额按频道分开记，免得点播好看的把搬史的额度吃掉
        block = self._store.manual_block_reason(
            f"{stream_id}#pretty",
            cooldown_minutes=self.config.pretty.cooldown_minutes,
            daily_limit=self.config.pretty.daily_limit_per_group,
        )
        if block:
            return self._tool_result(f"先不搬：{block}。", tunnel)

        self._store.record_manual(f"{stream_id}#pretty")
        await self._store.save()
        extra = [topic.strip()] if topic and topic.strip() else []
        self._manual_busy = True
        task = asyncio.create_task(
            self._manual_share(stream_id, wanted, extra, channel=tunnel),
            name="mai-repost-shit-pretty",
        )
        task.add_done_callback(self._on_manual_done)
        if self.config.tool.suppress_reply:
            self._reply_suppress[stream_id] = time.time() + max(
                10, self.config.tool.suppress_reply_window_seconds
            )
        return self._tool_result(
            "已提交，视频会在后台下载并以聊天记录形式自动发到这个群。"
            "不要再调用 reply，也不要回复任何文字。",
            tunnel,
        )

    async def _manual_share(
        self, stream_id: str, wanted: int, extra_keywords: List[str], channel: str = "shi"
    ) -> None:
        """后台跑完手动点播：每条都是完整的检索→判据→下载→投递。"""
        try:
            cfg = self._run_config()
            for index in range(wanted):
                report = await self._run_and_report(
                    cfg, override_stream=stream_id, extra_keywords=extra_keywords, channel=channel
                )
                if report.get("sent"):
                    continue
                reason = str(report.get("reason") or "没找到合适的")
                # 第一条就没搬成才回话；后续条数失败保持安静，避免刷屏
                if index == 0:
                    with contextlib.suppress(Exception):
                        await self.ctx.send.text(f"这次没搬成：{reason}", stream_id)
                break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.error(f"点播搬运异常: {exc!r}", exc_info=True)

    def _on_manual_done(self, task: asyncio.Task[None]) -> None:
        """收尾：清忙标志，并把异常取出来记日志（否则会被静默吞掉）。"""
        self._manual_busy = False
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.ctx.logger.error(f"点播任务失败: {exc!r}")

    @staticmethod
    def _tool_result(content: str, tool_name: str = TOOL_REPOST) -> Dict[str, Any]:
        """构造 Tool 返回值（内容会进入模型上下文）。"""
        return {"name": tool_name, "content": content}

    # ===== 命令 =====

    @Command(
        "shitpost_now",
        description="立刻按配置搬一条史到指定群（需管理员）",
        pattern=r"^/搬史$",
        permission="operator",
    )
    async def cmd_share_now(self, stream_id: str = "", **kwargs: Any) -> Tuple[bool, str, int]:
        """手动触发一次搬运（仍走配置里的目标群）。"""
        del kwargs
        if not self.config.plugin.enabled:
            return False, "插件未启用，请先在配置里打开", 0
        if not self.config.target.groups:
            return False, "还没有配置目标群号", 0
        cfg = self._run_config()
        report = await self._run_and_report(cfg)
        if report.get("sent"):
            candidate = report.get("candidate") or {}
            return True, f"已搬：{candidate.get('title')}", 1
        return False, f"没搬成：{report.get('reason')}", 0

    @Command(
        "shitpost_here",
        description="立刻搬一条史发到当前会话，用来试效果（需管理员）",
        pattern=r"^/搬史测试$",
        permission="operator",
    )
    async def cmd_share_here(self, stream_id: str = "", **kwargs: Any) -> Tuple[bool, str, int]:
        """把一条视频发到当前会话，便于在不打扰群的情况下验证投递链路。"""
        del kwargs
        if not stream_id:
            return False, "拿不到当前会话 ID", 0
        cfg = self._run_config()
        report = await self._run_and_report(cfg, override_stream=stream_id)
        if report.get("sent"):
            candidate = report.get("candidate") or {}
            return True, f"已发到本会话：{candidate.get('title')}", 1
        return False, f"没成功：{report.get('reason')}", 0

    @staticmethod
    def _mask_groups(groups: List[str]) -> str:
        """群号打码：/搬史状态 任何会话都能调，配置里的群号不必暴露完整。

        只保留首 3 位与末 2 位（群号本来就是群里可见的，打码是防止被顺手抄走）。
        """
        masked: List[str] = []
        for raw in groups:
            value = str(raw).strip()
            if not value:
                continue
            masked.append(value if len(value) <= 5 else f"{value[:3]}****{value[-2:]}")
        return "、".join(masked) if masked else "（未配置）"

    @staticmethod
    def _sanitize_bvid(bvid: str) -> str:
        """文件名只允许字母数字，避免上游返回值里混进路径字符。"""
        safe = "".join(ch for ch in str(bvid) if ch.isalnum())
        return safe or "video"

    @Command(
        "pretty_here",
        description="立刻搬一条「好看的」二次元视频到当前会话（需管理员）",
        pattern=r"^/搬好看$",
        permission="operator",
    )
    async def cmd_pretty_here(self, stream_id: str = "", **kwargs: Any) -> Tuple[bool, str, int]:
        """把一条好看的二次元视频发到当前会话。"""
        del kwargs
        if not stream_id:
            return False, "拿不到当前会话 ID", 0
        if not self.config.pretty.enabled:
            return False, "「好看的」频道未启用", 0
        report = await self._run_and_report(
            self._run_config(), override_stream=stream_id, channel="pretty"
        )
        if report.get("sent"):
            candidate = report.get("candidate") or {}
            return True, f"已搬：{candidate.get('title')}", 1
        return False, f"没搬成：{report.get('reason')}", 0

    @Command(
        "shitpost_status",
        description="查看今日搬运计划与最近记录",
        pattern=r"^/搬史状态$",
    )
    async def cmd_status(self, stream_id: str = "", **kwargs: Any) -> Tuple[bool, str, int]:
        """回报计划、配额与最近搬过什么。"""
        del kwargs
        if self._store is None:
            return False, "插件尚未就绪", 0
        now = datetime.now()
        self._store.ensure_plan(
            now,
            times_per_day=self.config.target.daily_times,
            hour_start=int(self.config.target.active_hours[0]),
            hour_end=int(self.config.target.active_hours[1]),
        )
        plan = self._store.plan_summary(now)
        recent = self._store.recent_sent(5)
        lines = [
            f"今日计划 {plan['done']}/{plan['total']} 次，时刻 {', '.join(plan['times']) or '无'}",
            f"时间窗 {self.config.source.window_hours} 小时，播放 "
            f"{self.config.source.play_min}~{self.config.source.play_max}",
            f"判据：艾特率 ≥ {self.config.judge.at_ratio_threshold:.0%}"
            f" 且评论 ≥ {self.config.judge.min_comments} 条",
            f"目标群 {self._mask_groups(self.config.target.groups)}",
            "投递方式 聊天记录（合并转发）；失败只记本地日志，不在群里说话",
            f"自然语言点播：{'开' if self.config.tool.enabled else '关'}"
            f"（{self._describe_permission(self._permission_mode())}，"
            f"冷却 {self.config.tool.cooldown_minutes} 分钟，每群每天 "
            f"{self.config.tool.daily_limit_per_group} 次）",
        ]
        if recent:
            lines.append("最近搬过：")
            for record in reversed(recent):
                when = datetime.fromtimestamp(record.get("sent_at", 0)).strftime("%m-%d %H:%M")
                lines.append(
                    f"· {when} {str(record.get('title') or '')[:24]}"
                    f"（艾特率 {float(record.get('at_ratio') or 0):.0%}）"
                )
        text = "\n".join(lines)
        if stream_id:
            await self.ctx.send.text(text, stream_id)
        return True, text, 1


def create_plugin() -> BilibiliShitpostPlugin:
    """插件工厂。"""
    return BilibiliShitpostPlugin()
