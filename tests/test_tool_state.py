from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

import json
import sys
import unittest

from maibot_sdk import CONFIG_RELOAD_SCOPE_SELF
from maibot_sdk.context import PluginContext, PluginPaths


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ID = "github.erhan111119.mai-repost-shit"
spec = spec_from_file_location("repost_test_plugin", ROOT / "plugin.py", submodule_search_locations=[str(ROOT)])
plugin_module = module_from_spec(spec)
sys.modules[spec.name] = plugin_module
spec.loader.exec_module(plugin_module)


class ToolStateTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.rpc = AsyncMock(return_value={"success": True})
        self.plugin = plugin_module.create_plugin()
        self.plugin._ctx = PluginContext(PLUGIN_ID, rpc_call=self.rpc)
        self.set_config()

    def set_config(self, enabled: bool = True, tool: bool = True, pretty: bool = False) -> None:
        config = self.plugin.build_default_config()
        config["plugin"]["enabled"] = enabled
        config["tool"]["enabled"] = tool
        config["pretty"]["enabled"] = pretty
        self.plugin.set_plugin_config(config)

    async def update_config(self, scope: str = CONFIG_RELOAD_SCOPE_SELF) -> None:
        await self.plugin.on_config_update(scope, self.plugin.get_plugin_config_data(), "test")

    def expected_call(self, name: str, enabled: bool) -> Dict[str, Any]:
        return {
            "capability": "component.enable" if enabled else "component.disable",
            "args": {
                "name": f"{PLUGIN_ID}.{name}",
                "component_type": "TOOL",
                "scope": "global",
                "stream_id": "",
                "version": "",
            },
        }

    def payloads(self):
        return [call.args[2] for call in self.rpc.await_args_list]

    async def test_initial_config_combinations(self) -> None:
        for enabled, tool, pretty in product((False, True), repeat=3):
            with self.subTest(enabled=enabled, tool=tool, pretty=pretty):
                self.setUp()
                self.set_config(enabled, tool, pretty)
                await self.plugin._sync_tool_enabled()
                self.assertEqual(self.payloads(), [
                    self.expected_call("repost_shi", enabled and tool),
                    self.expected_call("repost_pretty", enabled and pretty),
                ])

    async def test_on_load_syncs_tools(self) -> None:
        client = Mock(aclose=AsyncMock())
        store = Mock(save=AsyncMock())
        with TemporaryDirectory() as temp_dir:
            self.plugin.ctx.paths = PluginPaths(Path(temp_dir) / "data", Path(temp_dir) / "runtime")
            with (
                patch.object(plugin_module, "BiliClient", return_value=client),
                patch.object(plugin_module, "StateStore", return_value=store),
                patch.object(plugin_module, "Pipeline"),
                patch.object(self.plugin, "_schedule_loop", new_callable=AsyncMock),
                patch.object(self.plugin, "_check_targets", new_callable=AsyncMock),
            ):
                try:
                    await self.plugin.on_load()
                    self.assertEqual(self.payloads(), [
                        self.expected_call("repost_shi", True),
                        self.expected_call("repost_pretty", False),
                    ])
                finally:
                    await self.plugin.on_unload()

    async def test_pretty_toggle_only_updates_pretty(self) -> None:
        await self.plugin._sync_tool_enabled()
        for pretty in (True, False):
            self.rpc.reset_mock()
            self.set_config(pretty=pretty)
            await self.update_config()
            self.assertEqual(self.payloads(), [self.expected_call("repost_pretty", pretty)])

    async def test_tool_toggle_does_not_disable_pretty(self) -> None:
        self.set_config(pretty=True)
        await self.plugin._sync_tool_enabled()
        for tool in (False, True):
            self.rpc.reset_mock()
            self.set_config(tool=tool, pretty=True)
            await self.update_config()
            self.assertEqual(self.payloads(), [self.expected_call("repost_shi", tool)])

    async def test_master_toggle_restores_channel_settings(self) -> None:
        self.set_config(tool=False, pretty=True)
        await self.plugin._sync_tool_enabled()
        for enabled in (False, True):
            self.rpc.reset_mock()
            self.set_config(enabled=enabled, tool=False, pretty=True)
            await self.update_config()
            self.assertEqual(self.payloads(), [self.expected_call("repost_pretty", enabled)])

    async def test_disabled_master_keeps_both_tools_disabled(self) -> None:
        self.set_config(enabled=False)
        await self.plugin._sync_tool_enabled()
        self.rpc.reset_mock()
        self.set_config(enabled=False, tool=False, pretty=True)
        await self.update_config()
        self.rpc.assert_not_awaited()

    async def test_unrelated_update_preserves_admin_disabled_tool(self) -> None:
        await self.plugin._sync_tool_enabled()
        await self.plugin.ctx.component.disable_component(f"{PLUGIN_ID}.repost_shi", "tool")
        self.rpc.reset_mock()
        config = self.plugin.get_plugin_config_data()
        config["pretty"]["keywords"] = ["MMD"]
        self.plugin.set_plugin_config(config)
        await self.update_config()
        await self.update_config()
        self.rpc.assert_not_awaited()

    async def test_global_config_notification_does_not_toggle_tools(self) -> None:
        for scope in ("bot", "model"):
            await self.update_config(scope)
        self.rpc.assert_not_awaited()

    async def test_failed_sync_is_reported_and_not_cached(self) -> None:
        self.rpc.side_effect = [{"success": True}, {"success": False, "error": "未找到组件"}]
        with self.assertRaisesRegex(RuntimeError, "repost_pretty.*未找到组件"):
            await self.plugin._sync_tool_enabled()
        self.rpc.reset_mock(side_effect=True)
        await self.update_config()
        self.assertEqual(self.payloads(), [self.expected_call("repost_pretty", False)])

    async def test_rpc_error_propagates(self) -> None:
        self.rpc.side_effect = RuntimeError("capability denied")
        with self.assertRaisesRegex(RuntimeError, "capability denied"):
            await self.plugin._sync_tool_enabled()
        self.assertEqual(self.plugin._tool_enabled, {})

    async def test_on_load_stops_before_resources_when_sync_fails(self) -> None:
        self.rpc.return_value = {"success": False, "error": "未找到组件"}
        with patch.object(plugin_module, "BiliClient") as client:
            with self.assertRaisesRegex(RuntimeError, "同步工具启停失败"):
                await self.plugin.on_load()
            client.assert_not_called()
        self.assertIsNone(self.plugin._loop_task)

    async def test_disabled_tool_keeps_execution_guard(self) -> None:
        result = await self.plugin.tool_pretty(stream_id="test-stream")
        self.assertIn("频道被关掉了", result["content"])
        self.rpc.assert_not_awaited()
        self.assertFalse(self.plugin._manual_busy)

    def test_manifest_declares_toggle_capabilities(self) -> None:
        manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue({"component.enable", "component.disable"} <= set(manifest["capabilities"]))


if __name__ == "__main__":
    unittest.main()
