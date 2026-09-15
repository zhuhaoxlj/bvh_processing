"""pytest 全局配置。"""

import webbrowser

import pytest


@pytest.fixture(scope="session", autouse=True)
def _never_open_browser() -> None:
    """整个测试会话都不真的拉起系统浏览器。

    本地 .env 可能同时开着 BVH_DEV_UI 和 BVH_DEV_UI_OPEN_BROWSER，而自动打开是
    延迟触发的：即使某个测试期间打了补丁，线程也可能在补丁撤销后才执行。
    所以这里在整个会话内替换掉实现。
    """
    original = webbrowser.open
    webbrowser.open = lambda *args, **kwargs: False
    yield
    webbrowser.open = original
