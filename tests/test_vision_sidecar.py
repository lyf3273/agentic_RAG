"""视觉旁路：配图路径解析与 SVG 栅格化（不调用付费 API）。"""

from pathlib import Path

from vision_sidecar import build_figure_index, figure_relevant_to_query, resolve_figure_path


def test_resolve_fig0_1():
    import vision_sidecar as vs
    root = Path(__file__).resolve().parents[1] / "data"
    vs._figure_index = build_figure_index(root)
    assert "fig0-1.svg" in vs._figure_index or "fig0-1" in vs._figure_index
    path = resolve_figure_path("图0-1 Agent = LLM + 上下文 + 工具")
    assert path is not None
    assert path.name == "fig0-1.svg"


def test_resolve_with_file_hint():
    import vision_sidecar as vs
    root = Path(__file__).resolve().parents[1] / "data"
    vs._figure_index = build_figure_index(root)
    path = resolve_figure_path("随便标题", "images/fig1-1.svg")
    assert path is not None
    assert path.name == "fig1-1.svg"


def test_figure_gate():
    assert figure_relevant_to_query("这张架构图什么意思", "图5-1 Coding Agent 架构", "")
    assert not figure_relevant_to_query("你会什么安全", "图5-1 全书结构", "构建Agent")
