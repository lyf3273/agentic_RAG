"""查询规划 JSON 解析。"""

from query_plan import QueryPlan, _extract_json, looks_like_chitchat, plan_query


def test_extract_json_fence():
    data = _extract_json('```json\n{"complexity":"simple","strategies":[]}\n```')
    assert data["complexity"] == "simple"


def test_plan_fallback_on_bad_llm():
    plan = plan_query("什么是 RAG", lambda s, u: "not-json")
    assert plan.search_query == "什么是 RAG"
    assert plan.complexity == "simple"


def test_hello_is_chitchat():
    assert looks_like_chitchat("你好")
    assert looks_like_chitchat("谢谢！")
    assert not looks_like_chitchat("什么是 KV Cache")


def test_plan_dataclass_search_query():
    p = QueryPlan(original="口语问法", rewritten="规范化问句")
    assert p.search_query == "规范化问句"


def test_vision_only_when_query_matches_figure():
    from vision_sidecar import figure_relevant_to_query

    assert figure_relevant_to_query("这张架构图什么意思", "图5-1 Coding Agent 架构", "")
    assert not figure_relevant_to_query("你会什么安全", "图5-1 全书结构", "构建Agent")
