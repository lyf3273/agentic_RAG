"""查询规划：复杂度分流 + 口语改写 / HyDE / Step-back / 多 Query。

思路参考小林 coding《RAG 查询改写》（MIT）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

PLAN_SYSTEM = """你是 RAG 查询规划器。只输出一个 JSON 对象，不要 Markdown。

根据用户问题选择策略（可多选）：
- colloquial：口语、指代、口语化、句子残缺，需要改写成适合检索的规范短句
- hyde：用户问法和知识库书面体差异大，需要先写一段假想知识库短文再做向量检索
- step_back：问题过细（具体产品参数、某版本配置、某一条命令），知识库更可能只有原理/背景，需要上升一个抽象层次
- multi_query：只从一个角度问，需要 2～3 个互补检索问法（原因/做法/风险等）
- none：已经适合直接检索

复杂度：
- chitchat：寒暄打招呼、谢谢再见、闲聊、不需要查知识库（如「你好」「在吗」）
- simple：单点事实、定义、一个知识点能答完，必须检索
- complex：完整流程、多领域拼接、条件分支、必须多步查工具

字段：
{
  "complexity": "chitchat" 或 "simple" 或 "complex",
  "strategies": ["colloquial"],
  "rewritten": "规范化检索问句",
  "step_back": "上升一层的背景/原理问题；没有则空字符串。例：Qdrant 的 HNSW ef 设多少 → HNSW 索引参数调优原则是什么",
  "hyde": "80-150字假想知识库短文，仅当 strategies 含 hyde，否则空",
  "multi_queries": ["最多3条互补问法"]
}

step_back 只要问题偏细就尽量给，即使 strategies 里暂时没选它。
"""


@dataclass
class QueryPlan:
    original: str
    complexity: str = "simple"
    strategies: list[str] = field(default_factory=list)
    rewritten: str = ""
    step_back: str = ""
    hyde: str = ""
    multi_queries: list[str] = field(default_factory=list)

    @property
    def search_query(self) -> str:
        return (self.rewritten or self.original).strip()


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        raise


_CHITCHAT_RE = re.compile(
    r"^(你好|您好|嗨|哈喽|在吗|在嘛|谢谢|感谢|再见|拜拜|早上好|晚上好|中午好"
    r"|hi|hello|hey|thanks|thank you|bye)[！!。.?？~\s]*$",
    re.I,
)


def looks_like_chitchat(question: str) -> bool:
    return bool(_CHITCHAT_RE.match((question or "").strip()))


def plan_query(question: str, llm_complete, history: list[dict] | None = None) -> QueryPlan:
    """用 LLM 判断寒暄 / 简单检索 / 复杂 ReAct，并产出改写策略。"""
    if looks_like_chitchat(question):
        return QueryPlan(original=question, complexity="chitchat", rewritten=question)
    hist = ""
    if history:
        recent = history[-3:]
        lines = []
        for turn in recent:
            lines.append(f"Q: {turn.get('question', '')}")
            lines.append(f"A: {(turn.get('answer') or '')[:200]}")
        hist = "最近对话：\n" + "\n".join(lines) + "\n\n"
    user = f"{hist}当前问题：{question}"
    try:
        raw = llm_complete(PLAN_SYSTEM, user)
        data = _extract_json(raw)
    except Exception:
        return QueryPlan(original=question, rewritten=question)

    strategies = data.get("strategies") or []
    if isinstance(strategies, str):
        strategies = [strategies]
    multi = data.get("multi_queries") or []
    if isinstance(multi, str):
        multi = [multi]
    complexity = str(data.get("complexity") or "simple").lower()
    if complexity not in {"simple", "complex", "chitchat"}:
        complexity = "simple"
    return QueryPlan(
        original=question,
        complexity=complexity,
        strategies=[str(s).lower() for s in strategies if s],
        rewritten=str(data.get("rewritten") or question).strip(),
        step_back=str(data.get("step_back") or "").strip(),
        hyde=str(data.get("hyde") or "").strip(),
        multi_queries=[str(q).strip() for q in multi if str(q).strip()][:3],
    )
