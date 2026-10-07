from __future__ import annotations

import operator
import os
import re
from pathlib import Path
from typing import Annotated, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
import json
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL_NAME = os.getenv("PLANNER_MODEL", "nvidia/nemotron-3-super-120b-a12b")
OUTPUT_DIR = Path(os.getenv("PLANNER_OUTPUT_DIR", "outputs"))
MAX_QUERIES = int(os.getenv("PLANNER_MAX_QUERIES", "5"))

# print("NVIDIA_API_KEY",NVIDIA_API_KEY)
# --------------------------------------------------
# 1) Schemas
# --------------------------------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3-6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(
        ..., description="Target word count for this section (120-550)."
    )
    section_type: Literal[
        "intro",
        "core",
        "examples",
        "checklist",
        "common_mistakes",
        "conclusion",
    ] = Field(..., description="Role of this section in the blog.")

    # FIX: prompts mention these flags but the schema never had them
    requires_code: bool = Field(
        default=False, description="True if this section needs a code snippet."
    )
    requires_research: bool = Field(
        default=False, description="True if this section uses fresh evidence."
    )
    requires_citations: bool = Field(
        default=False, description="True if outside-world claims must be cited."
    )


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ]
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # keep if Tavily provides; DO NOT rely on it
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]  # FIX: was misspelled "avidence"
    plan: Optional[Plan]
    sections: Annotated[List[tuple[int, str]], operator.add]
    final: str
    file_path: str


# --------------------------------------------------
# 2) LLM
# --------------------------------------------------
llm = ChatNVIDIA(
    model=MODEL_NAME,
    api_key=NVIDIA_API_KEY,
    temperature=0.7,
    top_p=1,
    max_completion_tokens=4096,
    timeout=180,        # default 60 tha
    seed=42,
)


def _extract_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models
    text = re.sub(r"```(?:json)?", "", text)                    # markdown fences
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return text[start : end + 1]


def structured_invoke(schema: type[BaseModel], messages: list, retries: int = 3):
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    msgs = [
        *messages,
        HumanMessage(
            content=(
                "Respond with ONLY one valid JSON object that matches this JSON Schema. "
                "No markdown, no commentary.\n\nSchema:\n" + schema_json
            )
        ),
    ]
    last_err = None
    for _ in range(retries):
        out = llm.invoke(msgs)
        try:
            return schema.model_validate_json(_extract_json(out.content))
        except (ValueError, ValidationError) as e:
            last_err = e
            msgs += [
                AIMessage(content=out.content),
                HumanMessage(content=f"Invalid output: {e}\nReturn the corrected JSON only."),
            ]
    raise RuntimeError(f"Structured output failed after {retries} tries: {last_err}")

# --------------------------------------------------
# 3) Prompts
# --------------------------------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:

- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).

- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.

- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""

WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a "## <Section Title>" heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
    - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
    - For each event claim, attach a source as a Markdown link: ([Source](URL)).
    - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
    - For outside-world claims, cite Evidence URLs the same way.
    - Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""


# --------------------------------------------------
# 4) Research
# --------------------------------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


def research_node(state: State) -> dict:
    queries = state.get("queries", [])

    MAX_RESULTS_PER_QUERY = 2
    MAX_CONTENT_CHARS = 800
    MAX_TOTAL_RESULTS = 10

    raw_results: List[dict] = []
    for q in queries:
        try:
            raw_results.extend(_tavily_search(q, max_results=MAX_RESULTS_PER_QUERY))
        except Exception as exc:  # one bad query should not kill the run
            print(f"[research] query failed: {q!r} -> {exc}")

    raw_results = raw_results[:MAX_TOTAL_RESULTS]

    # FIX: normalized dicts carry "snippet", not "content" (old code always got "")
    research_context = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "published_at": r.get("published_at"),
            "content": (r.get("snippet") or "")[:MAX_CONTENT_CHARS],
        }
        for r in raw_results
        if r.get("url")
    ]

    if not research_context:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    # pack = extractor.invoke(
    #     [
    #         SystemMessage(content=RESEARCH_SYSTEM),
    #         HumanMessage(content=f"Research results:\n{research_context}"),
    #     ]
    # )
    pack = structured_invoke(
    EvidencePack,
    [SystemMessage(content=RESEARCH_SYSTEM), HumanMessage(content=f"Research results:\n{research_context}")],
)

    dedup = {e.url: e for e in pack.evidence if e.url}
    return {"evidence": list(dedup.values())}


# --------------------------------------------------
# 5) Router + Orchestrator
# --------------------------------------------------
def router_node(state: State) -> dict:
    decision = structured_invoke(
        RouterDecision,
        [SystemMessage(content=ROUTER_SYSTEM), HumanMessage(content=f"Topic: {state['topic']}")],
    )

    # dedupe (order safe) + hard cap
    queries = list(dict.fromkeys(q.strip() for q in decision.queries if q.strip()))[:MAX_QUERIES]

    return {
        "needs_research": decision.needs_research and bool(queries),
        "mode": decision.mode,
        "queries": queries,
    }


def router_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = structured_invoke(
    Plan,
    [
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n"
                f"Mode: {mode}\n\n"
                f"Evidence (ONLY use for fresh claims; may be empty):\n"
                f"{[e.model_dump() for e in evidence[:16]]}"
            )
        ),
    ],
)
    return {"plan": plan}


# --------------------------------------------------
# 6) Fan-out -> Worker -> Reducer
# --------------------------------------------------
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
            for e in evidence[:20]
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Section type: {task.section_type}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"Bullets:\n{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n"
                    f"{evidence_text}"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


def _slug(text: str) -> str:
    # FIX: blog titles contain ":" "?" "/" etc. which break file names on Windows
    return re.sub(r"[^\w\-]+", "-", text).strip("-")[:80] or "blog"


def reducer_node(state: State) -> dict:
    plan = state["plan"]
    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{_slug(plan.blog_title)}.md"
    path.write_text(final_md, encoding="utf-8")

    return {"final": final_md, "file_path": str(path)}


# --------------------------------------------------
# 7) Graph
# --------------------------------------------------
agent = StateGraph(State)

agent.add_node("router", router_node)
agent.add_node("research", research_node)
agent.add_node("orchestrator", orchestrator_node)
agent.add_node("worker", worker_node)
agent.add_node("reducer", reducer_node)

agent.add_edge(START, "router")
agent.add_conditional_edges(
    "router", router_next, {"research": "research", "orchestrator": "orchestrator"}
)
agent.add_edge("research", "orchestrator")
agent.add_conditional_edges("orchestrator", fanout, ["worker"])
agent.add_edge("worker", "reducer")
agent.add_edge("reducer", END)

app = agent.compile()


def initial_state(topic: str) -> dict:
    return {
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "sections": [],
        "final": "",
        "file_path": "",
    }


def run(topic: str) -> dict:
    return app.invoke(initial_state(topic))


# FIX: this used to run on import, so Streamlit would fire the whole agent
# on every rerun. Now it only runs via `python main.py`.
if __name__ == "__main__":
    out = run("write a blog about the gpt 6 astra")
    print("planner agent response -->>", out["final"])