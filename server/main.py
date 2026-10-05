from __future__ import annotations
from typing import TypedDict , List,Annotated,Literal,Optional
from langgraph.graph import StateGraph ,START,END
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage,HumanMessage
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq
from langgraph.types import Send
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.tools.tavily_search import TavilySearchResults

import operator
load_dotenv()
import os


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand."
    )

    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 concrete, non-overlapping subpoints to cover in this section."
    )

    target_words: int = Field(
        ...,
        description="Target word count for this section (120-450)."
    )

    section_type: Literal[
        "intro",
        "core",
        "examples",
        "checklist",
        "common_mistakes",
        "conclusion"
    ] = Field(
        ...,
        description="Use 'common_mistakes' exactly once in the plan."
    )
    
# class Plan(BaseModel):
#     blog_title:str
#     audience:str=Field(...,description="who this blog is for")
#     tone:str=Field(...,description="write tone (e.g.. practical crisp)")
#     tasks:List[Task]

class Plan(BaseModel):
    blog_title:str
    audience:str
    tone:str
    blog_kind=Literal("explainer","tutorial","news_roundup","comparison","system_design")="explainer"
    constraints:List[str]=Field(default_factory=list)
    # tone:str=Field(...,description="write tone (e.g.. practical crisp)")
    tasks:List[Task]
    
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
    evidence:List[EvidenceItem]=Field(default_factory=list)
    
class State(TypedDict):
    topic:str
    mode:str
    needs_research:bool
    queries:List[str]
    avidence:List[EvidenceItem]
    plan:Optional[Plan]
    sections:Annotated[List[tuple[int,str]],operator.add]
    final:str
    
llm_groq = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=1,
    max_tokens=4000,
)

llm_nvidia = ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    api_key=NVIDIA_API_KEY , 
    temperature=1,
    top_p=1,
    max_completion_tokens=1024,
    seed=42,
)


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

def research_node(state:State)->dict:
    queries=state.get("queries",[] or [])
    max_result=6
    raw_results= List[dict]=[]
    
    for q in queries:
         raw_results.extend(_tavily_search(q,max_result=max_result))
         
    if not raw_results:
        return {"evidence":[]}

    extractor =llm_groq.with_structured_output(EvidencePack)
    pack=extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"raw_result:\n{raw_results}"),
        ]
    )
    dedup={}
    for e in pack.evidence:
        if e.url:
            dedup[e.url]=e
            
    return {"evidence":list(dedup.values())}
            
    
    
def router_node(state: dict) -> dict:
    topic = state["topic"]

    # Bind the schema for structured output to your model instance
    decider = llm_groq.with_structured_output(RouterDecision)

    # Prompt the model with system and human instructions
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}"),
        ]
    )

    # Return dictionary updates for the LangGraph state
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }

def router_next(state:State) ->str:
    return "research" if state["needs_research"] else "orchestrator"


def orchestrator_node(state: State) -> dict:
    planner = llm_groq.with_structured_output(Plan)

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke(
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
        ]
    )

    return {"plan": plan}

# def orchestrator(state:State)->dict:
#     plan=llm_groq.with_structured_output(Plan).invoke([
#         SystemMessage(
#             content=(
#                 """
#                 You are a senior technical writer and developer advocate. Your job is to plan a
# highly actionable outline for a technical blog post.

# Hard requirements:
# - Create 5–7 sections (tasks) that fit a technical blog.
# - Each section must include:
#   1) goal (1 sentence: what the reader can do/understand after the section)
#   2) 3–5 bullets that are concrete, specific, and non-overlapping
#   3) target word count (120–450)
# - Include EXACTLY ONE section with section_type='common_mistakes'.

# Make it technical (not generic):
# - Assume the reader is a developer; use correct terminology.
# - Prefer design/engineering structure: problem → intuition → approach → implementation →
#   trade-offs → testing/observability → conclusion.
# - Bullets must be actionable and testable (e.g., 'Show a minimal code snippet for X',
#   'Explain why Y fails under Z condition', 'Add a checklist for production readiness').
# - Explicitly include at least ONE of the following somewhere in the plan (as bullets):
#   * a minimal working example (MWE) or code sketch
#   * edge cases / failure modes
#   * performance/cost considerations
#   * security/privacy considerations (if relevant)
#   * debugging tips / observability (logs, metrics, traces)
# - Avoid vague bullets like 'Explain X' or 'Discuss Y'. Every bullet should state what
#   to build/compare/measure/verify.

# Ordering guidance:
# - Start with a crisp intro and problem framing.
# - Build core concepts before advanced details.
# - Include one section for common mistakes and how to avoid them.
#                 """
#             )
#         ),
#         HumanMessage(content=f"Topic: {state['topic']}")
        
#     ])
#     return {"plan":plan}

# 6) Fanout
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
                "evidence": [
                    e.model_dump()
                    for e in state.get("evidence", [])
                ],
            },
        )
        for task in state["plan"].tasks
    ]
    

def worker_node(payload: dict) -> dict:

    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [
        EvidenceItem(**e)
        for e in payload.get("evidence", [])
    ]

    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | "
            f"{(e.published_at or 'date:unknown')}".strip()
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
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n"
                    f"{evidence_text}"
                )
            ),
        ]
    ).content.strip()

    return {
        "sections": [(task.id, section_md)]
    }

# def worker(payload:dict)->dict:
#     task=payload["task"]
#     topic=payload["topic"]
#     plan=payload["plan"]
    
#     blog_title=plan.blog_title
    
#     section_md = llm_groq.invoke(
#         [
#             SystemMessage(content="""
#                           You are a senior technical writer and developer advocate. Write ONE section of a technical blog post.

# Hard constraints:
# - Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
# - Stay close to the Target words (±15%).
# - Output ONLY the section content in Markdown (no blog title H1, no extra commentary).

# Technical quality bar:
# - Be precise and implementation-oriented (developers should be able to apply it).
# - Prefer concrete details over abstractions: APIs, data structures, protocols, and exact terms.
# - When relevant, include at least one of:
#   * a small code snippet (minimal, correct, and idiomatic)
#   * a tiny example input/output
#   * a checklist of steps
#   * a diagram described in text (e.g., 'Flow: A → B → C')
# - Explain trade-offs briefly (performance, cost, complexity, reliability).
# - Call out edge cases / failure modes and what to do about them.
# - If you mention a best practice, add the 'why' in one sentence.

# Markdown style:
# - Start with a ## <Section Title> heading.
# - Use short paragraphs, bullet lists where helpful, and code fences for code.
# - Avoid fluff. Avoid marketing language.
# - If you include code, keep it focused on the bullet being addressed.
#                           """),
#            HumanMessage(
#     content=(
#         f"Blog: {plan.blog_title}\n"
#         f"Audience: {plan.audience}\n"
#         f"Tone: {plan.tone}\n"
#         f"Topic: {topic}\n\n"
#         f"Section: {task.title}\n"
#         f"Section type: {task.section_type}\n"
#         f"Goal: {task.goal}\n"
#         f"Target words: {task.target_words}\n"
#         # f"Bullets:\n{bullets_text}\n"
#     ),
# )
#         ]
#     ).content.strip()
#     return {"section":[section_md]}
    

def reducer(state: State) -> dict:

    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()

    final_md = f"# {title}\n\n{body}\n"

    # ----- save to file -----
    filename = title.lower().replace(" ", "_") + ".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    return {"final": final_md}

agent=StateGraph(State)
agent.add_node("orchestrator",orchestrator)
agent.add_node("worker",worker)
agent.add_node("reducer",reducer)

agent.add_edge(START,"orchestrator")
agent.add_conditional_edges("orchestrator",fanout,["worker"])
agent.add_edge("worker","reducer")
agent.add_edge("reducer",END)

app=agent.compile()

result=app.invoke({"topic":"write a blog on self attention","section":[]})
print("planer agent response -->>",result)













# from __future__ import annotations

# from typing import TypedDict, List, Annotated
# from langgraph.graph import StateGraph, START, END
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_core.messages import SystemMessage, HumanMessage
# from pydantic import BaseModel, Field
# from langchain_groq import ChatGroq
# from langgraph.types import Send
# from dotenv import load_dotenv
# from pathlib import Path
# import operator
# import os


# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")


# # --------------------------------------------------
# # 1. Task schema
# # --------------------------------------------------

# class Task(BaseModel):
#     id: int
#     title: str
#     brief: str = Field(..., description="what to cover")


# # --------------------------------------------------
# # 2. Plan schema
# # --------------------------------------------------

# class Plan(BaseModel):
#     blog_title: str
#     tasks: List[Task]


# # --------------------------------------------------
# # 3. LangGraph State
# # --------------------------------------------------

# class State(TypedDict):
#     topic: str
#     plan: Plan
#     sections: Annotated[List[str], operator.add]
#     final: str


# # --------------------------------------------------
# # 4. LLM
# # --------------------------------------------------

# llm_groq = ChatGroq(
#     model="openai/gpt-oss-120b",
#     api_key=GROQ_API_KEY,
#     temperature=1,
#     max_tokens=1024,
# )


# llm_nvidia = ChatNVIDIA(
#     model="nvidia/nemotron-3-super-120b-a12b",
#     api_key=NVIDIA_API_KEY,
#     temperature=1,
#     top_p=1,
#     max_completion_tokens=1024,
#     seed=42,
# )


# # --------------------------------------------------
# # 5. Orchestrator
# # --------------------------------------------------

# def orchestrator(state: State) -> dict:

#     plan = llm_groq.with_structured_output(Plan).invoke(
#         [
#             SystemMessage(
#                 content=(
#                     "Create a blog plan with 5-7 sections "
#                     "on the following topic."
#                 )
#             ),
#             HumanMessage(
#                 content=f"Topic: {state['topic']}"
#             )
#         ]
#     )

#     return {
#         "plan": plan
#     }


# # --------------------------------------------------
# # 6. Fanout
# # --------------------------------------------------

# def fanout(state: State):

#     return [
#         Send(
#             "worker",
#             {
#                 "task": task,
#                 "topic": state["topic"],
#                 "plan": state["plan"]
#             }
#         )
#         for task in state["plan"].tasks
#     ]


# # --------------------------------------------------
# # 7. Worker
# # --------------------------------------------------

# def worker(payload: dict) -> dict:

#     task = payload["task"]
#     topic = payload["topic"]
#     plan = payload["plan"]

#     blog_title = plan.blog_title

#     section_md = llm_groq.invoke(
#         [
#             SystemMessage(
#                 content="Write one clean Markdown section."
#             ),

#             HumanMessage(
#                 content=(
#                     f"Blog: {blog_title}\n"
#                     f"Topic: {topic}\n"
#                     f"Section: {task.title}\n"
#                     f"Brief: {task.brief}\n"
#                     "Return only the section content in Markdown."
#                 )
#             ),
#         ]
#     ).content.strip()

#     return {
#         "sections": [section_md]
#     }


# # --------------------------------------------------
# # 8. Reducer
# # --------------------------------------------------

# def reducer(state: State) -> dict:

#     title = state["plan"].blog_title

#     body = "\n\n".join(
#         state["sections"]
#     ).strip()

#     final_md = f"# {title}\n\n{body}\n"

#     # Save to file
#     filename = (
#         title.lower()
#         .replace(" ", "_")
#         + ".md"
#     )

#     output_path = Path(filename)

#     output_path.write_text(
#         final_md,
#         encoding="utf-8"
#     )

#     return {
#         "final": final_md
#     }


# # --------------------------------------------------
# # 9. Build Graph
# # --------------------------------------------------

# agent = StateGraph(State)

# agent.add_node(
#     "orchestrator",
#     orchestrator
# )

# agent.add_node(
#     "worker",
#     worker
# )

# agent.add_node(
#     "reducer",
#     reducer
# )


# # START
# agent.add_edge(
#     START,
#     "orchestrator"
# )


# # Orchestrator → dynamic workers
# agent.add_conditional_edges(
#     "orchestrator",
#     fanout,
#     ["worker"]
# )


# # Worker → reducer
# agent.add_edge(
#     "worker",
#     "reducer"
# )


# # Reducer → END
# agent.add_edge(
#     "reducer",
#     END
# )


# # Compile
# app = agent.compile()


# # --------------------------------------------------
# # 10. Run
# # --------------------------------------------------

# result = app.invoke(
#     {
#         "topic": "write a blog on self attention",
#         "sections": []
#     }
# )

# print(
#     "planner agent response -->",
#     result
# )