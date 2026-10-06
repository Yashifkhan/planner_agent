"""
Blog Planner UI  --  run with:  streamlit run index.py
Backend: main.py (LangGraph: router -> research -> orchestrator -> workers -> reducer)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from html import escape

import streamlit as st

st.set_page_config(
    page_title="Blog Planner",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# Styling
# --------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg:#F4F5F8; --surface:#FFFFFF; --ink:#171A22; --muted:#667085; --line:#E3E6EC;
  --accent:#3D5AFE; --accent-soft:#E8ECFF;
  --ok:#0E9F6E; --ok-soft:#DDF5EA;
  --warn:#B7791F; --warn-soft:#FBF0D9;
  --slate-soft:#EAECF1;
}
html, body, .stApp, [class*="css"]{ font-family:'Manrope',system-ui,sans-serif; }
.stApp{ background:var(--bg); color:var(--ink); }
#MainMenu, footer, header[data-testid="stHeader"]{ visibility:hidden; height:0; }
.block-container{ padding-top:1.4rem; padding-bottom:2rem; max-width:1120px; }

h1.app-title{ font-size:1.55rem; font-weight:800; letter-spacing:-.02em; margin:0; padding:0; }
p.app-sub{ color:var(--muted); font-size:.92rem; margin:.15rem 0 1rem 0; }

/* inputs + buttons */
div[data-testid="stTextInput"] input{
  background:var(--surface); border:1px solid var(--line); border-radius:10px;
  height:46px; font-size:.98rem; color:var(--ink);
}
div[data-testid="stTextInput"] input:focus{ border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-soft); }
.stButton>button, [data-testid="stFormSubmitButton"] button, .stDownloadButton>button{
  width:100%; border-radius:10px; font-weight:600; border:1px solid var(--line);
  background:var(--surface); color:var(--ink); min-height:46px;
}
.stButton>button:hover, .stDownloadButton>button:hover{ border-color:var(--accent); color:var(--accent); }
[data-testid="stFormSubmitButton"] button{ background:var(--accent); color:#fff; border-color:var(--accent); }
[data-testid="stFormSubmitButton"] button:hover{ background:#2F49DB; color:#fff; }
.ex .stButton>button{ min-height:34px; font-size:.8rem; font-weight:500; color:var(--muted); border-radius:999px; }
button:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; }

/* pipeline */
.pipe{ display:flex; gap:8px; margin:.4rem 0 .6rem 0; }
.step{ flex:1; background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:9px 12px; min-width:0; }
.step .t{ display:flex; align-items:center; gap:8px; font-weight:700; font-size:.86rem; }
.step .s{ color:var(--muted); font-size:.76rem; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-height:1.1em; }
.dot{ width:8px; height:8px; border-radius:50%; background:#C5CAD6; flex:none; }
.step.active{ border-color:var(--accent); }
.step.active .dot{ background:var(--accent); animation:pulse 1.1s ease-in-out infinite; }
.step.done .dot{ background:var(--ok); }
.step.skipped{ opacity:.55; }
.step.error .dot{ background:#D92D20; }
@keyframes pulse{ 50%{ box-shadow:0 0 0 5px var(--accent-soft); } }
@media (prefers-reduced-motion: reduce){ .step.active .dot{ animation:none; } }
@media (max-width:760px){ .pipe{ flex-wrap:wrap; } .step{ flex:1 1 45%; } }

/* stat bar */
.stats{ display:flex; flex-wrap:wrap; gap:0; background:var(--surface); border:1px solid var(--line); border-radius:10px; margin:.2rem 0 .8rem 0; }
.stat{ flex:1; padding:10px 16px; border-right:1px solid var(--line); min-width:110px; }
.stat:last-child{ border-right:none; }
.stat b{ display:block; font-size:1.15rem; font-weight:800; letter-spacing:-.01em; }
.stat span{ color:var(--muted); font-size:.76rem; }

/* chips */
.chip{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.74rem; font-weight:600; margin:0 6px 4px 0; background:var(--slate-soft); color:#3B4252; }
.chip.hybrid{ background:var(--warn-soft); color:var(--warn); }
.chip.open_book{ background:var(--ok-soft); color:var(--ok); }
.chip.closed_book{ background:var(--slate-soft); color:#3B4252; }
.chip.flag{ background:var(--accent-soft); color:var(--accent); }

/* cards */
.card{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-bottom:8px; }
.card a{ color:var(--accent); font-weight:600; text-decoration:none; }
.card a:hover{ text-decoration:underline; }
.card .u{ font-family:'JetBrains Mono',monospace; font-size:.72rem; color:var(--muted); word-break:break-all; }
.card p{ margin:.35rem 0 0 0; font-size:.86rem; color:#3B4252; }
.goal{ color:var(--muted); font-size:.9rem; margin-bottom:.4rem; }

/* tabs + expanders + blog */
button[data-baseweb="tab"]{ font-weight:600; }
div[data-testid="stExpander"]{ background:var(--surface); border:1px solid var(--line); border-radius:10px; margin-bottom:6px; }
.blog-wrap{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:8px 28px 18px 28px; }
.blog-wrap p, .blog-wrap li{ line-height:1.7; }
.empty{ background:var(--surface); border:1px dashed var(--line); border-radius:10px; padding:20px 22px; color:var(--muted); font-size:.92rem; }
.empty b{ color:var(--ink); }
.log{ font-family:'JetBrains Mono',monospace; font-size:.76rem; color:var(--muted); line-height:1.7; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------
# Constants + helpers
# --------------------------------------------------
STAGES = [
    ("router", "Route"),
    ("research", "Research"),
    ("orchestrator", "Plan"),
    ("worker", "Write"),
    ("reducer", "Merge"),
]
MODE_LABEL = {"closed_book": "Closed book", "hybrid": "Hybrid", "open_book": "Open book"}
EXAMPLES = [
    "How self-attention works in transformers",
    "RAG vs fine-tuning: when to use which",
    "Latest open-source LLM releases this week",
    "Designing a rate limiter for an API",
]


@st.cache_resource(show_spinner=False)
def load_backend():
    import main  # heavy imports + LLM client created once, not on every rerun

    return main


def stepper_html(status: dict, detail: dict) -> str:
    parts = []
    for key, label in STAGES:
        parts.append(
            f'<div class="step {status[key]}"><div class="t"><span class="dot"></span>{label}</div>'
            f'<div class="s">{escape(detail.get(key, ""))}</div></div>'
        )
    return '<div class="pipe">' + "".join(parts) + "</div>"


def words(text: str) -> int:
    return len(text.split())


def dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


# --------------------------------------------------
# Run the graph with live progress
# --------------------------------------------------
def execute(topic: str, stepper_ph, bar_ph, log_ph) -> dict:
    backend = load_backend()
    state = backend.initial_state(topic)

    status = {k: "pending" for k, _ in STAGES}
    detail = {k: "" for k, _ in STAGES}
    status["router"], detail["router"] = "active", "Choosing mode"
    logs: list[str] = []
    total = 0
    t0 = time.time()

    def paint():
        stepper_ph.markdown(stepper_html(status, detail), unsafe_allow_html=True)
        log_ph.markdown(
            '<div class="log">' + "<br>".join(escape(x) for x in logs[-6:]) + "</div>",
            unsafe_allow_html=True,
        )

    def log(msg: str):
        logs.append(f"{time.time() - t0:5.1f}s  {msg}")

    paint()
    current = "router"
    try:
        for chunk in backend.app.stream(state, stream_mode="updates"):
            for node, upd in chunk.items():
                if not upd:
                    continue
                for k, v in upd.items():
                    if k == "sections":
                        state["sections"] = list(state["sections"]) + list(v)
                    else:
                        state[k] = v

                if node == "router":
                    status["router"] = "done"
                    detail["router"] = MODE_LABEL.get(state["mode"], state["mode"])
                    log(f"router: mode={state['mode']}, {len(state['queries'])} queries")
                    if state["needs_research"]:
                        status["research"], detail["research"] = "active", f"{len(state['queries'])} queries"
                        current = "research"
                    else:
                        status["research"], detail["research"] = "skipped", "Not needed"
                        status["orchestrator"], detail["orchestrator"] = "active", "Drafting outline"
                        current = "orchestrator"

                elif node == "research":
                    status["research"] = "done"
                    detail["research"] = f"{len(state['evidence'])} sources"
                    status["orchestrator"], detail["orchestrator"] = "active", "Drafting outline"
                    log(f"research: {len(state['evidence'])} sources kept")
                    current = "orchestrator"

                elif node == "orchestrator":
                    total = len(state["plan"].tasks)
                    status["orchestrator"], detail["orchestrator"] = "done", f"{total} sections"
                    status["worker"], detail["worker"] = "active", f"0 / {total}"
                    log(f"plan: '{state['plan'].blog_title}' ({total} sections)")
                    current = "worker"

                elif node == "worker":
                    done = len(state["sections"])
                    detail["worker"] = f"{done} / {total}"
                    bar_ph.progress(min(done / max(total, 1), 1.0))
                    log(f"worker: section {state['sections'][-1][0]} written")
                    if done >= total:
                        status["worker"] = "done"
                        status["reducer"], detail["reducer"] = "active", "Assembling"
                        current = "reducer"

                elif node == "reducer":
                    status["reducer"], detail["reducer"] = "done", "Saved"
                    log(f"reducer: saved {state.get('file_path', '')}")
                paint()
    except Exception as exc:
        status[current] = "error"
        detail[current] = "Failed"
        paint()
        bar_ph.empty()
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc

    bar_ph.empty()
    return {
        "topic": topic,
        "mode": state["mode"],
        "queries": state["queries"],
        "evidence": [dump(e) for e in state["evidence"]],
        "plan": dump(state["plan"]),
        "final": state["final"],
        "file_path": state.get("file_path", ""),
        "elapsed": time.time() - t0,
    }


# --------------------------------------------------
# Result rendering
# --------------------------------------------------
def render_result(res: dict):
    plan = res["plan"]
    stats = [
        (f'<span class="chip {res["mode"]}">{MODE_LABEL.get(res["mode"], res["mode"])}</span>', "Mode"),
        (str(len(plan["tasks"])), "Sections"),
        (f'{words(res["final"]):,}', "Words"),
        (str(len(res["evidence"])), "Sources"),
        (f'{res["elapsed"]:.0f}s', "Run time"),
    ]
    st.markdown(
        '<div class="stats">'
        + "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in stats)
        + "</div>",
        unsafe_allow_html=True,
    )

    tab_blog, tab_plan, tab_ev, tab_raw = st.tabs(["Blog", "Plan", "Sources", "Raw"])

    with tab_blog:
        d1, d2, _ = st.columns([1, 1, 3])
        fname = os.path.basename(res["file_path"]) or "blog.md"
        d1.download_button("Download .md", res["final"], file_name=fname, mime="text/markdown")
        d2.download_button(
            "Download plan",
            json.dumps(plan, indent=2),
            file_name="plan.json",
            mime="application/json",
        )
        st.markdown('<div class="blog-wrap">', unsafe_allow_html=True)
        st.markdown(res["final"])
        st.markdown("</div>", unsafe_allow_html=True)
        if res["file_path"]:
            st.caption(f"Saved on disk: {res['file_path']}")

    with tab_plan:
        st.markdown(
            f'<div class="card"><b>{escape(plan["blog_title"])}</b>'
            f'<p>Audience: {escape(plan["audience"])}. Tone: {escape(plan["tone"])}.</p>'
            f'<span class="chip flag">{escape(plan["blog_kind"])}</span>'
            + "".join(f'<span class="chip">{escape(c)}</span>' for c in plan.get("constraints", []))
            + "</div>",
            unsafe_allow_html=True,
        )
        for t in plan["tasks"]:
            with st.expander(f'{t["id"]}. {t["title"]}'):
                chips = f'<span class="chip">{t["section_type"]}</span><span class="chip">{t["target_words"]} words</span>'
                if t.get("requires_code"):
                    chips += '<span class="chip flag">code</span>'
                if t.get("requires_research"):
                    chips += '<span class="chip flag">fresh info</span>'
                if t.get("requires_citations"):
                    chips += '<span class="chip flag">citations</span>'
                st.markdown(f'<div class="goal">{escape(t["goal"])}</div>{chips}', unsafe_allow_html=True)
                st.markdown("\n".join(f"- {b}" for b in t["bullets"]))

    with tab_ev:
        if res["queries"]:
            st.markdown(
                "".join(f'<span class="chip">{escape(q)}</span>' for q in res["queries"]),
                unsafe_allow_html=True,
            )
        if not res["evidence"]:
            st.markdown(
                '<div class="empty">No web sources were used. Closed-book topics are written from model knowledge only.</div>',
                unsafe_allow_html=True,
            )
        for e in res["evidence"]:
            date = e.get("published_at") or "date unknown"
            snippet = f'<p>{escape(e["snippet"])}</p>' if e.get("snippet") else ""
            st.markdown(
                f'<div class="card"><a href="{escape(e["url"])}" target="_blank" rel="noopener">{escape(e["title"] or e["url"])}</a>'
                f'<div class="u">{escape(e["url"])}</div>'
                f'<span class="chip">{escape(date)}</span>{snippet}</div>',
                unsafe_allow_html=True,
            )

    with tab_raw:
        st.code(res["final"], language="markdown")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.session_state.setdefault("history", [])
st.session_state.setdefault("result", None)

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL_NAME = os.getenv("PLANNER_MODEL", "nvidia/nemotron-3-super-120b-a12b")
OUTPUT_DIR = Path(os.getenv("PLANNER_OUTPUT_DIR", "outputs"))

with st.sidebar:
    st.markdown("**Setup**")
    for key in (NVIDIA_API_KEY, "TAVILY_API_KEY"):
        ok = bool(os.getenv(key))
        st.markdown(f"{'🟢' if ok else '🔴'} {key}")
    st.caption(f"Model: {os.getenv('PLANNER_MODEL', 'nvidia/nemotron-3-super-120b-a12b')}")
    st.divider()
    st.markdown("**History**")
    if not st.session_state["history"]:
        st.caption("Your runs this session show up here.")
    for i, h in enumerate(reversed(st.session_state["history"])):
        if st.button(h["topic"][:38], key=f"hist_{i}"):
            st.session_state["result"] = h
    if st.session_state["history"] and st.button("Clear history", key="clear_hist"):
        st.session_state["history"], st.session_state["result"] = [], None
        st.rerun()

# --------------------------------------------------
# Main layout
# --------------------------------------------------
st.markdown('<h1 class="app-title">Blog Planner</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Give a topic. It decides whether to research, plans the outline, then writes every section in parallel.</p>',
    unsafe_allow_html=True,
)

with st.form("topic_form", border=False):
    c1, c2 = st.columns([6, 1.2])
    topic_in = c1.text_input(
        "Topic",
        placeholder="What should the blog be about?",
        label_visibility="collapsed",
    )
    submitted = c2.form_submit_button("Generate")

topic_to_run = topic_in.strip() if submitted and topic_in.strip() else None
if submitted and not topic_in.strip():
    st.warning("Add a topic first.")

st.markdown('<div class="ex">', unsafe_allow_html=True)
ex_cols = st.columns(len(EXAMPLES))
for col, ex in zip(ex_cols, EXAMPLES):
    if col.button(ex, key=f"ex_{ex}"):
        topic_to_run = ex
st.markdown("</div>", unsafe_allow_html=True)

stepper_ph = st.empty()
bar_ph = st.empty()
log_ph = st.empty()

if topic_to_run:
    missing = [k for k in ("NVIDIA_API_KEY",) if not os.getenv(k)]
    if missing:
        st.error(f"Missing {', '.join(missing)}. Add it to your .env file and restart the app.")
    else:
        try:
            result = execute(topic_to_run, stepper_ph, bar_ph, log_ph)
            st.session_state["result"] = result
            st.session_state["history"].append(result)
        except Exception as exc:
            st.error(f"Run failed. {exc}")
            st.caption("Check your API keys and network, then try again. Tavily errors are skipped per query; model errors stop the run.")

res = st.session_state["result"]
if res:
    if not topic_to_run:  # loaded from history or a rerun: show a finished pipeline
        done = {k: "done" for k, _ in STAGES}
        det = {
            "router": MODE_LABEL.get(res["mode"], res["mode"]),
            "research": f'{len(res["evidence"])} sources' if res["queries"] else "Not needed",
            "orchestrator": f'{len(res["plan"]["tasks"])} sections',
            "worker": f'{len(res["plan"]["tasks"])} / {len(res["plan"]["tasks"])}',
            "reducer": "Saved",
        }
        if not res["queries"]:
            done["research"] = "skipped"
        stepper_ph.markdown(stepper_html(done, det), unsafe_allow_html=True)
    st.caption(f'Topic: {res["topic"]}')
    render_result(res)
elif not topic_to_run:
    st.markdown(
        '<div class="empty"><b>What you get</b><br>'
        "A routed run (closed book, hybrid or open book), a sectioned plan with goals and bullets, "
        "the sources it used, and a finished Markdown blog you can download.</div>",
        unsafe_allow_html=True,
    )