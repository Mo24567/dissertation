import html as _html
import streamlit as st
from src.retrieval.hybrid_query import HybridRetriever

st.set_page_config(
    page_title="Loughborough University Student Assistant",
    page_icon="\U0001f393",
    layout="centered",
)

st.markdown(
    """
<style>
.block-container { max-width: 720px; padding-top: 3rem; }

.app-title {
    font-size: 1.8rem; font-weight: 600;
    letter-spacing: -0.01em; margin-bottom: 0.25rem;
}
.app-subtitle {
    font-size: 0.95rem; opacity: 0.5; margin-bottom: 2.5rem;
}
.answer-card {
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    border: 0.5px solid rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.source-citation {
    font-size: 0.8rem; opacity: 0.5; margin-top: 0.75rem;
}
.matched-question {
    font-size: 0.8rem; opacity: 0.45; margin-bottom: 0.75rem;
    font-style: italic;
}
.confidence-bar-wrap {
    height: 3px; border-radius: 2px;
    background: rgba(0,0,0,0.08); margin-bottom: 0.9rem;
}
.confidence-bar {
    height: 3px; border-radius: 2px; background: #22c55e;
}
.confidence-low .confidence-bar { background: #f59e0b; }
@media (prefers-color-scheme: dark) {
    .confidence-bar-wrap { background: rgba(255,255,255,0.1); }
}
.chunk-card {
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #f59e0b;
    background: rgba(245,158,11,0.05);
    margin-bottom: 1rem;
}
.chunk-label {
    font-size: 0.8rem; font-weight: 600;
    color: #b45309; margin-bottom: 0.5rem;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.llm-card {
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    background: rgba(59,130,246,0.05);
    margin-bottom: 1rem;
}
.llm-label {
    font-size: 0.8rem; font-weight: 600;
    color: #1d4ed8; margin-bottom: 0.5rem;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.llm-disclaimer {
    font-size: 0.8rem; opacity: 0.55;
    margin-top: 0.75rem; font-style: italic;
}
.ai-link {
    font-size: 0.85rem; opacity: 0.6; margin-top: 1rem;
}
.empty-state {
    text-align: center; padding: 4rem 1rem; opacity: 0.35;
}
.retrieval-trail {
    display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
    font-size: 0.75rem; opacity: 0.5; margin-bottom: 1rem;
}
.trail-step { display: flex; align-items: center; gap: 0.3rem; }
.trail-sep { opacity: 0.3; }
.trail-pass { color: #16a34a; }
.trail-fail { color: #dc2626; }
.trail-skip { color: #9ca3af; }
.result-mode-badge {
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 0.2rem 0.55rem; border-radius: 99px; margin-bottom: 0.9rem;
}
.badge-qa   { background: rgba(34,197,94,0.12);  color: #15803d; }
.badge-chunk{ background: rgba(245,158,11,0.12); color: #b45309; }
.badge-llm  { background: rgba(59,130,246,0.12); color: #1d4ed8; }
.badge-none { background: rgba(156,163,175,0.15);color: #6b7280; }
@media (prefers-color-scheme: dark) {
    .answer-card { border-color: rgba(255,255,255,0.1); }
    .trail-pass { color: #4ade80; }
    .trail-fail { color: #f87171; }
    .badge-qa   { background: rgba(34,197,94,0.15);  color: #4ade80; }
    .badge-chunk{ background: rgba(245,158,11,0.15); color: #fbbf24; }
    .badge-llm  { background: rgba(59,130,246,0.15); color: #60a5fa; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_retriever():
    try:
        return HybridRetriever()
    except FileNotFoundError as e:
        return None


retriever = get_retriever()

if retriever is None:
    st.error(
        "Knowledge base indexes not found. "
        "Add documents to `data/raw_docs/` or CSVs to `data/raw/`, "
        "then run the ingestion pipeline from the Admin dashboard to build the indexes."
    )
    st.stop()

# Initialise session state
for _key, _default in [
    ("show_llm", False),
    ("last_query", ""),
    ("last_result", None),
    ("llm_override", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">Loughborough Student Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask a question about fees, accommodation, attendance,<br>'
    "assessments or university regulations.</div>",
    unsafe_allow_html=True,
)

# ── Search form ──────────────────────────────────────────────────────────────
with st.form("search_form"):
    query_input = st.text_input(
        "Question",
        placeholder="e.g. What happens if I miss too many lectures?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Search", use_container_width=True)

# ── Handle submission ────────────────────────────────────────────────────────
if submitted:
    q = query_input.strip()
    if not q:
        st.warning("Please enter a question.")
    elif len(q) < 3:
        st.warning("Please enter a longer question.")
    elif len(query_input) > 500:
        st.warning("Please keep your question under 500 characters.")
    else:
        st.session_state.show_llm = False
        st.session_state.llm_override = None
        st.session_state.last_query = q
        st.session_state.search_error = None
        try:
            st.session_state.last_result = retriever.search(q)
        except Exception as e:
            st.session_state.last_result = None
            st.session_state.search_error = str(e)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _qa_score(result: dict):
    """Best Q&A score from result, or None."""
    qa = result.get("qa_result", {})
    if qa.get("ok") and qa.get("best"):
        return qa["best"].get("score")
    if qa.get("best_candidate"):
        return qa["best_candidate"].get("score")
    return None

def _chunk_score(result: dict):
    """Best chunk score from result, or None."""
    cr = result.get("chunk_result", {})
    if cr.get("ok") and cr.get("best"):
        return cr["best"].get("score")
    if cr.get("best"):
        return cr["best"].get("score")
    return None

def _render_trail(result: dict, show_llm_requested: bool = False):
    """Render a small retrieval trail showing what was tried."""
    mode = result.get("mode", "")
    qa_sc = _qa_score(result)
    ch_sc = _chunk_score(result)

    qa_ok  = mode == "qa_answer"
    ch_ok  = mode in ("chunk_fallback", "llm_fallback", "no_answer") and result.get("chunk_result", {}).get("ok")
    llm_ok = mode == "llm_fallback" or show_llm_requested

    def step(icon, label, cls):
        return f'<span class="trail-step {cls}">{icon} {label}</span>'

    parts = []

    # Q&A step
    if qa_ok:
        qa_txt = f"Q&amp;A match ({qa_sc:.2f})" if qa_sc else "Q&amp;A match"
        parts.append(step("✓", qa_txt, "trail-pass"))
    elif qa_sc is not None:
        parts.append(step("✗", f"Q&amp;A no match ({qa_sc:.2f})", "trail-fail"))
    else:
        parts.append(step("✗", "Q&amp;A no match", "trail-fail"))

    # Chunk step — only show if Q&A failed
    if not qa_ok:
        parts.append('<span class="trail-sep">→</span>')
        if ch_ok:
            ch_txt = f"Document search ({ch_sc:.2f})" if ch_sc else "Document search"
            parts.append(step("✓", ch_txt, "trail-pass"))
        elif ch_sc is not None:
            parts.append(step("✗", f"Document search no match ({ch_sc:.2f})", "trail-fail"))
        else:
            parts.append(step("✗", "Document search no match", "trail-fail"))

    # LLM step — only show if both retrieval layers failed
    if not qa_ok and not ch_ok:
        parts.append('<span class="trail-sep">→</span>')
        if llm_ok:
            parts.append(step("✓", "AI answer", "trail-pass"))
        else:
            parts.append(step("—", "AI answer not requested", "trail-skip"))

    html_trail = "".join(parts)
    st.markdown(
        f'<div class="retrieval-trail">{html_trail}</div>',
        unsafe_allow_html=True,
    )

# ── Render results ───────────────────────────────────────────────────────────
result = st.session_state.last_result
query  = st.session_state.last_query

if st.session_state.get("search_error"):
    st.error(
        f"Something went wrong while searching: {st.session_state.search_error}\n\n"
        "Please try again or rephrase your question."
    )

elif result is None:
    st.markdown(
        '<div class="empty-state"><p>Enter a question above to get started.</p></div>',
        unsafe_allow_html=True,
    )

elif st.session_state.show_llm or result.get("mode") == "llm_fallback":
    # ── LLM answer ───────────────────────────────────────────────────────────
    _render_trail(result, show_llm_requested=True)

    auto_llm = result.get("mode") == "llm_fallback"
    if auto_llm:
        llm_result = result.get("llm_result", {})
    else:
        if st.session_state.llm_override is None:
            with st.spinner("Generating AI answer..."):
                try:
                    st.session_state.llm_override = retriever.llm_fallback.generate(query)
                except Exception as e:
                    st.session_state.llm_override = {"ok": False, "reason": str(e)}
        llm_result = st.session_state.llm_override

    if llm_result.get("ok"):
        answer_escaped = _html.escape(llm_result["answer"])
        st.markdown(
            f"""
<div class="llm-card">
  <span class="result-mode-badge badge-llm">AI-Generated Answer</span>
  <p>{answer_escaped}</p>
  <div class="llm-disclaimer">
    This answer was generated by an AI model and has not been verified
    against official university sources. Always confirm important information
    at lboro.ac.uk
  </div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        reason = llm_result.get("reason", "Unknown error")
        st.error(
            f"AI generation failed: {reason}\n\n"
            "This may be due to an API issue. Please try again in a moment."
        )

elif result.get("mode") == "qa_answer":
    # ── Q&A answer ────────────────────────────────────────────────────────────
    _render_trail(result)
    best = result["qa_result"]["best"]
    answer_escaped      = _html.escape(str(best.get("answer", "")))
    source_escaped      = _html.escape(str(best.get("source", "")))
    source_file_escaped = _html.escape(str(best.get("source_file", "")))
    matched_q_escaped   = _html.escape(str(best.get("matched_question", "")))
    page  = best.get("page", "")
    score = best.get("score", 0)

    bar_pct   = min(int(score * 100), 100)
    low_conf  = score < 0.75
    wrap_cls  = "confidence-bar-wrap confidence-low" if low_conf else "confidence-bar-wrap"
    conf_lbl  = "Moderate confidence" if low_conf else "High confidence"

    st.markdown(
        f"""
<div class="answer-card">
  <span class="result-mode-badge badge-qa">Answered from knowledge base</span>
  <div class="{wrap_cls}"><div class="confidence-bar" style="width:{bar_pct}%"></div></div>
  <div class="matched-question">Matched to: \u201c{matched_q_escaped}\u201d</div>
  <p>{answer_escaped}</p>
  <div class="source-citation">
    {conf_lbl} \u00b7 Score {score:.2f} \u00b7 {source_escaped} \u2014 {source_file_escaped}, Page {page}
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    if low_conf:
        st.caption(
            "\u26a0\ufe0f Moderate confidence — the matched question above may not perfectly match "
            "what you asked. If the answer looks wrong, try rephrasing or use the AI option below."
        )

    st.markdown('<div class="ai-link">', unsafe_allow_html=True)
    if st.button("Not satisfied? Get an AI-generated answer \u2192", key="ai_btn_qa"):
        st.session_state.show_llm = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

elif result.get("mode") == "chunk_fallback":
    # ── Chunk fallback ────────────────────────────────────────────────────────
    _render_trail(result)
    chunk_best          = result["chunk_result"]["best"]
    text_escaped        = _html.escape(str(chunk_best.get("text", "")))
    source_file_escaped = _html.escape(str(chunk_best.get("source_file", "")))
    page                = chunk_best.get("page", "")

    st.markdown(
        f"""
<div class="chunk-card">
  <span class="result-mode-badge badge-chunk">Relevant Document Passage</span>
  <div class="chunk-label">No direct answer found \u2014 showing closest match from source documents</div>
  <p>{text_escaped}</p>
  <div class="source-citation">{source_file_escaped}, Page {page}</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.caption(
        "The knowledge base did not contain a direct answer to your question. "
        "The passage above is the most relevant section found in the uploaded documents. "
        "For a fuller answer, try the AI option below."
    )

    st.markdown('<div class="ai-link">', unsafe_allow_html=True)
    if st.button("Get an AI-generated answer instead \u2192", key="ai_btn_chunk"):
        st.session_state.show_llm = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # ── No answer ─────────────────────────────────────────────────────────────
    _render_trail(result, show_llm_requested=False)

    st.warning(
        "No relevant information was found for your question in the knowledge base or documents."
    )
    st.caption(
        "Both the Q&A knowledge base and the document search were checked but nothing "
        "closely matched your question. Try rephrasing, or use the AI option below for "
        "a general answer."
    )

    st.markdown('<div class="ai-link">', unsafe_allow_html=True)
    if st.button("Get an AI-generated answer instead \u2192", key="ai_btn_noanswer"):
        st.session_state.show_llm = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
