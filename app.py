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

/* ── Answer card (tier 1 — high confidence Q&A) ── */
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
.confidence-bar { height: 3px; border-radius: 2px; background: #22c55e; }

/* ── Result mode badges ── */
.result-mode-badge {
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 0.2rem 0.55rem; border-radius: 99px; margin-bottom: 0.9rem;
}
.badge-qa       { background: rgba(34,197,94,0.12);   color: #15803d; }
.badge-qa-low   { background: rgba(245,158,11,0.12);  color: #b45309; }
.badge-grounded { background: rgba(139,92,246,0.12);  color: #6d28d9; }
.badge-llm      { background: rgba(59,130,246,0.12);  color: #1d4ed8; }
.badge-none     { background: rgba(156,163,175,0.15); color: #6b7280; }

/* ── Score pills ── */
.score-pill {
    display: inline-block; font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.04em; padding: 0.15rem 0.45rem;
    border-radius: 99px; vertical-align: middle;
}
.score-high { background: rgba(34,197,94,0.12);  color: #15803d; }
.score-mid  { background: rgba(245,158,11,0.12); color: #b45309; }
.score-low  { background: rgba(239,68,68,0.10);  color: #b91c1c; }

/* ── Candidate cards (tier 2 — low confidence list) ── */
.candidate-card {
    padding: 1rem 1.25rem;
    border-radius: 10px;
    border: 1px solid rgba(0,0,0,0.08);
    margin-bottom: 0.75rem;
}
.candidate-rank {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
    color: #9ca3af; margin-bottom: 0.35rem;
}
.candidate-question {
    font-size: 0.9rem; font-weight: 600;
    margin-bottom: 0.45rem; line-height: 1.4;
}
.candidate-answer { font-size: 0.875rem; line-height: 1.55; opacity: 0.8; }
.candidate-meta   { font-size: 0.75rem; opacity: 0.4; margin-top: 0.5rem; }

/* ── Alt cards (tier 1 alternatives expander) ── */
.alt-card {
    padding: 0.85rem 1.1rem;
    border-radius: 9px;
    border: 1px solid rgba(0,0,0,0.07);
    background: rgba(0,0,0,0.01);
    margin-bottom: 0.6rem;
}
.alt-question { font-size: 0.85rem; font-weight: 600; margin-bottom: 0.35rem; line-height: 1.4; }
.alt-answer   { font-size: 0.82rem; line-height: 1.5; opacity: 0.65; }
.alt-meta     { font-size: 0.72rem; opacity: 0.35; margin-top: 0.4rem; }

/* ── Info cards (tier 3 & 4) ── */
.found-card {
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #8b5cf6;
    background: rgba(139,92,246,0.05);
    margin-bottom: 1rem;
}
.no-match-card {
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    border: 1px dashed rgba(0,0,0,0.15);
    background: rgba(0,0,0,0.015);
    margin-bottom: 1rem;
}

/* ── LLM answer cards ── */
.llm-grounded-card {
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #8b5cf6;
    background: rgba(139,92,246,0.05);
    margin-bottom: 1rem;
}
.llm-card {
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    background: rgba(59,130,246,0.05);
    margin-bottom: 1rem;
}
.llm-disclaimer {
    font-size: 0.8rem; opacity: 0.55;
    margin-top: 0.75rem; font-style: italic;
}

/* ── Retrieval trail ── */
.retrieval-trail {
    display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
    font-size: 0.75rem; opacity: 0.5; margin-bottom: 1rem;
}
.trail-step { display: flex; align-items: center; gap: 0.3rem; }
.trail-sep  { opacity: 0.3; }
.trail-pass { color: #16a34a; }
.trail-fail { color: #dc2626; }
.trail-warn { color: #d97706; }
.trail-skip { color: #9ca3af; }

/* ── Misc ── */
.section-heading {
    font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; opacity: 0.4; margin-bottom: 0.75rem;
}
.empty-state { text-align: center; padding: 4rem 1rem; opacity: 0.35; }

/* ── Dark mode ── */
@media (prefers-color-scheme: dark) {
    .answer-card        { border-color: rgba(255,255,255,0.1); }
    .confidence-bar-wrap{ background: rgba(255,255,255,0.1); }
    .candidate-card     { border-color: rgba(255,255,255,0.09); }
    .alt-card           { border-color: rgba(255,255,255,0.07); background: rgba(255,255,255,0.01); }
    .no-match-card      { border-color: rgba(255,255,255,0.12); background: rgba(255,255,255,0.02); }
    .trail-pass { color: #4ade80; }
    .trail-fail { color: #f87171; }
    .trail-warn { color: #fbbf24; }
    .badge-qa       { background: rgba(34,197,94,0.15);   color: #4ade80; }
    .badge-qa-low   { background: rgba(245,158,11,0.15);  color: #fbbf24; }
    .badge-grounded { background: rgba(139,92,246,0.15);  color: #a78bfa; }
    .badge-llm      { background: rgba(59,130,246,0.15);  color: #60a5fa; }
    .score-high { background: rgba(34,197,94,0.15);  color: #4ade80; }
    .score-mid  { background: rgba(245,158,11,0.15); color: #fbbf24; }
    .score-low  { background: rgba(239,68,68,0.12);  color: #f87171; }
}
</style>
""",
    unsafe_allow_html=True,
)

HIGH_CONF_THRESHOLD = 0.75


@st.cache_resource(show_spinner=False)
def get_retriever():
    try:
        return HybridRetriever()
    except FileNotFoundError:
        return None


with st.spinner("Loading knowledge base..."):
    retriever = get_retriever()

if retriever is None:
    st.error(
        "Knowledge base indexes not found. "
        "Add documents to `data/raw_docs/` or CSVs to `data/raw/`, "
        "then run the ingestion pipeline from the Admin dashboard to build the indexes."
    )
    st.stop()

# ── Session state ────────────────────────────────────────────────────────────
for _key, _default in [
    ("last_query", ""),
    ("last_result", None),
    ("llm_answer", None),   # result dict from LLMFallback.generate()
    ("llm_type", None),     # "grounded" or "general"
    ("search_error", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Header ───────────────────────────────────────────────────────────────────
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
        st.session_state.llm_answer = None
        st.session_state.llm_type = None
        st.session_state.last_query = q
        st.session_state.search_error = None
        try:
            st.session_state.last_result = retriever.search(q, llm_enabled=False)
        except Exception as e:
            st.session_state.last_result = None
            st.session_state.search_error = str(e)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _qa_score(result: dict):
    qa = result.get("qa_result", {})
    if qa.get("ok") and qa.get("best"):
        return qa["best"].get("score")
    if qa.get("best_candidate"):
        return qa["best_candidate"].get("score")
    return None


def _chunk_score(result: dict):
    cr = result.get("chunk_result", {})
    if cr.get("best"):
        return cr["best"].get("score")
    return None


def _score_pill(score: float) -> str:
    if score >= HIGH_CONF_THRESHOLD:
        cls = "score-high"
    elif score >= 0.55:
        cls = "score-mid"
    else:
        cls = "score-low"
    return f'<span class="score-pill {cls}">{score:.2f}</span>'


def _build_chunk_context(chunk_result: dict, max_chars: int = 3000) -> str:
    results = chunk_result.get("results", [])
    if not results and chunk_result.get("best"):
        results = [chunk_result["best"]]
    parts = []
    total = 0
    for r in results:
        text = r.get("text", "")
        source = r.get("source_file", "")
        page = r.get("page", "")
        snippet = f"[{source}, Page {page}]\n{text}"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts)


def _render_trail(result: dict, llm_triggered: bool = False):
    mode = result.get("mode", "")
    qa_sc = _qa_score(result)
    ch_sc = _chunk_score(result)

    qa_ok  = mode == "qa_answer"
    qa_low = qa_ok and qa_sc is not None and qa_sc < HIGH_CONF_THRESHOLD
    ch_ok  = mode == "chunk_fallback"

    def step(icon, label, cls):
        return f'<span class="trail-step {cls}">{icon} {label}</span>'

    parts = []

    if qa_ok and not qa_low:
        lbl = f"Q&amp;A match ({qa_sc:.2f})" if qa_sc else "Q&amp;A match"
        parts.append(step("✓", lbl, "trail-pass"))
    elif qa_ok and qa_low:
        lbl = f"Q&amp;A low confidence ({qa_sc:.2f})" if qa_sc else "Q&amp;A low confidence"
        parts.append(step("~", lbl, "trail-warn"))
    elif qa_sc is not None:
        parts.append(step("✗", f"Q&amp;A no match ({qa_sc:.2f})", "trail-fail"))
    else:
        parts.append(step("✗", "Q&amp;A no match", "trail-fail"))

    # Chunk step — only when Q&A fully failed (not low-confidence Q&A)
    if not qa_ok:
        parts.append('<span class="trail-sep">→</span>')
        if ch_ok:
            lbl = f"Document search ({ch_sc:.2f})" if ch_sc else "Document search"
            parts.append(step("✓", lbl, "trail-pass"))
        elif ch_sc is not None:
            parts.append(step("✗", f"Document search ({ch_sc:.2f})", "trail-fail"))
        else:
            parts.append(step("✗", "Document search", "trail-fail"))

    if llm_triggered:
        parts.append('<span class="trail-sep">→</span>')
        parts.append(step("✓", "AI answer", "trail-pass"))

    st.markdown(
        f'<div class="retrieval-trail">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def _render_qa_high(best: dict, alternatives: list):
    score        = best.get("score", 0)
    answer_esc   = _html.escape(str(best.get("answer", "")))
    matched_esc  = _html.escape(str(best.get("matched_question", "")))
    bar_pct      = min(int(score * 100), 100)

    st.markdown(
        f"""
<div class="answer-card">
  <span class="result-mode-badge badge-qa">Answered from university sources</span>
  <div class="confidence-bar-wrap">
    <div class="confidence-bar" style="width:{bar_pct}%"></div>
  </div>
  <div class="matched-question">Matched to: \u201c{matched_esc}\u201d</div>
  <p>{answer_esc}</p>
</div>""",
        unsafe_allow_html=True,
    )

    if alternatives:
        n = len(alternatives)
        label = f"{n} other possible answer{'s' if n != 1 else ''}"
        with st.expander(label, expanded=False):
            st.markdown(
                '<div class="section-heading">Other matches — check if one is more relevant to your question</div>',
                unsafe_allow_html=True,
            )
            for alt in alternatives:
                alt_score = alt.get("score", 0)
                alt_q   = _html.escape(str(alt.get("matched_question", "")))
                alt_a   = _html.escape(str(alt.get("answer", "")))
                st.markdown(
                    f"""
<div class="alt-card">
  <div class="alt-question">{alt_q} {_score_pill(alt_score)}</div>
  <div class="alt-answer">{alt_a}</div>
</div>""",
                    unsafe_allow_html=True,
                )


def _render_qa_low(candidates: list):
    n = len(candidates)
    st.markdown(
        f"""
<div class="no-match-card">
  <span class="result-mode-badge badge-qa-low">Low confidence matches</span>
  <p style="margin:0;opacity:0.7;">
    We found {n} possible answer{'s' if n != 1 else ''} drawn from university sources,
    but none are a strong match for your question. Check whether any of the results
    below address what you asked — if not, generate an AI answer at the bottom.
  </p>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">Possible matches — drawn from university sources</div>',
        unsafe_allow_html=True,
    )

    for i, cand in enumerate(candidates, 1):
        score   = cand.get("score", 0)
        q_esc   = _html.escape(str(cand.get("matched_question", "")))
        a_esc   = _html.escape(str(cand.get("answer", "")))
        st.markdown(
            f"""
<div class="candidate-card">
  <div class="candidate-rank">#{i} match</div>
  <div class="candidate-question">{q_esc} {_score_pill(score)}</div>
  <div class="candidate-answer">{a_esc}</div>
</div>""",
            unsafe_allow_html=True,
        )


def _render_chunk_found(result: dict):
    chunk_result = result.get("chunk_result", {})
    best         = chunk_result.get("best", {}) or {}
    n_chunks     = len(chunk_result.get("results", [chunk_result.get("best")] if chunk_result.get("best") else []))
    file_esc     = _html.escape(str(best.get("source_file", "")))

    source_hint = f" (including <strong>{file_esc}</strong>)" if file_esc else ""
    st.markdown(
        f"""
<div class="found-card">
  <span class="result-mode-badge badge-grounded">Relevant documents found</span>
  <p style="margin:0;opacity:0.75;">
    The knowledge base didn't contain a direct answer, but we found
    {n_chunks} relevant passage{'s' if n_chunks != 1 else ''} in your uploaded
    documents{source_hint}. Generate an answer below — it will be grounded in
    those document passages, not invented.
  </p>
</div>""",
        unsafe_allow_html=True,
    )


def _render_no_match():
    st.markdown(
        """
<div class="no-match-card">
  <span class="result-mode-badge badge-none">No match found</span>
  <p style="margin:0;opacity:0.65;">
    Neither the Q&amp;A knowledge base nor the uploaded documents contained
    relevant information for your question. You can generate a general AI answer
    below, though it won't be drawn from official university sources.
  </p>
</div>""",
        unsafe_allow_html=True,
    )


def _render_llm_answer(llm_result: dict, llm_type: str):
    st.divider()

    if not llm_result.get("ok"):
        reason = llm_result.get("reason", "Unknown error")
        st.error(
            f"AI generation failed: {reason}\n\n"
            "This may be due to an API issue. Please try again in a moment."
        )
        return

    answer_esc = _html.escape(llm_result["answer"])

    if llm_type == "grounded":
        st.markdown(
            f"""
<div class="llm-grounded-card">
  <span class="result-mode-badge badge-grounded">Answer from your documents</span>
  <p>{answer_esc}</p>
  <div class="llm-disclaimer">
    Generated using passages from uploaded university documents.
    Always verify important information at lboro.ac.uk
  </div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
<div class="llm-card">
  <span class="result-mode-badge badge-llm">AI-Generated Answer</span>
  <p>{answer_esc}</p>
  <div class="llm-disclaimer">
    This answer was generated by an AI model and has not been verified
    against official university sources. Always confirm important information
    at lboro.ac.uk
  </div>
</div>""",
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

else:
    mode          = result.get("mode", "")
    llm_triggered = st.session_state.llm_answer is not None

    _render_trail(result, llm_triggered=llm_triggered)

    # ── Tier 1 & 2: Q&A returned a result ────────────────────────────────────
    if mode == "qa_answer":
        best         = result["qa_result"]["best"]
        score        = best.get("score", 0)
        alternatives = result["qa_result"].get("alternatives", [])[1:]

        if score >= HIGH_CONF_THRESHOLD:
            # Tier 1 — high confidence
            _render_qa_high(best, alternatives)
            if not llm_triggered:
                if st.button(
                    "Not satisfied? Get an AI-generated answer \u2192",
                    key="ai_btn_qa",
                ):
                    with st.spinner("Generating AI answer..."):
                        try:
                            st.session_state.llm_answer = retriever.llm_fallback.generate(query)
                            st.session_state.llm_type = "general"
                        except Exception as e:
                            st.session_state.llm_answer = {"ok": False, "reason": str(e)}
                            st.session_state.llm_type = "general"
                    st.rerun()
        else:
            # Tier 2 — low confidence, show all candidates
            all_candidates = result["qa_result"].get("alternatives", [])
            _render_qa_low(all_candidates)
            if not llm_triggered:
                if st.button(
                    "None of these answer your question? Get an AI-generated answer \u2192",
                    key="ai_btn_low",
                ):
                    with st.spinner("Generating AI answer..."):
                        try:
                            st.session_state.llm_answer = retriever.llm_fallback.generate(query)
                            st.session_state.llm_type = "general"
                        except Exception as e:
                            st.session_state.llm_answer = {"ok": False, "reason": str(e)}
                            st.session_state.llm_type = "general"
                    st.rerun()

    # ── Tier 3: chunk found, no direct Q&A answer ─────────────────────────────
    elif mode == "chunk_fallback":
        _render_chunk_found(result)
        if not llm_triggered:
            if st.button(
                "Generate answer from your documents \u2192",
                key="ai_btn_chunk",
            ):
                with st.spinner("Generating answer from your documents..."):
                    try:
                        context = _build_chunk_context(result["chunk_result"])
                        st.session_state.llm_answer = retriever.llm_fallback.generate(
                            query, context=context
                        )
                        st.session_state.llm_type = "grounded"
                    except Exception as e:
                        st.session_state.llm_answer = {"ok": False, "reason": str(e)}
                        st.session_state.llm_type = "grounded"
                st.rerun()

    # ── Tier 4: nothing found ─────────────────────────────────────────────────
    else:
        _render_no_match()
        if not llm_triggered:
            if st.button(
                "Generate a general AI answer \u2192",
                key="ai_btn_none",
            ):
                with st.spinner("Generating AI answer..."):
                    try:
                        st.session_state.llm_answer = retriever.llm_fallback.generate(query)
                        st.session_state.llm_type = "general"
                    except Exception as e:
                        st.session_state.llm_answer = {"ok": False, "reason": str(e)}
                        st.session_state.llm_type = "general"
                st.rerun()

    # ── LLM answer (rendered below whichever tier triggered it) ───────────────
    if llm_triggered:
        _render_llm_answer(st.session_state.llm_answer, st.session_state.llm_type)
