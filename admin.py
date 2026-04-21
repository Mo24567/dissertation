import hashlib
import io
import re
from datetime import datetime
from pathlib import Path

try:
    import plotly.express as px
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

import numpy as np
import faiss
import pandas as pd
import streamlit as st
from dotenv import find_dotenv, set_key

from src.retrieval.model_cache import get_model
from src.utils.config import settings

st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="\U0001f527",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _compute_kb_coverage_curve():
    """Compute query coverage (%) at increasing KB sizes using comparison_queries.csv."""
    qa_path = Path("data/processed/qa_dataset_clean.csv")
    eval_path = Path("evaluation/comparison_queries.csv")

    if not qa_path.exists() or not eval_path.exists():
        return None

    qa_df = pd.read_csv(qa_path)
    eval_df = pd.read_csv(eval_path)

    if qa_df.empty or eval_df.empty or "question" not in qa_df.columns or "query" not in eval_df.columns:
        return None

    questions = qa_df["question"].dropna().tolist()
    queries = eval_df["query"].dropna().tolist()

    model = get_model()
    qa_emb = model.encode(questions, normalize_embeddings=True).astype(np.float32)
    q_emb = model.encode(queries, normalize_embeddings=True).astype(np.float32)

    threshold = settings.similarity_threshold
    n_total = len(questions)
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(n_total)

    rows = []
    for pct in range(10, 101, 10):
        size = max(1, int(n_total * pct / 100))
        subset = qa_emb[shuffled[:size]]
        dim = subset.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(subset)
        scores, _ = idx.search(q_emb, 1)
        matched = int((scores[:, 0] >= threshold).sum())
        rows.append({
            "Q&A pairs in knowledge base": size,
            "Query coverage (%)": round(matched / len(queries) * 100, 1),
        })

    return pd.DataFrame(rows)


# ── Password protection ──────────────────────────────────────────────────────

def check_password() -> bool:
    if st.session_state.get("admin_authenticated"):
        return True
    st.markdown("### Admin Login")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if password == settings.admin_password:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


if not check_password():
    st.stop()

# ── Cached retrievers ─────────────────────────────────────────────────────────

@st.cache_resource
def get_admin_retriever():
    try:
        from src.retrieval.hybrid_query import HybridRetriever
        return HybridRetriever()
    except FileNotFoundError:
        return None


@st.cache_resource
def get_keyword_retriever():
    try:
        from src.retrieval.keyword_retriever import KeywordRetriever
        return KeywordRetriever()
    except Exception:
        return None


@st.cache_resource
def get_qa_retriever():
    try:
        from src.retrieval.query import QARetriever
        return QARetriever()
    except Exception:
        return None


# ── Helper utilities ──────────────────────────────────────────────────────────

EXCLUDED_PATH = Path("data/processed/excluded_qas.csv")


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", str(q)).lower().strip()


def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _queue_message(key: str, message: str, level: str = "success") -> None:
    """Store a confirmation message for a specific UI location."""
    st.session_state[f"_msg_{key}"] = (message, level)


def _show_message(key: str) -> None:
    """Display and clear a queued message for this location."""
    entry = st.session_state.pop(f"_msg_{key}", None)
    if entry:
        message, level = entry
        getattr(st, level)(message)


def _load_csv_safe(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        try:
            return pd.read_csv(p, engine="python", on_bad_lines="skip")
        except Exception:
            pass
    return pd.DataFrame()


def _mtime(path: str) -> float:
    p = Path(path)
    return p.stat().st_mtime if p.exists() else 0.0


def _mtime_label(path: str) -> str:
    p = Path(path)
    if p.exists():
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return "Not built yet"


def _row_count(path: str) -> int:
    return len(_load_csv_safe(path))


def _cleanup_orphaned_qas(current_pdf_names: set, drafts_path: Path, approved_path: Path) -> None:
    """Remove Q&A rows whose source_file is no longer in current_pdf_names."""
    for path in (drafts_path, approved_path):
        if not path.exists():
            continue
        df = _load_csv_safe(str(path))
        if df.empty or "source_file" not in df.columns:
            continue
        # Keep rows with no source_file (manually uploaded CSVs) or a current PDF
        mask = df["source_file"].isna() | (df["source_file"] == "") | df["source_file"].isin(current_pdf_names)
        cleaned = df[mask]
        if len(cleaned) < len(df):
            cleaned.to_csv(path, index=False)


# ── Batch tool helpers ────────────────────────────────────────────────────────

def _questions_match(a: str, b: str) -> bool:
    """Case-insensitive, whitespace-normalised equality check."""
    def norm(s):
        return re.sub(r"\s+", " ", str(s)).lower().strip()
    return norm(a) == norm(b)


def _log_feed_result(query: str, result: dict) -> None:
    """Write one evaluation-feed query to the dashboard query log."""
    from src.retrieval.query_logger import log_query, log_unanswered
    mode = result.get("mode", "no_answer")
    qa_res = result.get("qa_result", {}) or {}
    ch_res = result.get("chunk_result", {}) or {}
    qa_score, chunk_score, source, page = None, None, "", ""
    if mode == "qa_answer":
        best = qa_res.get("best", {})
        qa_score = best.get("score")
        source = best.get("source", "")
        page = best.get("page", "")
    elif mode == "chunk_fallback":
        qa_score = (qa_res.get("best_candidate") or {}).get("score")
        best_ch = ch_res.get("best", {})
        chunk_score = best_ch.get("score")
        source = best_ch.get("source", "")
        page = best_ch.get("page", "")
    else:
        qa_score = (qa_res.get("best_candidate") or {}).get("score")
        chunk_score = (ch_res.get("best") or {}).get("score")
    log_query(
        query=query,
        mode=mode,
        qa_score=qa_score,
        chunk_score=chunk_score,
        llm_used=(mode == "llm_fallback"),
        qa_threshold=settings.similarity_threshold,
        chunk_threshold=settings.chunk_similarity_threshold,
        source=source,
        page=page,
    )
    if mode in ("no_answer", "none"):
        log_unanswered(
            query=query,
            best_qa_score=qa_score,
            best_chunk_score=chunk_score,
            best_qa_match=qa_res.get("best_candidate"),
        )


# ── Page: Testing ─────────────────────────────────────────────────────────────

def page_testing():
    st.header("Testing")

    # ── Testing tools ─────────────────────────────────────────────────────────
    with st.expander("Test a Single Query"):
        st.caption("Run a query against the live assistant to see which layer handled it and what scores were returned. Useful for checking threshold settings.")
        retriever = get_admin_retriever()
        eval_query = st.text_input("Query", key="eval_single_query")
        if st.button("Run", key="eval_single_run") and eval_query.strip():
            if retriever is None:
                st.error("Retriever not available — build the indexes first.")
            else:
                result = retriever.search(
                    eval_query.strip(),
                    qa_threshold=settings.similarity_threshold,
                    chunk_threshold=settings.chunk_similarity_threshold,
                )
                st.write(f"**Handled by:** `{result.get('mode')}`")
                with st.expander("Q&A match details", expanded=True):
                    qa = result.get("qa_result", {})
                    st.write(f"**Matched:** {qa.get('ok')}  |  **Reason:** {qa.get('reason', 'n/a')}")
                    if qa.get("alternatives"):
                        rows = [{"score": r["score"], "matched_question": r["matched_question"][:80], "answer": r["answer"][:80]} for r in qa["alternatives"]]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    elif qa.get("best_candidate"):
                        bc = qa["best_candidate"]
                        st.write(f"Closest match score: {bc['score']:.4f} — {bc.get('matched_question', '')[:80]}")
                with st.expander("Passage search details"):
                    cr = result.get("chunk_result", {})
                    st.write(f"**Matched:** {cr.get('ok')}  |  **Reason:** {cr.get('reason', 'n/a')}")
                    if cr.get("results"):
                        rows = [{"score": r["score"], "text": r["text"][:100]} for r in cr["results"]]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    elif cr.get("best"):
                        st.write(f"Closest passage score: {cr['best']['score']:.4f}")
                with st.expander("Fallback details"):
                    llm = result.get("llm_result")
                    if llm:
                        st.write(f"**Used:** {llm.get('ok')}  |  **Model:** {llm.get('model')}  |  **Time:** {llm.get('latency_ms')} ms")
                        st.write(llm.get("answer", ""))
                    else:
                        st.write("Fallback was not used.")

    # ── Evaluation Feed ───────────────────────────────────────────────────────
    with st.expander("Evaluation Feed"):
        st.caption(
            "Upload a CSV with a `query` column. Fires each query through the live system "
            "and logs the results to the dashboard — use this to populate the Evaluation page "
            "without typing queries one by one. Uses live threshold settings."
        )
        retriever = get_admin_retriever()
        uploaded_feed = st.file_uploader("Upload queries CSV", type=["csv"], key="feed_upload")
        if uploaded_feed:
            try:
                feed_df = pd.read_csv(uploaded_feed)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                feed_df = pd.DataFrame()
            if not feed_df.empty:
                if "query" not in feed_df.columns:
                    st.error("CSV must have a `query` column.")
                else:
                    st.caption(f"{len(feed_df)} queries loaded.")
                    st.dataframe(feed_df[["query"]].head(5), use_container_width=True)
                    if st.button("Run Feed", key="feed_run", type="primary"):
                        if retriever is None:
                            st.error("Retriever not available — build the indexes first.")
                        else:
                            counts = {"qa_answer": 0, "chunk_fallback": 0, "llm_fallback": 0, "no_answer": 0}
                            prog = st.progress(0)
                            for i, row in enumerate(feed_df.itertuples(), start=1):
                                q = str(getattr(row, "query", "")).strip()
                                if not q:
                                    continue
                                res = retriever.search(q)
                                mode = res.get("mode", "no_answer")
                                counts[mode] = counts.get(mode, 0) + 1
                                prog.progress(i / len(feed_df))
                            st.success(f"Done — {len(feed_df)} queries logged to the dashboard.")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Q&A Match", counts.get("qa_answer", 0))
                            c2.metric("Passage Search", counts.get("chunk_fallback", 0))
                            c3.metric("Generated Answer", counts.get("llm_fallback", 0))
                            c4.metric("Unanswered", counts.get("no_answer", 0))

    # ── BM25 vs Semantic Comparison ───────────────────────────────────────────
    with st.expander("BM25 vs Semantic Comparison"):
        st.caption(
            "Upload a CSV with `query` and `expected_question` columns. "
            "Runs each query through keyword search (BM25) and semantic Q&A search only — "
            "no passage search or generated answer fallback. Measures accuracy of each method head-to-head. "
            "`expected_question` should be the exact question text from your knowledge base."
        )
        kw_retriever = get_keyword_retriever()
        qa_retriever = get_qa_retriever()
        uploaded_cmp = st.file_uploader("Upload comparison CSV", type=["csv"], key="cmp_upload")
        if uploaded_cmp:
            try:
                cmp_df = pd.read_csv(uploaded_cmp)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                cmp_df = pd.DataFrame()
            if not cmp_df.empty:
                missing = [c for c in ["query", "expected_question"] if c not in cmp_df.columns]
                if missing:
                    st.error(f"CSV is missing column(s): {', '.join(missing)}")
                else:
                    st.caption(f"{len(cmp_df)} queries loaded.")
                    st.dataframe(cmp_df.head(5), use_container_width=True)
                    if st.button("Run Comparison", key="cmp_run", type="primary"):
                        if kw_retriever is None or qa_retriever is None:
                            st.error("Retrievers not available — build the indexes first.")
                        else:
                            rows = []
                            prog = st.progress(0)
                            for i, row in enumerate(cmp_df.itertuples(), start=1):
                                q = str(getattr(row, "query", "")).strip()
                                expected = str(getattr(row, "expected_question", "")).strip()
                                bm25_res = kw_retriever.search(q)
                                bm25_match = bm25_res.get("matched_question", "")
                                bm25_correct = "✓" if _questions_match(bm25_match, expected) else "✗"
                                sem_res = qa_retriever.answer(q, threshold=settings.similarity_threshold)
                                sem_match = ""
                                sem_score = None
                                if sem_res.get("ok"):
                                    sem_match = sem_res["best"].get("matched_question", "")
                                    sem_score = sem_res["best"].get("score")
                                elif sem_res.get("best_candidate"):
                                    sem_match = sem_res["best_candidate"].get("matched_question", "")
                                    sem_score = sem_res["best_candidate"].get("score")
                                sem_correct = "✓" if sem_res.get("ok") and _questions_match(sem_match, expected) else "✗"
                                rows.append({
                                    "query": q[:70],
                                    "expected": expected[:70],
                                    "BM25 matched": bm25_match[:70],
                                    "BM25": bm25_correct,
                                    "BM25 score": f"{bm25_res.get('score', 0):.3f}",
                                    "Semantic matched": sem_match[:70],
                                    "Semantic": sem_correct,
                                    "Semantic score": f"{sem_score:.3f}" if sem_score is not None else "—",
                                })
                                prog.progress(i / len(cmp_df))
                            st.session_state.cmp_results = rows

        if st.session_state.get("cmp_results"):
            rows = st.session_state.cmp_results
            results_df = pd.DataFrame(rows)
            st.dataframe(results_df, use_container_width=True)
            total = len(rows)
            bm25_acc = sum(1 for r in rows if r["BM25"] == "✓") / total * 100
            sem_acc = sum(1 for r in rows if r["Semantic"] == "✓") / total * 100
            m1, m2 = st.columns(2)
            m1.metric("BM25 Accuracy", f"{bm25_acc:.1f}%")
            m2.metric("Semantic Accuracy", f"{sem_acc:.1f}%")
            st.download_button(
                "Export comparison CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="bm25_vs_semantic.csv",
                mime="text/csv",
            )

    # ── Full System Capture ───────────────────────────────────────────────────
    with st.expander("Full System Capture"):
        st.caption(
            "Upload a CSV with a `query` column. Runs each query through the complete pipeline "
            "and captures what each layer returned. Export the result and fill in the "
            "`Correct?` column yourself — this becomes your manual correctness sheet."
        )
        retriever = get_admin_retriever()
        include_llm = st.checkbox(
            "Include generated answer fallback (uses OpenAI API)",
            value=False,
            key="capture_llm",
            help="If unchecked, queries that reach the fallback layer will show as Unanswered instead.",
        )
        uploaded_cap = st.file_uploader("Upload queries CSV", type=["csv"], key="cap_upload")
        if uploaded_cap:
            try:
                cap_df = pd.read_csv(uploaded_cap)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                cap_df = pd.DataFrame()
            if not cap_df.empty:
                if "query" not in cap_df.columns:
                    st.error("CSV must have a `query` column.")
                else:
                    st.caption(f"{len(cap_df)} queries loaded.")
                    st.dataframe(cap_df[["query"]].head(5), use_container_width=True)
                    if st.button("Run Capture", key="cap_run", type="primary"):
                        if retriever is None:
                            st.error("Retriever not available — build the indexes first.")
                        else:
                            _mode_labels = {
                                "qa_answer": "Q&A Match",
                                "chunk_fallback": "Passage Search",
                                "llm_fallback": "Generated Answer",
                                "no_answer": "Unanswered",
                            }
                            rows = []
                            prog = st.progress(0)
                            for i, row in enumerate(cap_df.itertuples(), start=1):
                                q = str(getattr(row, "query", "")).strip()
                                if not q:
                                    continue
                                res = retriever.search(q, llm_enabled=include_llm)
                                mode = res.get("mode", "no_answer")
                                qa_answer, passage_text, ai_answer = "", "", ""
                                if mode == "qa_answer":
                                    best = res.get("qa_result", {}).get("best", {})
                                    qa_answer = best.get("answer", "")
                                elif mode == "chunk_fallback":
                                    best = res.get("chunk_result", {}).get("best", {})
                                    passage_text = best.get("text", "")
                                elif mode == "llm_fallback":
                                    ai_answer = (res.get("llm_result") or {}).get("answer", "")
                                rows.append({
                                    "query": q,
                                    "layer": _mode_labels.get(mode, mode),
                                    "Q&A answer": qa_answer,
                                    "Passage returned": passage_text,
                                    "Generated answer": ai_answer,
                                    "Correct?": "",
                                })
                                prog.progress(i / len(cap_df))
                            st.session_state.cap_results = rows

        if st.session_state.get("cap_results"):
            rows = st.session_state.cap_results
            cap_results_df = pd.DataFrame(rows)
            st.dataframe(cap_results_df, use_container_width=True)
            total = len(rows)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Q&A Match", sum(1 for r in rows if r["layer"] == "Q&A Match"))
            c2.metric("Passage Search", sum(1 for r in rows if r["layer"] == "Passage Search"))
            c3.metric("Generated Answer", sum(1 for r in rows if r["layer"] == "Generated Answer"))
            c4.metric("Unanswered", sum(1 for r in rows if r["layer"] == "Unanswered"))
            st.download_button(
                "Export for manual annotation",
                data=cap_results_df.to_csv(index=False).encode("utf-8"),
                file_name="full_system_capture.csv",
                mime="text/csv",
            )

    # ── Query Log ─────────────────────────────────────────────────────────────
    log = _load_csv_safe("data/processed/query_log.csv")

    st.divider()
    st.subheader("Query Log")

    if not log.empty:
        if "timestamp" in log.columns:
            log["timestamp"] = pd.to_datetime(log["timestamp"], errors="coerce")
        if "qa_score" in log.columns:
            log["qa_score"] = pd.to_numeric(log["qa_score"], errors="coerce")
        if "chunk_score" in log.columns:
            log["chunk_score"] = pd.to_numeric(log["chunk_score"], errors="coerce")

        tab_all, tab_unanswered = st.tabs(["All Queries", "Unanswered"])

        with tab_all:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                modes = ["All"] + sorted(log["mode"].dropna().unique().tolist())
                mode_filter = st.selectbox("Mode", modes, key="log_mode_filter")
            with fc2:
                if log["timestamp"].notna().any():
                    min_date = log["timestamp"].min()
                    max_date = log["timestamp"].max()
                    if pd.notna(min_date) and pd.notna(max_date):
                        date_range = st.date_input(
                            "Date range",
                            value=(min_date.date(), max_date.date()),
                            key="log_date_range",
                        )
                    else:
                        date_range = None
                else:
                    date_range = None
            with fc3:
                min_score = st.number_input(
                    "Min QA Score", min_value=0.0, max_value=1.0,
                    value=0.0, step=0.01, key="log_min_score",
                )

            filtered = log.copy()
            if mode_filter != "All":
                filtered = filtered[filtered["mode"] == mode_filter]
            if date_range and len(date_range) == 2 and "timestamp" in filtered.columns:
                start_dt = pd.Timestamp(date_range[0])
                end_dt = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
                filtered = filtered[
                    (filtered["timestamp"] >= start_dt) & (filtered["timestamp"] < end_dt)
                ]
            if "qa_score" in filtered.columns and min_score > 0:
                filtered = filtered[filtered["qa_score"].fillna(0) >= min_score]

            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Shown", len(filtered))
            avg_qa_f = filtered["qa_score"].dropna().mean() if "qa_score" in filtered.columns else None
            avg_chunk_f = filtered["chunk_score"].dropna().mean() if "chunk_score" in filtered.columns else None
            sm2.metric("Avg QA Score", f"{avg_qa_f:.4f}" if avg_qa_f is not None and not pd.isna(avg_qa_f) else "n/a")
            sm3.metric("Avg Chunk Score", f"{avg_chunk_f:.4f}" if avg_chunk_f is not None and not pd.isna(avg_chunk_f) else "n/a")

            st.dataframe(filtered, use_container_width=True)
            st.download_button(
                "Export filtered log",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="query_log_filtered.csv",
                mime="text/csv",
            )

        with tab_unanswered:
            unanswered = _load_csv_safe("data/processed/unanswered_log.csv")
            if unanswered.empty:
                st.info("No unanswered queries yet.")
            else:
                st.caption("These queries failed all retrieval layers. Use them to identify gaps in the knowledge base.")
                st.dataframe(unanswered, use_container_width=True)
                st.download_button(
                    "Export unanswered log",
                    data=unanswered.to_csv(index=False).encode("utf-8"),
                    file_name="unanswered_log.csv",
                    mime="text/csv",
                )
    else:
        st.info("No query data yet. The log will populate once users start asking questions.")


# ── Inline Q&A review (used inside PDFs tab) ─────────────────────────────────

def _review_drafts_inline():
    drafts_path = Path("data/processed/draft_qas.csv")
    approved_path = Path("data/raw/approved_qas.csv")

    if not drafts_path.exists():
        st.info("No suggestions yet — generate them above.")
        return

    df = _load_csv_safe(str(drafts_path))
    if df.empty:
        st.info("No suggestions found.")
        return

    if "review_status" not in df.columns:
        df["review_status"] = "pending"

    reviewable = df[df["review_status"] != "skipped"]
    total = len(reviewable)
    reviewed = int((reviewable["review_status"].isin(["approved", "rejected"])).sum())
    pending_count = int((reviewable["review_status"] == "pending").sum())

    st.write(f"**{reviewed} of {total} reviewed — {pending_count} pending**")
    st.progress(reviewed / total if total else 0)

    tabs = st.tabs(["Pending", "Approved", "Rejected", "All"])

    # ── Pending ───────────────────────────────────────────────────────────────
    with tabs[0]:
        _show_message("review_action")
        pending_df = df[df["review_status"] == "pending"].copy()

        if pending_df.empty:
            st.success("All suggestions have been reviewed.")
        else:
            search_q = st.text_input("Search questions", placeholder="Filter by keyword…", key="review_search")
            if search_q:
                pending_df = pending_df[
                    pending_df["question"].str.contains(search_q, case=False, na=False)
                ]

            st.caption(
                f"{len(pending_df)} suggestion(s) awaiting review. "
                "Select items using checkboxes and approve or reject in bulk, or use the buttons on each card."
            )

            select_all = st.checkbox("Select all", key="select_all_pending")
            if select_all:
                selected = list(pending_df.index)
            else:
                selected = [idx for idx in pending_df.index if st.session_state.get(f"chk_{idx}", False)]

            ba1, ba2 = st.columns(2)
            with ba1:
                if st.button(
                    f"Approve selected ({len(selected)})",
                    disabled=len(selected) == 0,
                    type="primary",
                    use_container_width=True,
                    key="bulk_approve",
                ):
                    new_rows = []
                    for idx in selected:
                        row = df.loc[idx]
                        df.loc[idx, "review_status"] = "approved"
                        new_rows.append({
                            "question": row.get("question", ""),
                            "answer": row.get("answer", ""),
                            "source": row.get("source", ""),
                            "source_file": row.get("source_file", ""),
                            "page": row.get("page", ""),
                        })
                    df.to_csv(drafts_path, index=False)
                    if new_rows:
                        new_df = pd.DataFrame(new_rows)
                        if approved_path.exists():
                            new_df.to_csv(approved_path, mode="a", header=False, index=False)
                        else:
                            new_df.to_csv(approved_path, index=False)
                    for idx in selected:
                        st.session_state.pop(f"chk_{idx}", None)
                    st.session_state.pop("select_all_pending", None)
                    _queue_message("review_action", f"{len(selected)} Q&A(s) approved.")
                    st.rerun()

            with ba2:
                if st.button(
                    f"Reject selected ({len(selected)})",
                    disabled=len(selected) == 0,
                    use_container_width=True,
                    key="bulk_reject",
                ):
                    for idx in selected:
                        df.loc[idx, "review_status"] = "rejected"
                    df.to_csv(drafts_path, index=False)
                    for idx in selected:
                        st.session_state.pop(f"chk_{idx}", None)
                    st.session_state.pop("select_all_pending", None)
                    _queue_message("review_action", f"{len(selected)} Q&A(s) rejected.", level="warning")
                    st.rerun()

            st.divider()

            for row_idx, row in pending_df.iterrows():
                col_chk, col_content = st.columns([0.05, 0.95])
                with col_chk:
                    st.checkbox(
                        "Select",
                        key=f"chk_{row_idx}",
                        value=select_all or st.session_state.get(f"chk_{row_idx}", False),
                        label_visibility="hidden",
                    )
                with col_content:
                    st.markdown(f"**Q: {row.get('question', '')}**")
                    st.write(f"A: {row.get('answer', '')}")
                    st.caption(
                        f"Source: {row.get('source_file', '') or 'n/a'} | "
                        f"Page: {row.get('page', '') or 'n/a'}"
                    )
                    ib1, ib2, ib3 = st.columns(3)
                    with ib1:
                        if st.button("Approve", key=f"approve_{row_idx}", type="primary"):
                            df.loc[row_idx, "review_status"] = "approved"
                            df.to_csv(drafts_path, index=False)
                            approved_row = pd.DataFrame([{
                                "question": row.get("question", ""),
                                "answer": row.get("answer", ""),
                                "source": row.get("source", ""),
                                "source_file": row.get("source_file", ""),
                                "page": row.get("page", ""),
                            }])
                            if approved_path.exists():
                                approved_row.to_csv(approved_path, mode="a", header=False, index=False)
                            else:
                                approved_row.to_csv(approved_path, index=False)
                            _queue_message("review_action", "Q&A approved.")
                            st.rerun()
                    with ib2:
                        if st.button("Reject", key=f"reject_{row_idx}"):
                            df.loc[row_idx, "review_status"] = "rejected"
                            df.to_csv(drafts_path, index=False)
                            _queue_message("review_action", "Q&A rejected.", level="warning")
                            st.rerun()
                    with ib3:
                        if st.button("Skip", key=f"skip_{row_idx}"):
                            df.loc[row_idx, "review_status"] = "skipped"
                            df.to_csv(drafts_path, index=False)
                            _queue_message("review_action", "Q&A skipped.", level="info")
                            st.rerun()
                st.divider()

    # ── Approved ──────────────────────────────────────────────────────────────
    with tabs[1]:
        approved = df[df["review_status"] == "approved"]
        if approved.empty:
            st.info("No approved Q&As yet.")
        else:
            st.caption(f"{len(approved)} pair(s) approved.")
            if EXCLUDED_PATH.exists():
                ex = _load_csv_safe(str(EXCLUDED_PATH))
                if not ex.empty and "question_norm" in ex.columns:
                    excl_norms = set(ex["question_norm"])
                    excluded_count = int(approved["question"].map(_norm_q).isin(excl_norms).sum())
                    if excluded_count > 0:
                        st.warning(
                            f"{excluded_count} approved Q&A(s) are currently excluded. "
                            "Go to **Knowledge Base → Deleted Q&As** to restore them."
                        )
            display_cols = [c for c in approved.columns if c != "is_useful"]
            st.dataframe(approved[display_cols], use_container_width=True)

    # ── Rejected ──────────────────────────────────────────────────────────────
    with tabs[2]:
        rejected = df[df["review_status"] == "rejected"]
        if rejected.empty:
            st.info("No rejected Q&As yet.")
        else:
            st.caption(f"{len(rejected)} rejected — click Approve to move a pair back to approved.")
            for row_idx, row in rejected.iterrows():
                c1, c2 = st.columns([0.9, 0.1])
                with c1:
                    st.markdown(f"**Q: {row.get('question', '')}**")
                    st.write(f"A: {row.get('answer', '')}")
                    st.caption(f"Source: {row.get('source_file', '') or 'n/a'} | Page: {row.get('page', '') or 'n/a'}")
                with c2:
                    if st.button("Approve", key=f"re_approve_{row_idx}", type="primary"):
                        df.loc[row_idx, "review_status"] = "approved"
                        df.to_csv(drafts_path, index=False)
                        approved_row = pd.DataFrame([{
                            "question": row.get("question", ""),
                            "answer": row.get("answer", ""),
                            "source": row.get("source", ""),
                            "source_file": row.get("source_file", ""),
                            "page": row.get("page", ""),
                        }])
                        if approved_path.exists():
                            approved_row.to_csv(approved_path, mode="a", header=False, index=False)
                        else:
                            approved_row.to_csv(approved_path, index=False)
                        _queue_message("review_action", "Q&A moved back to approved.")
                        st.rerun()
                st.divider()

    # ── All ───────────────────────────────────────────────────────────────────
    with tabs[3]:
        display_cols = [c for c in df.columns if c != "is_useful"]
        st.dataframe(df[display_cols], use_container_width=True)


# ── Page: Documents ───────────────────────────────────────────────────────────

def page_documents():
    st.header("Documents")

    if "pdf_upload_counter" not in st.session_state:
        st.session_state.pdf_upload_counter = 0
    if "csv_upload_counter" not in st.session_state:
        st.session_state.csv_upload_counter = 0
    if "qa_editor_counter" not in st.session_state:
        st.session_state.qa_editor_counter = 0

    raw_docs = Path("data/raw_docs")
    chunks_path = Path("data/processed/document_chunks.csv")
    chunk_index_path = Path("data/processed/chunk_index.faiss")
    drafts_path = Path("data/processed/draft_qas.csv")
    approved_path = Path("data/raw/approved_qas.csv")

    # ── Shared state computed once ────────────────────────────────────────────
    pdf_files = sorted(raw_docs.glob("*.pdf"))
    chunk_count = _row_count(str(chunks_path))

    if pdf_files:
        newest_pdf_mtime = max(p.stat().st_mtime for p in pdf_files)
        mtime_ok = chunks_path.exists() and _mtime(str(chunks_path)) >= newest_pdf_mtime
        if mtime_ok and chunk_count > 0:
            df_chk = _load_csv_safe(str(chunks_path))
            if not df_chk.empty and "source_file" in df_chk.columns:
                source_set_ok = set(df_chk["source_file"].dropna().unique()) == set(p.name for p in pdf_files)
            else:
                source_set_ok = False
        else:
            source_set_ok = False
        chunks_up_to_date = mtime_ok and source_set_ok
    else:
        chunks_up_to_date = False

    chunk_index_up_to_date = (
        chunk_index_path.exists()
        and chunk_count > 0
        and _mtime(str(chunk_index_path)) >= _mtime(str(chunks_path))
    )

    processed_ids: set = set()
    if drafts_path.exists():
        df_drafts = _load_csv_safe(str(drafts_path))
        if not df_drafts.empty and "chunk_id" in df_drafts.columns:
            processed_ids = set(df_drafts["chunk_id"].dropna())

    chunks_df = _load_csv_safe(str(chunks_path))
    unprocessed = 0
    if not chunks_df.empty and "chunk_id" in chunks_df.columns:
        unprocessed = int((~chunks_df["chunk_id"].isin(processed_ids)).sum())

    pending_drafts = 0
    if drafts_path.exists():
        df_pd = _load_csv_safe(str(drafts_path))
        if not df_pd.empty and "review_status" in df_pd.columns:
            pending_drafts = int((df_pd["review_status"] == "pending").sum())

    tab_pdf, tab_csv = st.tabs(["PDFs", "Q&A Files"])

    # ── Tab 1: PDFs ───────────────────────────────────────────────────────────
    with tab_pdf:
        st.subheader("PDF Documents")

        if st.session_state.get("pdf_upload_error"):
            st.error(st.session_state.pop("pdf_upload_error"))
        _show_message("pdf_action")

        uploaded_pdfs = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"pdf_uploader_{st.session_state.pdf_upload_counter}",
        )
        if uploaded_pdfs:
            errors = [f.name for f in uploaded_pdfs if not f.name.lower().endswith(".pdf")]
            if errors:
                st.session_state["pdf_upload_error"] = (
                    f"Only PDF files are allowed. Not accepted: {', '.join(errors)}"
                )
                st.session_state.pdf_upload_counter += 1
                st.rerun()
            else:
                skipped, saved = [], []
                existing_pdf_hashes = {
                    _file_hash(p.read_bytes()): p.name
                    for p in raw_docs.glob("*.pdf")
                }
                for f in uploaded_pdfs:
                    dest = raw_docs / f.name
                    data = f.getvalue()
                    if _file_hash(data) in existing_pdf_hashes:
                        skipped.append(f.name)
                    else:
                        # Same filename, different content — purge old chunks so re-extraction is forced
                        if dest.exists() and chunks_path.exists():
                            df_ch = _load_csv_safe(str(chunks_path))
                            if not df_ch.empty and "source_file" in df_ch.columns:
                                df_ch = df_ch[df_ch["source_file"] != f.name]
                                df_ch.to_csv(chunks_path, index=False)
                        dest.write_bytes(data)
                        saved.append(f.name)
                if skipped and not saved:
                    st.session_state["pdf_upload_error"] = (
                        f"No changes — file(s) already up to date: {', '.join(skipped)}"
                    )
                if saved:
                    try:
                        from src.ingestion import extract_documents
                        from src.retrieval import build_chunk_index
                        current_pdfs = set(p.name for p in raw_docs.glob("*.pdf"))
                        with st.spinner("Processing uploaded PDFs — please wait..."):
                            extract_documents.extract_documents()
                            build_chunk_index.build_chunk_index()
                        _cleanup_orphaned_qas(current_pdfs, drafts_path, approved_path)
                        st.cache_resource.clear()
                        df_new_chunks = _load_csv_safe(str(chunks_path))
                        chunked_sources = set(df_new_chunks["source_file"].dropna().unique()) if not df_new_chunks.empty else set()
                        zero_chunk = [n for n in saved if n not in chunked_sources]
                        if zero_chunk:
                            st.session_state["pdf_upload_error"] = (
                                f"The following PDF(s) could not be read and won't be searchable — "
                                f"they may be image-only or password-protected: {', '.join(zero_chunk)}"
                            )
                        st.session_state["generate_confirmed"] = False
                        _queue_message("pdf_action", f"Uploaded and ready: {', '.join(saved)}")
                    except Exception as e:
                        _queue_message(
                            "pdf_action",
                            f"Uploaded {', '.join(saved)} but processing failed: {e}",
                            level="warning",
                        )
                st.session_state.pdf_upload_counter += 1
                st.rerun()

        if pdf_files:
            for p in pdf_files:
                c1, c2, c3, c4 = st.columns([3, 1, 2, 1])
                c1.write(p.name)
                c2.write(f"{round(p.stat().st_size/1024,1)} KB")
                c3.write(datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"))
                ck = f"confirm_del_pdf_{p.name}"
                if st.session_state.get(ck):
                    if c4.button("Confirm", key=f"do_del_pdf_{p.name}", type="primary"):
                        deleted_name = p.name
                        p.unlink()
                        st.session_state.pop(ck, None)
                        remaining = set(f.name for f in raw_docs.glob("*.pdf"))
                        _cleanup_orphaned_qas(remaining, drafts_path, approved_path)
                        if not remaining:
                            for stale in [chunks_path, chunk_index_path, Path("data/processed/chunk_meta.json")]:
                                if stale.exists():
                                    stale.unlink()
                            st.cache_resource.clear()
                        else:
                            if chunks_path.exists():
                                df_chunks = _load_csv_safe(str(chunks_path))
                                if not df_chunks.empty and "source_file" in df_chunks.columns:
                                    df_chunks = df_chunks[df_chunks["source_file"] != deleted_name]
                                    df_chunks.to_csv(chunks_path, index=False)
                        st.session_state["generate_confirmed"] = False
                        try:
                            from src.ingestion import load_dataset
                            from src.retrieval import build_index
                            with st.spinner("Updating knowledge base — please wait..."):
                                load_dataset.load_dataset()
                                build_index.build_index()
                            st.cache_resource.clear()
                            _queue_message("pdf_action", f"{deleted_name} deleted and knowledge base updated.")
                        except Exception as e:
                            _queue_message("pdf_action", f"{deleted_name} deleted but knowledge base update failed: {e}", level="warning")
                        st.rerun()
                else:
                    if c4.button("Delete", key=f"del_pdf_{p.name}"):
                        st.session_state[ck] = True
                        st.rerun()
            if any(st.session_state.get(f"confirm_del_pdf_{p.name}") for p in pdf_files):
                st.caption("Click **Confirm** to permanently delete.")
        else:
            st.info("No PDFs uploaded yet.")

        st.divider()

        # Generate Q&A Suggestions
        st.markdown("**Auto-generate Q&A Suggestions**")
        st.caption(
            "Automatically generates question and answer pairs from your uploaded PDFs. "
            "This may take several minutes and uses the OpenAI API. "
            "Please do not navigate away until complete — if interrupted, re-running will pick up where it left off."
        )
        if chunk_count == 0:
            st.info("Upload a PDF first.")
            st.button("Generate Q&A Suggestions", disabled=True, use_container_width=True, key="gen_no_chunks")
        elif unprocessed == 0:
            st.success("All PDFs have been processed.")
            st.button("Generate Q&A Suggestions", disabled=True, use_container_width=True, key="gen_done")
        elif st.session_state.get("generating_qas"):
            st.button("Generating Q&A Suggestions…", disabled=True, use_container_width=True, key="gen_running")
            progress_bar = st.progress(0.0, text="Starting — processing sections…")

            def _on_progress(done: int, total: int) -> None:
                pct = done / total
                progress_bar.progress(pct, text=f"Processing section {done} of {total}…")

            _gen_error = None
            try:
                from src.ingestion import generate_draft_qas
                generate_draft_qas.generate_drafts(on_progress=_on_progress)
                progress_bar.progress(1.0, text="Complete!")
            except Exception as _e:
                _gen_error = _e

            st.session_state.pop("generating_qas", None)
            if _gen_error:
                st.error(f"Generation failed: {_gen_error}")
            else:
                st.rerun()
        else:
            st.warning(
                f"{unprocessed} section(s) not yet processed — generating may incur API costs. "
                "Please stay on this page until complete."
            )
            confirmed = st.checkbox("I understand this may incur API costs", key="generate_confirmed")
            if st.button("Generate Q&A Suggestions", disabled=not confirmed, use_container_width=True, type="primary", key="gen_btn"):
                st.session_state["generating_qas"] = True
                st.rerun()

        st.divider()

        # ── Inline Q&A Review ─────────────────────────────────────────────────
        review_label = (
            f"**Review Q&A Suggestions** — {pending_drafts} pending"
            if pending_drafts > 0
            else "**Review Q&A Suggestions**"
        )
        with st.expander(review_label, expanded=pending_drafts > 0):
            _review_drafts_inline()

        st.divider()

        # Update Knowledge Base (for approved PDF Q&As)
        st.markdown("**Update Knowledge Base**")
        st.caption("After reviewing and approving Q&A suggestions, click here to make them searchable.")

        qa_index_path = Path("data/processed/index.faiss")
        approved_path_kb = Path("data/raw/approved_qas.csv")
        df_clean_pdf = _load_csv_safe("data/processed/qa_dataset_clean.csv")
        if not df_clean_pdf.empty and "source_file" in df_clean_pdf.columns:
            approved_rows = int((~df_clean_pdf["source_file"].str.lower().str.endswith(".csv", na=False)).sum())
        else:
            approved_rows = 0

        if approved_rows > 0:
            st.metric("Approved PDF Q&As in knowledge base", approved_rows)

        approved_newer_than_index = (
            approved_path_kb.exists()
            and _row_count(str(approved_path_kb)) > 0
            and (
                not qa_index_path.exists()
                or _mtime(str(approved_path_kb)) > _mtime(str(qa_index_path))
            )
        )

        if not approved_path_kb.exists() or _row_count(str(approved_path_kb)) == 0:
            st.info("No approved Q&As yet — review suggestions first.")
            st.button("Update Knowledge Base", disabled=True, use_container_width=True, key="update_kb_disabled")
        elif not approved_newer_than_index:
            st.success("Knowledge base is up to date.")
            st.button("Update Knowledge Base", disabled=True, use_container_width=True, key="update_kb_ok")
        else:
            st.warning("Approved Q&As have not yet been added to the knowledge base.")
            if st.button("Update Knowledge Base", use_container_width=True, type="primary", key="update_kb_btn"):
                try:
                    from src.ingestion import load_dataset
                    from src.retrieval import build_index
                    with st.spinner("Updating knowledge base — please wait..."):
                        load_dataset.load_dataset()
                        build_index.build_index()
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Update failed: {e}")

    # ── Tab 2: Q&A Files ──────────────────────────────────────────────────────
    with tab_csv:
        st.subheader("Q&A Files")
        st.caption("Upload a CSV with `question` and `answer` columns to add pairs directly to the knowledge base.")

        if st.session_state.get("csv_upload_error"):
            st.error(st.session_state.pop("csv_upload_error"))
        for _info in st.session_state.pop("csv_upload_info", []):
            st.info(_info)
        _show_message("csv_action")

        uploaded_qas = st.file_uploader(
            "Choose CSV files",
            type=["csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"kb_csv_upload_{st.session_state.csv_upload_counter}",
        )
        if uploaded_qas:
            wrong_type = [f.name for f in uploaded_qas if not f.name.lower().endswith(".csv")]
            if wrong_type:
                st.session_state["csv_upload_error"] = (
                    f"The following files are not CSVs: {', '.join(wrong_type)}"
                )
                st.session_state.csv_upload_counter += 1
                st.rerun()
            else:
                saved, skipped, errors = [], [], []
                existing_hashes = {
                    _file_hash(p.read_bytes()): p.name
                    for p in Path("data/raw").glob("*.csv")
                    if p.name != "approved_qas.csv"
                }
                for f in uploaded_qas:
                    raw = f.getvalue()
                    dest = Path("data/raw") / f.name
                    upload_hash = _file_hash(raw)
                    # Identical content already on disk (any filename)
                    if upload_hash in existing_hashes:
                        skipped.append(f.name)
                        continue
                    preview_df = None
                    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
                        try:
                            preview_df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                            break
                        except (UnicodeDecodeError, Exception):
                            continue
                    if preview_df is None:
                        errors.append(f"{f.name}: could not read (try saving as CSV UTF-8)")
                    elif "question" not in preview_df.columns or "answer" not in preview_df.columns:
                        errors.append(f"{f.name}: missing `question` or `answer` column")
                    elif len(preview_df) == 0:
                        errors.append(f"{f.name}: file is empty — no data rows found")
                    else:
                        # Check overlap against existing knowledge base
                        existing_qa = _load_csv_safe("data/processed/qa_dataset_clean.csv")
                        total_rows = len(preview_df)
                        new_count = total_rows
                        if not existing_qa.empty and "question" in existing_qa.columns:
                            existing_qs = set(existing_qa["question"].map(_norm_q))
                            new_count = sum(
                                1 for _, row in preview_df.iterrows()
                                if _norm_q(row.get("question", "")) not in existing_qs
                            )
                        if new_count == 0:
                            errors.append(
                                f"{f.name}: all {total_rows} Q&A pairs already exist in the knowledge base — nothing new to add"
                            )
                            continue
                        dest.write_bytes(raw)
                        saved.append(f.name)
                        info_parts = []
                        if new_count < total_rows:
                            info_parts.append(f"{new_count} new, {total_rows - new_count} already existed and will be ignored")
                        # Warn about questions blocked by exclusion list
                        if EXCLUDED_PATH.exists():
                            ex = _load_csv_safe(str(EXCLUDED_PATH))
                            if not ex.empty and "question_norm" in ex.columns:
                                excl_norms = set(ex["question_norm"])
                                blocked = sum(
                                    1 for _, row in preview_df.iterrows()
                                    if _norm_q(str(row.get("question", ""))) in excl_norms
                                )
                                if blocked:
                                    info_parts.append(
                                        f"{blocked} question(s) are in the excluded list and will be skipped — restore them in Knowledge Base if needed"
                                    )
                        if info_parts:
                            st.session_state.setdefault("csv_upload_info", []).append(
                                f"{f.name}: " + "; ".join(info_parts)
                            )

                msgs = []
                if skipped:
                    msgs.append(f"No changes — already up to date: {', '.join(skipped)}")
                if errors:
                    msgs.extend(errors)
                if msgs and not saved:
                    st.session_state["csv_upload_error"] = "\n".join(msgs)
                if saved:
                    try:
                        from src.ingestion import load_dataset
                        from src.retrieval import build_index
                        with st.spinner("Adding to knowledge base — please wait..."):
                            load_dataset.load_dataset()
                            build_index.build_index()
                        st.cache_resource.clear()
                        _queue_message("csv_action", f"Uploaded and added to knowledge base: {', '.join(saved)}")
                    except Exception as e:
                        _queue_message(
                            "csv_action",
                            f"Uploaded {', '.join(saved)} but knowledge base update failed: {e}",
                            level="warning",
                        )
                    st.session_state.csv_upload_counter += 1
                st.rerun()

        raw_csvs_display = sorted(p for p in Path("data/raw").glob("*.csv") if p.name != "approved_qas.csv")
        if raw_csvs_display:
            st.write("**Uploaded files:**")
            for p in raw_csvs_display:
                c1, c2, c3 = st.columns([4, 1, 1])
                c1.write(p.name)
                c2.write(f"{_row_count(str(p))} rows")
                ck = f"confirm_del_csv_{p.name}"
                if st.session_state.get(ck):
                    if c3.button("Confirm", key=f"do_del_csv_{p.name}", type="primary"):
                        deleted_csv_name = p.name
                        p.unlink()
                        st.session_state.pop(ck, None)
                        try:
                            from src.ingestion import load_dataset
                            from src.retrieval import build_index
                            with st.spinner("Removing Q&As and updating knowledge base..."):
                                load_dataset.load_dataset()
                                build_index.build_index()
                            st.cache_resource.clear()
                            _queue_message("csv_action", f"{deleted_csv_name} deleted and knowledge base updated.")
                        except Exception as e:
                            st.warning(f"File deleted but knowledge base update failed: {e}")
                        st.rerun()
                else:
                    if c3.button("Delete", key=f"del_csv_{p.name}"):
                        st.session_state[ck] = True
                        st.rerun()
            if any(st.session_state.get(f"confirm_del_csv_{p.name}") for p in raw_csvs_display):
                st.caption("Click **Confirm** to permanently delete.")
        else:
            st.info("No Q&A files uploaded yet.")

        st.divider()

        # Summary stat only — full view is in Knowledge Base page
        df_clean_csv = _load_csv_safe("data/processed/qa_dataset_clean.csv")
        if not df_clean_csv.empty and "source_file" in df_clean_csv.columns:
            manual_rows = int(df_clean_csv["source_file"].str.lower().str.endswith(".csv", na=False).sum())
        else:
            manual_rows = 0
        c1, c2 = st.columns(2)
        c1.metric("Q&As from uploaded files", manual_rows)
        c2.metric("Last updated", _mtime_label("data/processed/index.faiss"))
        st.caption("To view or remove individual Q&As, go to **Knowledge Base** in the sidebar.")


# ── Page: Evaluation ─────────────────────────────────────────────────────────

def page_evaluation():
    st.header("Evaluation")
    st.caption("Dissertation evidence dashboard — all stats derived from live system data. Reset the query log in Settings before your final evaluation run to ensure clean data.")

    # ── Section 1: Knowledge Base Pipeline ───────────────────────────────────
    # Purpose: shows the methodology — how source documents became a searchable KB.
    # Useful for the dissertation's implementation/methodology chapter.
    st.subheader("Knowledge Base Pipeline")
    st.caption("How source documents were transformed into the searchable knowledge base.")

    raw_docs = Path("data/raw_docs")
    chunks_path = Path("data/processed/document_chunks.csv")
    drafts_path = Path("data/processed/draft_qas.csv")
    kb_path = Path("data/processed/qa_dataset_clean.csv")

    pdf_count = len(list(raw_docs.glob("*.pdf"))) if raw_docs.exists() else 0
    chunk_count = _row_count(str(chunks_path))

    generated, approved_n, rejected_n, skipped_n = 0, 0, 0, 0
    if drafts_path.exists():
        df_drafts = _load_csv_safe(str(drafts_path))
        if not df_drafts.empty and "review_status" in df_drafts.columns:
            generated = int((df_drafts["review_status"] != "skipped").sum())
            approved_n = int((df_drafts["review_status"] == "approved").sum())
            rejected_n = int((df_drafts["review_status"] == "rejected").sum())
            skipped_n = int((df_drafts["review_status"] == "skipped").sum())

    kb_size = _row_count(str(kb_path))

    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("PDF documents", pdf_count, help="Source documents uploaded to the system.")
    p2.metric("Text passages", chunk_count, help="Segments extracted from PDFs for semantic search.")
    p3.metric("Q&As generated", generated, help="Q&A pairs generated from the text passages.")
    p4.metric("Approved", approved_n, help="Pairs reviewed and approved by admin.")
    p5.metric("Rejected", rejected_n, help="Pairs rejected during review.")
    p6.metric("In knowledge base", kb_size, help="Final Q&A pairs after deduplication and exclusions.")

    st.caption(f"Pipeline: {pdf_count} PDF(s) → {chunk_count} passages → {generated} Q&As generated → {approved_n} approved → {kb_size} in knowledge base.")
    if skipped_n > 0:
        st.caption(f"{skipped_n} passages produced no valid Q&A pairs and were skipped during generation.")

    # ── Query data ────────────────────────────────────────────────────────────
    log = _load_csv_safe("data/processed/query_log.csv")
    unanswered = _load_csv_safe("data/processed/unanswered_log.csv")

    if log.empty:
        st.divider()
        st.info("No query data yet. Run queries against the system to populate evaluation stats.")
        return

    if "timestamp" in log.columns:
        log["timestamp"] = pd.to_datetime(log["timestamp"], errors="coerce")
    if "qa_score" in log.columns:
        log["qa_score"] = pd.to_numeric(log["qa_score"], errors="coerce")
    if "chunk_score" in log.columns:
        log["chunk_score"] = pd.to_numeric(log["chunk_score"], errors="coerce")
    if "qa_threshold" in log.columns:
        log["qa_threshold"] = pd.to_numeric(log["qa_threshold"], errors="coerce")

    total = len(log)
    qa_matches = log[log["mode"] == "qa_answer"]
    chunk_matches = log[log["mode"] == "chunk_fallback"]
    llm_matches = log[log["mode"] == "llm_fallback"]
    none_matches = log[log["mode"].isin(["no_answer", "none"])]

    # ── Section 2: Query Activity ─────────────────────────────────────────────
    st.divider()
    st.subheader("Query Activity")

    if "timestamp" in log.columns and log["timestamp"].notna().any():
        st.markdown("**Queries over time**")
        daily = (
            log.set_index("timestamp")
            .resample("D")
            .size()
            .reset_index(name="queries")
        )
        st.line_chart(daily.set_index("timestamp")["queries"])
    else:
        st.info("No timestamp data available.")

    st.markdown("**How queries were answered**")
    _mode_label_map = {
        "qa_answer": "Q&A Match",
        "chunk_fallback": "Passage Search",
        "llm_fallback": "Generated Answer",
        "no_answer": "Unanswered",
        "none": "Unanswered",
    }
    _all_methods = ["Q&A Match", "Passage Search", "Generated Answer", "Unanswered"]
    _raw_counts = log["mode"].map(_mode_label_map).fillna(log["mode"]).value_counts()
    _mode_counts = pd.DataFrame([
        {"method": m, "count": int(_raw_counts.get(m, 0))}
        for m in _all_methods
    ])
    if _PLOTLY:
        _fig = px.pie(
            _mode_counts,
            names="method",
            values="count",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        _fig.update_traces(textposition="inside", textinfo="percent+label")
        _fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
        st.plotly_chart(_fig, use_container_width=True)
    else:
        st.bar_chart(_mode_counts.set_index("method")["count"])

    # ── Section 3: Retrieval Performance ──────────────────────────────────────
    # Purpose: the primary table for the dissertation argument.
    # Shows what % of queries semantic search handled vs passage search vs generated answer fallback.
    # A high Q&A match rate is the main evidence that semantic search works.
    st.divider()
    st.subheader("Retrieval Performance")
    st.caption("The core evidence for your dissertation. Shows how often semantic Q&A matching answered a query without needing a generated answer.")

    mode_data = pd.DataFrame([
        {"Layer": "Semantic Q&A match",    "Queries": len(qa_matches),    "% of total": f"{len(qa_matches)/total*100:.1f}%",    "What it means": "Directly matched a stored Q&A — no generation needed"},
        {"Layer": "Semantic passage search","Queries": len(chunk_matches), "% of total": f"{len(chunk_matches)/total*100:.1f}%", "What it means": "Found a relevant PDF passage — no Q&A match"},
        {"Layer": "Generated answer",       "Queries": len(llm_matches),   "% of total": f"{len(llm_matches)/total*100:.1f}%",   "What it means": "No semantic match — fell back to generated answer"},
        {"Layer": "Unanswered",             "Queries": len(none_matches),  "% of total": f"{len(none_matches)/total*100:.1f}%",  "What it means": "No match found and fallback disabled or failed"},
    ])
    st.dataframe(mode_data, use_container_width=True, hide_index=True)

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Semantic Q&A match", f"{len(qa_matches)/total*100:.1f}%",
              help="Queries directly matched to a stored Q&A — no generation needed.")
    r2.metric("Passage search", f"{len(chunk_matches)/total*100:.1f}%",
              help="Queries answered via a relevant PDF passage — no Q&A match found.")
    r3.metric("Generated answer", f"{len(llm_matches)/total*100:.1f}%",
              help="Queries where semantic search found nothing and the fallback was used.")
    r4.metric("Unanswered", f"{len(none_matches)/total*100:.1f}%",
              help="Queries that failed all layers.")
    r5.metric("Total evaluated", total)

    # KB coverage curve
    # Purpose: shows that query coverage improves as the KB grows.
    # Argues that the unanswered log feedback loop drives adaptability — each time an admin
    # adds Q&A pairs in response to the unanswered log, the system moves right along this curve.
    st.markdown("**Knowledge base growth vs. query coverage**")
    st.caption(
        "Each point shows the % of test queries answered at a given KB size. "
        "As admins use the unanswered log to add Q&A pairs, coverage improves — "
        "a feedback loop that static systems lack."
    )
    with st.spinner("Computing coverage curve…"):
        coverage_df = _compute_kb_coverage_curve()
    if coverage_df is not None:
        if _PLOTLY:
            fig = px.line(
                coverage_df,
                x="Q&A pairs in knowledge base",
                y="Query coverage (%)",
                markers=True,
                labels={
                    "Q&A pairs in knowledge base": "Q&A pairs in knowledge base",
                    "Query coverage (%)": "Query coverage (%)",
                },
            )
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(coverage_df.set_index("Q&A pairs in knowledge base"))
        final_coverage = coverage_df["Query coverage (%)"].iloc[-1]
        baseline_coverage = coverage_df["Query coverage (%)"].iloc[0]
        gain = final_coverage - baseline_coverage
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Coverage at 10% KB", f"{baseline_coverage:.1f}%", help="Match rate with only 10% of Q&A pairs loaded.")
        cc2.metric("Coverage at full KB", f"{final_coverage:.1f}%", help="Match rate with all current Q&A pairs.")
        cc3.metric("Coverage gain", f"+{gain:.1f}pp", help="Percentage point improvement from growing the KB.")
    else:
        st.info("Coverage curve unavailable — requires qa_dataset_clean.csv and evaluation/comparison_queries.csv.")

    # ── Section 4: Match Quality ──────────────────────────────────────────────
    # Purpose: shows the confidence of semantic matches.
    # High scores mean the system isn't just returning approximate answers — it's returning good ones.
    # Score distribution and confidence bands are key dissertation evidence.
    st.divider()
    st.subheader("Match Quality")
    st.caption("How confident were the semantic matches? High scores indicate the system returns genuinely relevant answers, not just the closest approximation.")

    if not qa_matches.empty and "qa_score" in qa_matches.columns:
        scores = qa_matches["qa_score"].dropna()
        if len(scores) > 0:
            sq1, sq2, sq3, sq4 = st.columns(4)
            sq1.metric("Average match score", f"{scores.mean():.3f}", help="Mean similarity score across all Q&A matches.")
            sq2.metric("Median match score", f"{scores.median():.3f}", help="Half of matches scored above this value.")
            sq3.metric("Highest score", f"{scores.max():.3f}")
            sq4.metric("Lowest match score", f"{scores.min():.3f}", help="The weakest match that still passed the threshold.")

            # Confidence bands
            # Purpose: shows the quality distribution of matches — not just that queries matched, but HOW WELL.
            # High-confidence matches (>0.85) are the clearest evidence semantic search is working correctly.
            threshold_val = log["qa_threshold"].dropna().iloc[0] if "qa_threshold" in log.columns and len(log["qa_threshold"].dropna()) > 0 else 0.7
            high = int((scores >= 0.85).sum())
            medium = int(((scores >= float(threshold_val)) & (scores < 0.85)).sum())
            low = int((scores < float(threshold_val)).sum())

            st.markdown("**Confidence bands**")
            band_df = pd.DataFrame([
                {"Band": "High confidence  (≥ 0.85)",                              "Q&A matches": high,   "% of matches": f"{high/len(scores)*100:.1f}%",   "Significance": "Strong semantic match — clearly correct answer"},
                {"Band": f"Medium confidence  (threshold – 0.85)",                 "Q&A matches": medium, "% of matches": f"{medium/len(scores)*100:.1f}%", "Significance": "Good match, above threshold but not near-certain"},
                {"Band": f"Low confidence  (< threshold, {threshold_val:.2f})",    "Q&A matches": low,    "% of matches": f"{low/len(scores)*100:.1f}%",    "Significance": "Below threshold — these were NOT returned to user"},
            ])
            st.dataframe(band_df, use_container_width=True, hide_index=True)
            st.caption("Note: low-confidence rows are queries that attempted a match but fell below the threshold — they were not served as Q&A answers.")

            # Score distribution histogram
            # Purpose: visualises the spread of match scores — a cluster near 1.0 strongly supports semantic search.
            st.markdown("**Score distribution**")
            st.caption("A distribution skewed toward 1.0 shows the semantic matches are confident, not borderline.")
            bins = [i * 0.05 for i in range(21)]
            binned = pd.cut(scores, bins=bins).value_counts().sort_index()
            hist_df = pd.DataFrame({"score range": [str(b) for b in binned.index], "count": binned.values})
            st.bar_chart(hist_df.set_index("score range")["count"])

            # Threshold boundary cases
            # Purpose: shows how many queries were right on the edge of matching.
            # Helps argue for/against the chosen threshold value.
            if "qa_threshold" in log.columns and len(log["qa_threshold"].dropna()) > 0:
                boundary_all = log[(log["qa_score"] - log["qa_threshold"]).abs() < 0.05]
                boundary_missed = boundary_all[boundary_all["mode"] != "qa_answer"]
                boundary_matched = boundary_all[boundary_all["mode"] == "qa_answer"]
                bc1, bc2 = st.columns(2)
                bc1.metric("Near-miss queries (within 0.05 below threshold)", len(boundary_missed),
                           help="Queries that almost matched — lowering the threshold would have answered these.")
                bc2.metric("Borderline matches (within 0.05 above threshold)", len(boundary_matched),
                           help="Queries that only just passed the threshold — useful for sensitivity analysis.")
    else:
        st.info("No Q&A match data yet.")

    # ── Section 5: Query Characteristics ─────────────────────────────────────
    # Purpose: short, natural-language queries are where semantic search has the biggest advantage.
    # If short queries still achieve high match rates, that directly argues against keyword search.
    st.divider()
    st.subheader("Query Characteristics")
    st.caption("Short, natural-language queries are where semantic search has the biggest advantage over keyword-based approaches. This table shows match rate by query length.")

    log["word_count"] = log["query"].dropna().str.split().str.len()
    length_bins   = [0, 2, 4, 7, 11, 100]
    length_labels = ["1–2 words", "3–4 words", "5–7 words", "8–11 words", "12+ words"]
    log["length_band"] = pd.cut(log["word_count"], bins=length_bins, labels=length_labels)

    length_rows = []
    for band in length_labels:
        subset = log[log["length_band"] == band]
        n = len(subset)
        matched = int((subset["mode"] == "qa_answer").sum())
        length_rows.append({
            "Query length": band,
            "Total queries": n,
            "Q&A matched": matched,
            "Match rate": f"{matched/n*100:.1f}%" if n > 0 else "n/a",
            "Why it matters": "Keyword search struggles here" if band in ["1–2 words", "3–4 words"] else "Both approaches can handle these",
        })
    st.dataframe(pd.DataFrame(length_rows), use_container_width=True, hide_index=True)

    # Repeated queries
    # Purpose: shows real usage patterns — repeated queries confirm which topics users actually care about.
    repeated = log["query"].str.lower().value_counts()
    repeated = repeated[repeated > 1].reset_index()
    repeated.columns = ["query", "times asked"]
    if not repeated.empty:
        st.markdown("**Repeated queries**")
        st.caption("Questions asked more than once — confirms these are real user needs, not one-off tests.")
        st.dataframe(repeated.head(20), use_container_width=True, hide_index=True)

    # ── Section 6: Gap Analysis ───────────────────────────────────────────────
    # Purpose: intellectually honest — shows where the system fails.
    # Repeated unanswered queries are the strongest signal of KB gaps.
    # Important for dissertation to acknowledge limitations.
    st.divider()
    st.subheader("Gap Analysis")
    st.caption("Queries the system could not answer. Shows where the knowledge base has gaps — important for an honest evaluation.")

    if unanswered.empty:
        st.info("No unanswered queries recorded yet.")
    else:
        if "query" in unanswered.columns:
            repeated_un = unanswered["query"].str.lower().value_counts()
            repeated_un = repeated_un[repeated_un > 1].reset_index()
            repeated_un.columns = ["query", "times asked unanswered"]
        else:
            repeated_un = pd.DataFrame()

        g1, g2 = st.columns(2)
        g1.metric("Total unanswered queries", len(unanswered),
                  help="Queries that failed all retrieval layers.")
        g2.metric("Repeated unanswered queries", len(repeated_un),
                  help="Asked more than once and never answered — highest priority knowledge base gaps.")

        if not repeated_un.empty:
            st.markdown("**Repeated unanswered queries**")
            st.caption("These are the most important gaps — users keep asking and never get an answer.")
            st.dataframe(repeated_un, use_container_width=True, hide_index=True)

        st.markdown("**All unanswered queries**")
        display_cols = [c for c in ["timestamp", "query", "best_qa_score", "best_chunk_score"] if c in unanswered.columns]
        st.dataframe(unanswered[display_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Export unanswered queries",
            data=unanswered.to_csv(index=False).encode("utf-8"),
            file_name="unanswered_queries.csv",
            mime="text/csv",
        )

    # ── Section 7: Score Comparison by Mode ──────────────────────────────────
    # Purpose: shows that Q&A matches have higher confidence scores than passage matches.
    # This directly argues that the semantic Q&A layer is functioning correctly —
    # it doesn't just match more queries, it matches them with higher confidence.
    st.divider()
    st.subheader("Score Comparison by Mode")
    st.caption("Average similarity scores for each retrieval layer. Higher Q&A scores vs passage scores shows the semantic Q&A layer is returning more confident, targeted answers.")

    score_rows = []
    for mode_val, label in [
        ("qa_answer", "Semantic Q&A match"),
        ("chunk_fallback", "Passage Search"),
        ("llm_fallback", "Generated answer"),
        ("no_answer", "Unanswered"),
    ]:
        subset = log[log["mode"].isin([mode_val, "none"]) if mode_val == "no_answer" else log["mode"] == mode_val]
        avg_qa = subset["qa_score"].dropna().mean() if "qa_score" in subset.columns and len(subset) > 0 else None
        avg_chunk = subset["chunk_score"].dropna().mean() if "chunk_score" in subset.columns and len(subset) > 0 else None
        score_rows.append({
            "Mode": label,
            "Count": len(subset),
            "Avg Q&A score": f"{avg_qa:.3f}" if avg_qa is not None and not pd.isna(avg_qa) else "—",
            "Avg passage score": f"{avg_chunk:.3f}" if avg_chunk is not None and not pd.isna(avg_chunk) else "—",
        })
    st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.download_button(
        "Export full query log",
        data=log.to_csv(index=False).encode("utf-8"),
        file_name="query_log_full.csv",
        mime="text/csv",
    )


# ── Page: Settings ────────────────────────────────────────────────────────────

def page_settings():
    st.header("Settings")
    st.caption("Changes are saved to the `.env` file. Restart the app to apply them.")

    new_qa_threshold = st.slider(
        "Q&A match sensitivity",
        min_value=0.0, max_value=1.0,
        value=float(settings.similarity_threshold),
        step=0.01, key="settings_qa_threshold",
        help="How closely a user's question must match a stored question to return that answer. Higher = stricter.",
    )
    new_chunk_threshold = st.slider(
        "Passage match sensitivity",
        min_value=0.0, max_value=1.0,
        value=float(settings.chunk_similarity_threshold),
        step=0.01, key="settings_chunk_threshold",
        help="How closely a user's question must match a passage from a PDF to use it as context. Higher = stricter.",
    )
    new_llm_enabled = st.toggle(
        "Enable generated answer fallback",
        value=settings.llm_fallback_enabled,
        key="settings_llm_enabled",
        help="When no match is found, fall back to generate an answer.",
    )
    new_openai_model = st.text_input(
        "OpenAI model",
        value=settings.openai_model,
        key="settings_openai_model",
    )

    if st.button("Save Settings"):
        try:
            dotenv_path = find_dotenv() or ".env"
            if not Path(dotenv_path).exists():
                Path(dotenv_path).touch()
            set_key(dotenv_path, "SIMILARITY_THRESHOLD", str(new_qa_threshold))
            set_key(dotenv_path, "CHUNK_SIMILARITY_THRESHOLD", str(new_chunk_threshold))
            set_key(dotenv_path, "LLM_FALLBACK_ENABLED", str(new_llm_enabled).lower())
            set_key(dotenv_path, "OPENAI_MODEL", new_openai_model)
            from src.utils.config import reload_settings
            reload_settings()
            st.cache_resource.clear()
            st.success("Settings saved and applied.")
        except Exception as e:
            st.error(f"Failed to save settings: {e}")

    st.divider()

    # ── Reset System ──────────────────────────────────────────────────────────
    st.subheader("Reset")
    st.caption("Permanently delete data to start fresh. This cannot be undone.")

    reset_opts = st.multiselect(
        "Select what to delete:",
        options=[
            "PDF files",
            "Uploaded Q&A files",
            "Extracted PDF content",
            "PDF search index",
            "Q&A suggestions",
            "Approved Q&As",
            "Knowledge base",
            "Query log",
        ],
        key="reset_opts",
    )

    _show_message("reset_action")

    if reset_opts:
        if st.session_state.get("confirm_reset"):
            st.error(f"This will permanently delete: **{', '.join(reset_opts)}**. Are you sure?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, delete permanently", type="primary", key="reset_do_btn"):
                    deleted = []
                    errors = []

                    def _try_delete(path: str, label: str):
                        p = Path(path)
                        if p.exists():
                            try:
                                p.unlink()
                                deleted.append(label)
                            except Exception as exc:
                                errors.append(f"{label}: {exc}")

                    def _try_delete_dir_contents(pattern: str, label: str):
                        files = list(Path(".").glob(pattern))
                        if files:
                            for f in files:
                                try:
                                    f.unlink()
                                except Exception as exc:
                                    errors.append(f"{f.name}: {exc}")
                            deleted.append(label)

                    if "PDF files" in reset_opts:
                        _try_delete_dir_contents("data/raw_docs/*.pdf", "PDF files")
                    if "Uploaded Q&A files" in reset_opts:
                        _try_delete_dir_contents("data/raw/*.csv", "Uploaded Q&A files")
                    if "Extracted PDF content" in reset_opts:
                        _try_delete("data/processed/document_chunks.csv", "Extracted PDF content")
                    if "PDF search index" in reset_opts:
                        _try_delete("data/processed/chunk_index.faiss", "PDF search index")
                        _try_delete("data/processed/chunk_meta.json", "chunk_meta.json")
                    if "Q&A suggestions" in reset_opts:
                        _try_delete("data/processed/draft_qas.csv", "Q&A suggestions")
                    if "Approved Q&As" in reset_opts:
                        _try_delete("data/raw/approved_qas.csv", "Approved Q&As")
                    if "Knowledge base" in reset_opts:
                        _try_delete("data/processed/qa_dataset_clean.csv", "qa_dataset_clean.csv")
                        _try_delete("data/processed/index.faiss", "index.faiss")
                        _try_delete("data/processed/meta.json", "meta.json")
                    if "Query log" in reset_opts:
                        _try_delete("data/processed/query_log.csv", "query_log.csv")
                        _try_delete("data/processed/unanswered_log.csv", "unanswered_log.csv")

                    st.cache_resource.clear()
                    st.session_state.pop("confirm_reset", None)
                    msg = f"Deleted: {', '.join(deleted)}" if deleted else "Nothing was deleted."
                    if errors:
                        msg += f" Errors: {'; '.join(errors)}"
                    _queue_message("reset_action", msg, level="success" if not errors else "warning")
                    st.rerun()
            with c2:
                if st.button("Cancel", key="reset_cancel_btn"):
                    st.session_state.pop("confirm_reset", None)
                    st.rerun()
        else:
            if st.button("Reset selected", type="primary", key="reset_confirm_btn"):
                st.session_state["confirm_reset"] = True
                st.rerun()


# ── Page: Deleted Q&As ────────────────────────────────────────────────────────

def page_knowledge_base():
    st.header("Knowledge Base")

    if "qa_editor_counter" not in st.session_state:
        st.session_state.qa_editor_counter = 0

    tab_current, tab_deleted = st.tabs(["Current Q&As", "Deleted Q&As"])

    # ── Tab 1: Current Q&As ───────────────────────────────────────────────────
    with tab_current:
        _show_message("qa_dataset_action")

        df_kb = _load_csv_safe("data/processed/qa_dataset_clean.csv")
        if df_kb.empty:
            st.info("No Q&As in the knowledge base yet. Upload PDFs or Q&A files in Documents.")
            return

        # Summary metrics
        if "source_file" in df_kb.columns:
            pdf_count = int((~df_kb["source_file"].str.lower().str.endswith(".csv", na=False)).sum())
            csv_count = int(df_kb["source_file"].str.lower().str.endswith(".csv", na=False).sum())
        else:
            pdf_count, csv_count = 0, 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Q&As", len(df_kb))
        c2.metric("From PDFs", pdf_count)
        c3.metric("From uploaded files", csv_count)

        st.divider()

        filter_text = st.text_input("Filter by question", key="kb_filter")
        df_view = df_kb.copy()
        if filter_text:
            df_view = df_view[df_view["question"].str.contains(filter_text, case=False, na=False)]

        if "page" in df_view.columns:
            df_view["page"] = (
                df_view["page"].fillna("").astype(str)
                .str.replace(r"^nan$", "", regex=True)
                .str.replace(r"\.0$", "", regex=True)
            )
        if "source_file" in df_view.columns:
            df_view.insert(
                1, "source_type",
                df_view["source_file"].apply(
                    lambda f: "Uploaded file" if str(f).lower().endswith(".csv") else "PDF"
                ),
            )
        df_view.insert(0, "Delete", False)
        edited = st.data_editor(
            df_view,
            use_container_width=True,
            hide_index=True,
            column_config={"Delete": st.column_config.CheckboxColumn("Delete", default=False)},
            disabled=[c for c in df_view.columns if c != "Delete"],
            key=f"qa_dataset_editor_{st.session_state.qa_editor_counter}",
        )
        to_delete = edited[edited["Delete"] == True]
        if not to_delete.empty:
            if st.button(
                f"Delete {len(to_delete)} selected Q&A(s)",
                type="primary",
                key="delete_qas_btn",
            ):
                now = datetime.now().strftime("%Y-%m-%d %H:%M")
                new_excl = pd.DataFrame([
                    {
                        "question_norm": _norm_q(row["question"]),
                        "original_question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "source_file": row.get("source_file", ""),
                        "excluded_at": now,
                    }
                    for _, row in to_delete.iterrows()
                ])
                if EXCLUDED_PATH.exists():
                    ex = _load_csv_safe(str(EXCLUDED_PATH))
                    existing_norms = set(ex["question_norm"]) if not ex.empty and "question_norm" in ex.columns else set()
                    new_excl = new_excl[~new_excl["question_norm"].isin(existing_norms)]
                    if not new_excl.empty:
                        new_excl.to_csv(EXCLUDED_PATH, mode="a", header=False, index=False)
                else:
                    new_excl.to_csv(EXCLUDED_PATH, index=False)
                try:
                    from src.ingestion import load_dataset
                    from src.retrieval import build_index
                    with st.spinner("Removing Q&As and updating knowledge base..."):
                        load_dataset.load_dataset()
                        build_index.build_index()
                    st.cache_resource.clear()
                    st.session_state.qa_editor_counter += 1
                    _queue_message("qa_dataset_action", f"{len(to_delete)} Q&A(s) removed from the knowledge base.")
                except Exception as e:
                    st.error(f"Removal failed: {e}")
                st.rerun()

    # ── Tab 2: Deleted Q&As ───────────────────────────────────────────────────
    with tab_deleted:
        st.caption("Q&As removed from the knowledge base. They remain in their source files but are excluded. Restore to make them active again.")
        _show_message("restore_action")

        if not EXCLUDED_PATH.exists():
            st.info("No Q&As have been deleted yet.")
        else:
            df_excl = _load_csv_safe(str(EXCLUDED_PATH))
            if df_excl.empty:
                st.info("No Q&As have been deleted yet.")
            else:
                st.write(f"**{len(df_excl)} deleted Q&A(s)**")
                df_display = df_excl.copy()
                df_display.insert(0, "Restore", False)
                edited_excl = st.data_editor(
                    df_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Restore": st.column_config.CheckboxColumn("Restore", default=False)},
                    disabled=[c for c in df_display.columns if c != "Restore"],
                    key="deleted_qas_editor",
                )
                to_restore = edited_excl[edited_excl["Restore"] == True]
                if not to_restore.empty:
                    if st.button(
                        f"Restore {len(to_restore)} selected Q&A(s)",
                        type="primary",
                        key="restore_qas_btn",
                    ):
                        remaining = df_excl[~df_excl["question_norm"].isin(to_restore["question_norm"])]
                        remaining.to_csv(EXCLUDED_PATH, index=False)
                        try:
                            from src.ingestion import load_dataset
                            from src.retrieval import build_index
                            with st.spinner("Restoring Q&As and updating knowledge base..."):
                                load_dataset.load_dataset()
                                build_index.build_index()
                            st.cache_resource.clear()
                            _queue_message("restore_action", f"{len(to_restore)} Q&A(s) restored.")
                        except Exception as e:
                            st.error(f"Restore failed: {e}")
                        st.rerun()


# ── Navigation ────────────────────────────────────────────────────────────────

PAGES = {
    "Documents": page_documents,
    "Knowledge Base": page_knowledge_base,
    "Testing": page_testing,
    "Evaluation": page_evaluation,
    "Settings": page_settings,
}

with st.sidebar:
    st.title("Admin")
    page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

    # ── System status strip ───────────────────────────────────────────────────
    st.divider()

    _kb = _load_csv_safe("data/processed/qa_dataset_clean.csv")
    _kb_size = len(_kb)
    _log = _load_csv_safe("data/processed/query_log.csv")
    _query_count = len(_log)
    _drafts = _load_csv_safe("data/processed/draft_qas.csv")
    _pending = int((_drafts["review_status"] == "pending").sum()) if not _drafts.empty and "review_status" in _drafts.columns else 0

    st.caption("**System status**")
    st.caption(f"{'✓' if _kb_size > 0 else '✗'} KB: {_kb_size} Q&As")
    st.caption(f"Queries logged: {_query_count}")
    if _pending > 0:
        st.warning(f"{_pending} Q&A(s) pending review")

PAGES[page]()
