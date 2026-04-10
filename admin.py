import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import find_dotenv, set_key

from src.utils.config import settings

st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="\U0001f527",
    layout="wide",
)

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

# ── Cached retriever ──────────────────────────────────────────────────────────

@st.cache_resource
def get_admin_retriever():
    try:
        from src.retrieval.hybrid_query import HybridRetriever
        return HybridRetriever()
    except FileNotFoundError:
        return None


# ── Helper utilities ──────────────────────────────────────────────────────────

def _load_csv_safe(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        try:
            return pd.read_csv(p)
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


# ── Page: Overview ────────────────────────────────────────────────────────────

def page_overview():
    st.header("Overview")
    log = _load_csv_safe("data/processed/query_log.csv")

    if log.empty:
        st.info("No query data yet. The log will populate once users start asking questions.")
        return

    total = len(log)
    qa_rate = (log["mode"] == "qa_answer").sum() / total * 100 if total else 0
    chunk_rate = (log["mode"] == "chunk_fallback").sum() / total * 100 if total else 0
    llm_rate = (log["mode"] == "llm_fallback").sum() / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries", total)
    c2.metric("Q&A Match Rate", f"{qa_rate:.1f}%")
    c3.metric("Chunk Fallback Rate", f"{chunk_rate:.1f}%")
    c4.metric("LLM Fallback Rate", f"{llm_rate:.1f}%")

    st.subheader("Mode Distribution")
    st.bar_chart(log["mode"].value_counts())

    st.subheader("Recent Queries")
    recent = log.tail(20).copy()
    if "query" in recent.columns:
        recent["query"] = recent["query"].str[:60]
    display_cols = [c for c in ["timestamp", "query", "mode", "qa_score"] if c in recent.columns]
    st.dataframe(recent[display_cols], use_container_width=True)


# ── Page: Documents ───────────────────────────────────────────────────────────

def page_documents():
    st.header("Documents")

    if "pdf_upload_counter" not in st.session_state:
        st.session_state.pdf_upload_counter = 0
    if "csv_upload_counter" not in st.session_state:
        st.session_state.csv_upload_counter = 0

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

    tab_pdf, tab_csv = st.tabs(["PDF Pipeline", "Q&A CSV"])

    # ── Tab 1: PDF Pipeline ───────────────────────────────────────────────────
    with tab_pdf:
        st.subheader("PDF Documents")

        uploaded = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            label_visibility="collapsed",
            key=f"pdf_uploader_{st.session_state.pdf_upload_counter}",
        )
        if uploaded:
            (raw_docs / uploaded.name).write_bytes(uploaded.getvalue())
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
                        p.unlink()
                        st.session_state.pop(ck, None)
                        remaining = set(f.name for f in raw_docs.glob("*.pdf"))
                        _cleanup_orphaned_qas(remaining, drafts_path, approved_path)
                        if not remaining:
                            for stale in [chunks_path, chunk_index_path, Path("data/processed/chunk_meta.json")]:
                                if stale.exists():
                                    stale.unlink()
                            st.cache_resource.clear()
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

        # Step 1
        st.markdown("**Step 1 — Extract Text Chunks**")
        st.caption("Splits PDFs into passages. Re-run only after uploading new PDFs. Runs synchronously — do not navigate away until complete.")
        if not pdf_files:
            st.info("Upload a PDF first.")
            st.button("Extract Chunks", disabled=True, use_container_width=True, key="extract_no_pdf")
        elif chunks_up_to_date:
            st.success(f"{chunk_count} chunks — up to date.")
            st.button("Extract Chunks", disabled=True, use_container_width=True, key="extract_ok")
        else:
            if chunk_count > 0:
                df_chk2 = _load_csv_safe(str(chunks_path))
                chunked_src = set(df_chk2["source_file"].dropna().unique()) if not df_chk2.empty and "source_file" in df_chk2.columns else set()
                current_src = set(p.name for p in pdf_files)
                removed = chunked_src - current_src
                added = current_src - chunked_src
                reasons = []
                if removed:
                    reasons.append(f"{len(removed)} PDF(s) removed")
                if added:
                    reasons.append(f"{len(added)} new PDF(s) added")
                st.warning("; ".join(reasons) + " since last extraction.")
            else:
                st.info("Not extracted yet.")
            if st.button("Extract Chunks from All PDFs", use_container_width=True, type="primary", key="extract_btn"):
                try:
                    from src.ingestion import extract_documents
                    with st.spinner("Extracting..."):
                        extract_documents.extract_documents()
                    _cleanup_orphaned_qas(set(p.name for p in pdf_files), drafts_path, approved_path)
                    st.rerun()
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

        st.divider()

        # Step 2
        st.markdown("**Step 2 — Build Chunk Index**")
        st.caption("Makes extracted chunks searchable. Rebuild only after re-extracting. Runs synchronously — do not navigate away until complete.")
        if chunk_count == 0:
            st.info("Extract chunks first.")
            st.button("Build Chunk Index", disabled=True, use_container_width=True, key="chunk_idx_no_chunks")
        elif chunk_index_up_to_date:
            st.success(f"Up to date. Last built: {_mtime_label(str(chunk_index_path))}")
            st.button("Build Chunk Index", disabled=True, use_container_width=True, key="chunk_idx_ok")
        else:
            st.warning("Out of date — rebuild to reflect latest chunks.")
            if st.button("Build Chunk Index", use_container_width=True, type="primary", key="chunk_idx_btn"):
                try:
                    from src.retrieval import build_chunk_index
                    with st.spinner("Building..."):
                        build_chunk_index.build_chunk_index()
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Build failed: {e}")

        st.divider()

        # Step 3
        st.markdown("**Step 3 — Generate Draft Q&As** *(optional)*")
        st.caption(
            "Uses OpenAI to generate Q&A pairs from chunks. This can take several minutes. "
            "Do not navigate away or refresh — if interrupted, re-running will resume from where it left off."
        )
        if chunk_count == 0:
            st.info("Extract chunks first.")
            st.button("Generate Draft Q&As", disabled=True, use_container_width=True, key="gen_no_chunks")
        elif unprocessed == 0:
            st.success("All chunks processed.")
            st.button("Generate Draft Q&As", disabled=True, use_container_width=True, key="gen_done")
        else:
            st.warning(
                f"{unprocessed} chunk(s) will be sent to OpenAI — this incurs costs. "
                "Stay on this page until complete."
            )
            confirmed = st.checkbox("I understand this will use the OpenAI API", key="generate_confirmed")
            if st.button("Generate Draft Q&As", disabled=not confirmed, use_container_width=True, type="primary", key="gen_btn"):
                try:
                    from src.ingestion import generate_draft_qas
                    with st.spinner(f"Generating Q&A pairs for {unprocessed} chunk(s)... Do not navigate away."):
                        generate_draft_qas.generate_drafts()
                    st.success("Generation complete — go to PDF Q&A Review to approve pairs.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Generation failed: {e}")

        if pending_drafts > 0:
            st.info(f"**{pending_drafts} draft Q&A(s) ready for review.** Open **PDF Q&A Review** in the sidebar.")

    # ── Tab 2: Q&A CSV ────────────────────────────────────────────────────────
    with tab_csv:
        st.subheader("Q&A CSV Upload")
        st.caption("Upload a CSV with `question` and `answer` columns to add pairs directly to the knowledge base.")

        uploaded_qa = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            label_visibility="collapsed",
            key=f"kb_csv_upload_{st.session_state.csv_upload_counter}",
        )
        if uploaded_qa:
            try:
                preview_df = pd.read_csv(io.BytesIO(uploaded_qa.getvalue()))
                if "question" not in preview_df.columns or "answer" not in preview_df.columns:
                    st.error("CSV must have `question` and `answer` columns.")
                else:
                    st.dataframe(preview_df.head(5), use_container_width=True)
                    if st.button("Save this CSV", key="kb_save_csv"):
                        (Path("data/raw") / uploaded_qa.name).write_bytes(uploaded_qa.getvalue())
                        st.session_state.csv_upload_counter += 1
                        st.rerun()
            except Exception as e:
                st.error(f"Could not read file: {e}")

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
                        p.unlink()
                        st.session_state.pop(ck, None)
                        st.rerun()
                else:
                    if c3.button("Delete", key=f"del_csv_{p.name}"):
                        st.session_state[ck] = True
                        st.rerun()
            if any(st.session_state.get(f"confirm_del_csv_{p.name}") for p in raw_csvs_display):
                st.caption("Click **Confirm** to permanently delete.")
        else:
            st.info("No CSV files uploaded yet.")

    # ── Shared: Build Q&A Index ───────────────────────────────────────────────
    st.divider()
    _section_qa_index()


def _section_qa_index():
    st.subheader("Final Step — Build Q&A Index")
    st.caption(
        "This index is what the assistant searches first. "
        "It combines approved PDF Q&As (from the PDF Pipeline tab) and manually uploaded CSVs (from the Q&A CSV tab). "
        "Rebuild it after approving new Q&As in **PDF Q&A Review**, or after adding/removing a CSV. "
        "Runs synchronously — do not navigate away until complete."
    )

    qa_index_path = Path("data/processed/index.faiss")
    approved_path = Path("data/raw/approved_qas.csv")

    # Sources summary
    manual_csvs = [p for p in Path("data/raw").glob("*.csv") if p.name != "approved_qas.csv"]
    manual_rows = sum(_row_count(str(p)) for p in manual_csvs)
    approved_rows = _row_count(str(approved_path)) if approved_path.exists() else 0
    total_rows = manual_rows + approved_rows

    src1, src2, src3 = st.columns(3)
    src1.metric("Approved PDF Q&As", approved_rows)
    src2.metric("Manual CSV rows", manual_rows)
    src3.metric("Index last built", _mtime_label(str(qa_index_path)))

    if total_rows == 0:
        st.info(
            "No Q&A data yet. Upload PDFs and approve generated pairs, or upload a CSV above."
        )
        st.button("Build Q&A Index", disabled=True, use_container_width=True, key="build_qa_disabled")
        return

    all_source_paths = list(Path("data/raw").glob("*.csv"))
    newest_source_mtime = max(_mtime(str(p)) for p in all_source_paths) if all_source_paths else 0.0
    qa_index_up_to_date = (
        qa_index_path.exists()
        and _mtime(str(qa_index_path)) >= newest_source_mtime
    )

    if qa_index_up_to_date:
        indexed_count = _row_count("data/processed/qa_dataset_clean.csv")
        st.success(
            f"Q&A index is up to date — **{indexed_count} pairs** indexed. "
            "Rebuild only after adding new data."
        )
    else:
        st.warning("Q&A index is out of date — new data is available. Rebuild to include it.")

    if st.button(
        "Rebuild Q&A Index" if qa_index_up_to_date else "Build Q&A Index",
        use_container_width=True,
        type="primary",
        disabled=qa_index_up_to_date,
        key="build_qa_enabled",
    ):
        try:
            from src.ingestion import load_dataset
            from src.retrieval import build_index
            with st.spinner("Step 1/2 — Merging and cleaning Q&A data..."):
                load_dataset.load_dataset()
            with st.spinner("Step 2/2 — Building search index..."):
                build_index.build_index()
            st.success("Q&A index built. The assistant will now use the updated knowledge base.")
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Build failed: {e}")

    df = _load_csv_safe("data/processed/qa_dataset_clean.csv")
    if not df.empty:
        with st.expander("Preview current Q&A dataset"):
            filter_text = st.text_input("Filter by question", key="kb_filter")
            if filter_text:
                df = df[df["question"].str.contains(filter_text, case=False, na=False)]
            st.dataframe(df, use_container_width=True)


# ── Page: PDF Q&A Review ──────────────────────────────────────────────────────

def page_draft_review():
    st.header("PDF Q&A Review")
    st.caption(
        "Q&A pairs on this page were automatically generated from your uploaded PDFs (via Documents → PDF Documents → Step 3). "
        "Review each pair and approve or reject it. **Approved pairs are saved to `data/raw/approved_qas.csv`** "
        "and will be included in the Q&A index the next time you rebuild it from the Documents page."
    )

    if "draft_skipped" not in st.session_state:
        st.session_state.draft_skipped = set()

    drafts_path = Path("data/processed/draft_qas.csv")
    approved_path = Path("data/raw/approved_qas.csv")

    if not drafts_path.exists():
        st.info(
            "No draft Q&As found. "
            "Upload PDFs in **Documents → PDF Documents** and run Step 3 (Generate Draft Q&As)."
        )
        return

    df = _load_csv_safe(str(drafts_path))
    if df.empty:
        st.info("Draft Q&A file is empty.")
        return

    if "review_status" not in df.columns:
        df["review_status"] = "pending"

    # Exclude auto-skipped chunks (no valid pairs generated) from review counts
    reviewable = df[df["review_status"] != "skipped"]
    total = len(reviewable)
    reviewed = int((reviewable["review_status"].isin(["approved", "rejected"])).sum())
    pending_count = int((reviewable["review_status"] == "pending").sum())

    st.write(f"**{reviewed} of {total} reviewed — {pending_count} pending**")
    st.progress(reviewed / total if total else 0)

    tabs = st.tabs(["Pending", "Approved", "Rejected", "All"])

    # ── Pending tab ───────────────────────────────────────────────────────────
    with tabs[0]:
        pending_df = df[
            (df["review_status"] == "pending")
            & (~df.index.isin(st.session_state.draft_skipped))
        ].copy()

        if pending_df.empty:
            st.success("All pending items have been reviewed or skipped.")
        else:
            st.caption(
                f"{len(pending_df)} pair(s) awaiting review. "
                "Use checkboxes to select multiple items, then approve or reject them in bulk. "
                "Or use the individual buttons on each card."
            )

            # Collect which items are currently checked (from previous rerun)
            select_all = st.checkbox("Select all", key="select_all_pending")
            if select_all:
                selected = list(pending_df.index)
            else:
                selected = [idx for idx in pending_df.index if st.session_state.get(f"chk_{idx}", False)]

            # Bulk action buttons (read state set on previous rerun)
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
                            "section": "",
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
                    st.rerun()

            st.divider()

            # Individual cards with checkboxes
            for row_idx, row in pending_df.iterrows():
                col_chk, col_content = st.columns([0.05, 0.95])
                with col_chk:
                    st.checkbox(
                        "",
                        key=f"chk_{row_idx}",
                        value=select_all or st.session_state.get(f"chk_{row_idx}", False),
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
                                "section": "",
                            }])
                            if approved_path.exists():
                                approved_row.to_csv(approved_path, mode="a", header=False, index=False)
                            else:
                                approved_row.to_csv(approved_path, index=False)
                            st.rerun()
                    with ib2:
                        if st.button("Reject", key=f"reject_{row_idx}"):
                            df.loc[row_idx, "review_status"] = "rejected"
                            df.to_csv(drafts_path, index=False)
                            st.rerun()
                    with ib3:
                        if st.button("Skip", key=f"skip_{row_idx}"):
                            st.session_state.draft_skipped.add(row_idx)
                            st.rerun()
                st.divider()

    # ── Approved tab ──────────────────────────────────────────────────────────
    with tabs[1]:
        approved = df[df["review_status"] == "approved"]
        if approved.empty:
            st.info("No approved Q&As yet.")
        else:
            st.caption(
                f"{len(approved)} pair(s) approved. "
                "These are saved in `data/raw/approved_qas.csv` and will be included "
                "the next time you rebuild the Q&A index."
            )
            st.dataframe(approved, use_container_width=True)

    # ── Rejected tab ──────────────────────────────────────────────────────────
    with tabs[2]:
        rejected = df[df["review_status"] == "rejected"]
        if rejected.empty:
            st.info("No rejected Q&As yet.")
        else:
            st.dataframe(rejected, use_container_width=True)

    # ── All tab ───────────────────────────────────────────────────────────────
    with tabs[3]:
        st.dataframe(df, use_container_width=True)


# ── Page: Settings ────────────────────────────────────────────────────────────

def page_settings():
    st.header("Settings")
    st.caption("Save to update the `.env` file. Restart the app to apply changes.")

    st.subheader("Current Configuration")
    st.write(f"**Embedding Model:** {settings.embedding_model}")
    st.write(f"**Top-K (Q&A):** {settings.top_k}")
    st.write(f"**Top-K (Chunk):** {settings.chunk_top_k}")
    st.write(f"**LLM Fallback Enabled:** {settings.llm_fallback_enabled}")
    st.write(f"**OpenAI Model:** {settings.openai_model}")

    st.subheader("Edit Settings")

    new_qa_threshold = st.slider(
        "Q&A Similarity Threshold",
        min_value=0.0, max_value=1.0,
        value=float(settings.similarity_threshold),
        step=0.01, key="settings_qa_threshold",
    )
    new_chunk_threshold = st.slider(
        "Chunk Similarity Threshold",
        min_value=0.0, max_value=1.0,
        value=float(settings.chunk_similarity_threshold),
        step=0.01, key="settings_chunk_threshold",
    )
    new_llm_enabled = st.toggle(
        "LLM Fallback Enabled",
        value=settings.llm_fallback_enabled,
        key="settings_llm_enabled",
    )
    new_openai_model = st.text_input(
        "OpenAI Model",
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
            st.success("Settings saved. Restart the app to apply changes.")
        except Exception as e:
            st.error(f"Failed to save settings: {e}")

    st.divider()

    with st.expander("Test a Single Query"):
        st.caption(
            "Run a query against the live retriever to see exactly which layer handles it "
            "and what scores are returned. Useful for tuning thresholds."
        )
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
                st.write(f"**Mode:** `{result.get('mode')}`")
                with st.expander("Q&A Layer", expanded=True):
                    qa = result.get("qa_result", {})
                    st.write(f"**ok:** {qa.get('ok')}  |  **reason:** {qa.get('reason', 'n/a')}")
                    st.write(f"Threshold applied: {settings.similarity_threshold}")
                    if qa.get("alternatives"):
                        rows = [
                            {
                                "score": r["score"],
                                "matched_question": r["matched_question"][:80],
                                "answer": r["answer"][:80],
                            }
                            for r in qa["alternatives"]
                        ]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    elif qa.get("best_candidate"):
                        bc = qa["best_candidate"]
                        st.write(
                            f"Best candidate score: {bc['score']:.4f} — "
                            f"{bc.get('matched_question', '')[:80]}"
                        )
                with st.expander("Chunk Layer"):
                    cr = result.get("chunk_result", {})
                    st.write(f"**ok:** {cr.get('ok')}  |  **reason:** {cr.get('reason', 'n/a')}")
                    st.write(f"Threshold applied: {settings.chunk_similarity_threshold}")
                    if cr.get("results"):
                        rows = [
                            {
                                "score": r["score"],
                                "chunk_id": r.get("chunk_id", ""),
                                "text": r["text"][:100],
                            }
                            for r in cr["results"]
                        ]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    elif cr.get("best"):
                        st.write(f"Best chunk score: {cr['best']['score']:.4f}")
                with st.expander("LLM Layer"):
                    llm = result.get("llm_result")
                    if llm:
                        st.write(
                            f"**ok:** {llm.get('ok')}  |  "
                            f"**model:** {llm.get('model')}  |  "
                            f"**latency:** {llm.get('latency_ms')} ms"
                        )
                        st.write(llm.get("answer", ""))
                    else:
                        st.write("LLM was not called.")

    with st.expander("Batch Test Runner"):
        st.caption(
            "Upload a CSV with `query` and `expected_answer_id` columns to evaluate retrieval accuracy "
            "across multiple queries at once."
        )
        retriever = get_admin_retriever()
        uploaded_test = st.file_uploader(
            "Upload test_queries.csv", type=["csv"], key="eval_batch_upload"
        )
        if uploaded_test:
            try:
                test_df = pd.read_csv(uploaded_test)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                test_df = pd.DataFrame()

            if not test_df.empty:
                st.dataframe(test_df.head(5), use_container_width=True)
                if st.button("Run Batch Test", key="eval_batch_run"):
                    if retriever is None:
                        st.error("Retriever not available — build the indexes first.")
                    else:
                        _run_batch_test(retriever, test_df)

        if st.session_state.get("eval_batch_results"):
            _display_batch_results(st.session_state.eval_batch_results)

    st.divider()

    # ── Reset System ──────────────────────────────────────────────────────────
    st.subheader("Reset System")
    st.caption(
        "Permanently delete uploaded files and/or processed data to start fresh. "
        "This cannot be undone."
    )

    reset_opts = st.multiselect(
        "Select what to delete:",
        options=[
            "PDF files (data/raw_docs/)",
            "Q&A CSV files (data/raw/)",
            "Extracted chunks (document_chunks.csv)",
            "Chunk index (chunk_index.faiss)",
            "Draft Q&As (draft_qas.csv)",
            "Approved Q&As (approved_qas.csv)",
            "Q&A dataset + index (qa_dataset_clean.csv, index.faiss, meta.json)",
        ],
        key="reset_opts",
    )

    if reset_opts:
        st.warning(
            f"You have selected **{len(reset_opts)} item(s)** to delete. "
            "This is permanent and cannot be undone."
        )
        if st.button("Reset selected", type="primary", key="reset_confirm_btn"):
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

            if "PDF files (data/raw_docs/)" in reset_opts:
                _try_delete_dir_contents("data/raw_docs/*.pdf", "PDF files")
            if "Q&A CSV files (data/raw/)" in reset_opts:
                _try_delete_dir_contents("data/raw/*.csv", "Q&A CSV files")
            if "Extracted chunks (document_chunks.csv)" in reset_opts:
                _try_delete("data/processed/document_chunks.csv", "document_chunks.csv")
            if "Chunk index (chunk_index.faiss)" in reset_opts:
                _try_delete("data/processed/chunk_index.faiss", "chunk_index.faiss")
                _try_delete("data/processed/chunk_meta.json", "chunk_meta.json")
            if "Draft Q&As (draft_qas.csv)" in reset_opts:
                _try_delete("data/processed/draft_qas.csv", "draft_qas.csv")
            if "Approved Q&As (approved_qas.csv)" in reset_opts:
                _try_delete("data/raw/approved_qas.csv", "approved_qas.csv")
            if "Q&A dataset + index (qa_dataset_clean.csv, index.faiss, meta.json)" in reset_opts:
                _try_delete("data/processed/qa_dataset_clean.csv", "qa_dataset_clean.csv")
                _try_delete("data/processed/index.faiss", "index.faiss")
                _try_delete("data/processed/meta.json", "meta.json")

            st.cache_resource.clear()
            if deleted:
                st.success(f"Deleted: {', '.join(deleted)}")
            if errors:
                st.error(f"Errors: {'; '.join(errors)}")
            st.rerun()


# ── Page: Query Log ───────────────────────────────────────────────────────────

def page_query_log():
    st.header("Query Log")

    tab_all, tab_unanswered = st.tabs(["All Queries", "Unanswered Queries"])

    with tab_all:
        log = _load_csv_safe("data/processed/query_log.csv")
        if log.empty:
            st.info("No query log data yet.")
        else:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                modes = ["All"] + sorted(log["mode"].dropna().unique().tolist())
                mode_filter = st.selectbox("Mode", modes, key="log_mode_filter")
            with fc2:
                if "timestamp" in log.columns:
                    log["timestamp"] = pd.to_datetime(log["timestamp"], errors="coerce")
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
                filtered = filtered[
                    filtered["qa_score"].fillna(0).astype(float) >= min_score
                ]

            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Total Shown", len(filtered))
            avg_qa = filtered["qa_score"].dropna().astype(float).mean() if "qa_score" in filtered.columns else 0
            avg_chunk = filtered["chunk_score"].dropna().astype(float).mean() if "chunk_score" in filtered.columns else 0
            sm2.metric("Avg QA Score", f"{avg_qa:.4f}" if avg_qa else "n/a")
            sm3.metric("Avg Chunk Score", f"{avg_chunk:.4f}" if avg_chunk else "n/a")

            st.dataframe(filtered, use_container_width=True)
            st.download_button(
                "Export Filtered Results",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="query_log_filtered.csv",
                mime="text/csv",
            )

    with tab_unanswered:
        unanswered = _load_csv_safe("data/processed/unanswered_log.csv")
        if unanswered.empty:
            st.info("No unanswered queries logged yet.")
        else:
            st.caption(
                "These queries failed both retrieval layers. "
                "Use them to identify gaps in your knowledge base."
            )
            st.dataframe(unanswered, use_container_width=True)
            st.download_button(
                "Download Unanswered Log",
                data=unanswered.to_csv(index=False).encode("utf-8"),
                file_name="unanswered_log.csv",
                mime="text/csv",
            )


# ── Batch test helpers ────────────────────────────────────────────────────────

def _run_batch_test(retriever, test_df: pd.DataFrame, qa_top_k=None, chunk_top_k=None, key_suffix=""):
    results = []
    progress = st.progress(0)
    total = len(test_df)

    for i, row in enumerate(test_df.itertuples(), start=1):
        query = str(getattr(row, "query", ""))
        expected = str(getattr(row, "expected_answer_id", ""))

        kwargs = {}
        if qa_top_k is not None:
            kwargs["qa_top_k"] = qa_top_k
        if chunk_top_k is not None:
            kwargs["chunk_top_k"] = chunk_top_k

        result = retriever.search(query, **kwargs)
        mode = result.get("mode", "")

        if mode == "qa_answer":
            got_id = result["qa_result"]["best"].get("id", "")
            qa_score = result["qa_result"]["best"].get("score")
            correct = "\u2713" if got_id == expected else "\u2717"
        else:
            got_id = ""
            qa_score = None
            if result.get("qa_result", {}).get("best_candidate"):
                qa_score = result["qa_result"]["best_candidate"].get("score")
            correct = "\u2014"

        results.append({
            "query": query[:60],
            "expected": expected,
            "got_id": got_id,
            "mode": mode,
            "score": qa_score,
            "correct": correct,
            "qa_score": qa_score,
        })
        progress.progress(i / total)

    st.session_state.eval_batch_results = results
    st.session_state.eval_batch_df = test_df


def _display_batch_results(results: list):
    results_df = pd.DataFrame(results)[["query", "expected", "got_id", "mode", "score", "correct"]]
    st.dataframe(results_df, use_container_width=True)

    total = len(results)
    qa_attempts = [r for r in results if r.get("mode") == "qa_answer"]
    correct_results = [r for r in qa_attempts if r.get("correct") == "\u2713"]
    incorrect_results = [r for r in qa_attempts if r.get("correct") == "\u2717"]

    top1_acc = len(correct_results) / max(len(qa_attempts), 1) * 100
    qa_rate = len(qa_attempts) / total * 100
    chunk_rate = sum(1 for r in results if r.get("mode") == "chunk_fallback") / total * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Top-1 Accuracy", f"{top1_acc:.1f}%")
    m2.metric("Q&A Match Rate", f"{qa_rate:.1f}%")
    m3.metric("Chunk Fallback Rate", f"{chunk_rate:.1f}%")

    st.download_button(
        "Download Results CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="eval_results.csv",
        mime="text/csv",
    )


# ── Navigation ────────────────────────────────────────────────────────────────

PAGES = {
    "Overview": page_overview,
    "Documents": page_documents,
    "PDF Q\u0026A Review": page_draft_review,
    "Settings": page_settings,
    "Query Log": page_query_log,
}

with st.sidebar:
    st.title("Admin")
    page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

    # Pending review nudge in sidebar
    drafts_path = Path("data/processed/draft_qas.csv")
    if drafts_path.exists():
        df_check = _load_csv_safe(str(drafts_path))
        if not df_check.empty and "review_status" in df_check.columns:
            pending = int((df_check["review_status"] == "pending").sum())
            if pending > 0:
                st.warning(f"{pending} draft Q&A(s) pending review")

PAGES[page]()
