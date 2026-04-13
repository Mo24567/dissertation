# Evaluation Reference Document

This document is the single reference for your dissertation evaluation chapter. It covers every stat on the Evaluation page, every batch testing tool, and the manual tests you need to complete yourself. Nothing is included unless it directly supports a dissertation claim.

---

## The Core Argument

> **Semantic retrieval improves answer relevance over keyword-based search, while reducing reliance on LLM-generated responses.**

Every metric in this document either:
1. Provides direct evidence for that claim, or
2. Contextualises the system so the evaluation is credible (scale, methodology, honest limitations)

---

## Evaluation Page — Section by Section

---

### Section 1: Knowledge Base Pipeline

**Dissertation question answered:** *How was the knowledge base constructed, and at what scale?*

**Metrics:**

| Metric | What it is |
|---|---|
| PDF documents | Number of source PDFs uploaded |
| Text passages | Text segments extracted from PDFs for passage search |
| Q&As generated | Q&A pairs produced by AI from those passages |
| Approved | Pairs that passed human review |
| Rejected | Pairs rejected during review |
| In knowledge base | Final count after deduplication and exclusion filtering |

**Why it matters for the dissertation:** This is your methodology in numbers. It demonstrates the pipeline is non-trivial and that the system is being evaluated on real, curated content — not toy data. The pipeline caption (`X PDFs → Y passages → Z Q&As → N approved → M in knowledge base`) is a dissertation-ready summary sentence for your implementation chapter.

**Cite in:** Implementation / Methodology chapter.

**Data sources:** `data/raw_docs/`, `data/processed/document_chunks.csv`, `data/processed/draft_qas.csv`, `data/processed/qa_dataset_clean.csv`

---

### Section 2: Query Activity

**Dissertation question answered:** *What does real-world usage of the system look like?*

#### Queries over time (line chart)
Shows query volume per day. If you ran the Evaluation Feed in distinct sessions, you will see clear spikes corresponding to each run. Demonstrates the system was actively tested, not evaluated on a trivial sample.

#### How queries were answered (pie chart)
Breaks down all queries into four categories:

| Category | Internal mode | Meaning |
|---|---|---|
| Q&A Match | `qa_answer` | Answered directly by the semantic Q&A layer |
| Passage Search | `chunk_fallback` | No Q&A match; answered from a PDF passage |
| AI Fallback | `llm_fallback` | No semantic match; answered by OpenAI |
| Unanswered | `no_answer` | Failed all layers |

**Why it matters for the dissertation:** This is the headline visual. A pie chart dominated by "Q&A Match" directly supports your argument. "AI Fallback" should be small — that is the layer you argue the system reduces reliance on. Screenshot this for your results chapter.

**Cite in:** Results chapter — overview of system behaviour.

---

### Section 3: Retrieval Performance

**Dissertation question answered:** *How often does semantic search answer queries without needing AI?*

This is your **primary evidence section**.

#### Mode breakdown table
A table showing exact query counts and percentages per retrieval layer, with plain-English explanations.

**Why it matters:** This table is the core quantitative result. You would quote these percentages directly in your results chapter: *"X% of queries were answered by the semantic Q&A layer without requiring AI generation."*

#### Summary metrics
- **Handled by semantic search** — Q&A Match + Passage Search combined. Everything answered without AI hallucination risk.
- **Needed AI fallback** — your argument is that this should be low.
- **Unanswered** — complete failures across all layers.
- **Total evaluated** — establishes sample size for all percentages.

#### Q&A match rate over time (line chart)
Match rate per day. If the knowledge base was expanded mid-evaluation, this should show an upward trend — directly supporting the argument that a better-curated KB improves semantic search performance.

**Cite in:** Results chapter — core quantitative findings.

---

### Section 4: Match Quality

**Dissertation question answered:** *When semantic search matches, how confident is it?*

Addresses the key critique: *"it might match, but does it match the right thing?"*

#### Score metrics (average, median, highest, lowest)
Similarity scores range from 0 to 1. A high average (e.g. 0.85+) shows the system returns genuinely relevant answers, not just the closest approximation regardless of quality.

#### Confidence bands table

| Band | Meaning |
|---|---|
| High confidence (≥ 0.85) | Strong semantic match — very likely correct |
| Medium confidence (threshold to 0.85) | Good match, above threshold but not near-certain |
| Low confidence (< threshold) | Did NOT match — fell through to the next layer |

**Why it matters:** Shows the distribution of match quality, not just the average. If most matches cluster in the high-confidence band, that is strong evidence the semantic layer is functioning correctly. The low-confidence row is honest — it shows queries that attempted a Q&A match but failed, and is important for your limitations discussion.

#### Score distribution histogram
Bins scores in 0.05-wide intervals from 0 to 1. A distribution skewed toward 1.0 is standard evidence in information retrieval that the system returns confident, relevant matches.

#### Threshold boundary cases
- **Near-miss queries** — scored within 0.05 below threshold. These almost matched; a lower threshold would have answered them.
- **Borderline matches** — scored within 0.05 above threshold. These just barely passed.

**Why it matters:** Supports your threshold sensitivity discussion. Shows you understand the system's parameters and chose the threshold deliberately, not arbitrarily.

**Cite in:** Results chapter — quality of semantic matching.

---

### Section 5: Query Characteristics

**Dissertation question answered:** *Does semantic search handle short, natural-language queries that keyword search cannot?*

#### Query length vs match rate table
Groups queries by word count and shows Q&A match rate per group.

| Length band | Why it matters |
|---|---|
| 1–2 words | Keyword search fails here — insufficient terms for overlap |
| 3–4 words | Still weak for keyword search — meaning is ambiguous without context |
| 5–7 words | Both approaches become more viable |
| 8+ words | Keyword search recovers somewhat as more terms are available |

**Why it matters:** This is the most direct comparison point between semantic and keyword approaches. If your system achieves a reasonable match rate on short queries, you have direct evidence semantic search handles the cases keyword search cannot. Cite this table explicitly in your comparison section.

#### Repeated queries
Queries asked more than once. These are confirmed real user needs — if the system consistently answers them, that is evidence of practical utility beyond the controlled evaluation.

**Cite in:** Results chapter — argument for semantic search over keyword search.

---

### Section 6: Gap Analysis

**Dissertation question answered:** *Where does the system fail, and what does that reveal?*

An honest evaluation acknowledges failures. This section is important for your limitations and future work sections.

| Metric | What it shows |
|---|---|
| Total unanswered queries | Complete failures — no match at any layer |
| Repeated unanswered queries | Asked more than once and never answered — highest-priority gaps |

**Why it matters:** Repeated unanswered queries show real user needs the system cannot meet. The critical distinction for your dissertation: these failures indicate **knowledge base gaps**, not failures of the semantic search approach itself. A keyword search system would fail on the same queries for the same reason. This is a key argument — semantic search is not the bottleneck; content coverage is.

**Data source:** `data/processed/unanswered_log.csv`

**Cite in:** Limitations and future work sections.

---

### Section 7: Score Comparison by Mode

**Dissertation question answered:** *Do Q&A matches have higher confidence than passage matches?*

A table showing average similarity scores for each retrieval layer.

**Why it matters:** If Q&A match mode has higher average scores than passage search mode, it shows the semantic Q&A layer answers queries with greater confidence than passage search — directly countering the argument that you should "just use passage search for everything." The two layers serve different purposes and the score difference justifies the layered architecture.

**Cite in:** Results chapter — justification of system design.

---

## Batch Testing Tools — What They Produce

These tools are in the **Testing** page and produce evidence that complements the dashboard stats.

---

### Evaluation Feed

**What it does:** Fires a CSV of plain queries through the live system. Each query is logged to `query_log.csv` and populates the Evaluation page dashboard.

**CSV format:** `query` column only.

**Dissertation use:** Simulates realistic user activity so the dashboard stats reflect a meaningful sample. Run this after fixing the threshold and before your final evaluation session.

**Workflow:** Tune threshold → Reset query log → Run Evaluation Feed → Take dashboard screenshots.

---

### BM25 vs Semantic Comparison

**What it does:** Runs each query through keyword search (BM25) and the semantic Q&A layer only — no passage search or AI fallback. Checks whether each method returned the expected answer.

**CSV format:** `query` + `expected_question` (the exact question text from your knowledge base).

**Output table:**

| Column | Meaning |
|---|---|
| query | The paraphrased test query you wrote |
| expected | The KB question it should match |
| BM25 matched | The question BM25 returned |
| BM25 ✓/✗ | Whether BM25 returned the right answer |
| BM25 score | Raw BM25 score (not normalised — higher is more keyword overlap) |
| Semantic matched | The question semantic search returned |
| Semantic ✓/✗ | Whether semantic search returned the right answer |
| Semantic score | Cosine similarity score (0–1) |

**Summary metrics:** BM25 accuracy % vs Semantic accuracy % side by side.

**Dissertation use:** This is the core of your baseline comparison — the table that directly answers *"is semantic search better than keyword search?"* A higher Semantic accuracy, particularly on short and paraphrased queries, is your primary finding. Export this CSV and include the accuracy comparison in your results chapter.

**Note on BM25 scores:** BM25 scores are raw and not normalised to 0–1. They cannot be compared directly to semantic similarity scores — only the ✓/✗ column is the fair comparison.

**Cite in:** Results chapter — baseline comparison, main finding.

---

### Full System Capture

**What it does:** Runs queries through the complete pipeline (Q&A → Passage Search → AI Fallback) and captures what each layer returned. Exports a CSV with a blank `Correct?` column for you to fill in manually.

**CSV format:** `query` column only. Optional: enable AI fallback (costs OpenAI API credits).

**Output table:**

| Column | Meaning |
|---|---|
| query | The query sent |
| layer | Which layer handled it (Q&A Match / Passage Search / AI Fallback / Unanswered) |
| Q&A answer | The stored answer returned, if Q&A layer matched |
| Passage returned | The document passage returned, if Passage Search matched |
| AI answer | The generated answer, if AI Fallback was used |
| Correct? | **You fill this in** — see manual tests below |

**Dissertation use:** The layer distribution column shows how the full system behaves on your test set, independent of the dashboard's real-usage data. The exported CSV becomes your manual correctness sheet.

**Cite in:** Results chapter — full system evaluation, grounded vs ungrounded analysis.

---

## Manual Tests — What You Need to Do

These cannot be automated and must be completed by you. The Full System Capture export is the starting point for most of them.

---

### Manual Test 1: BM25 vs Semantic — Write Your Test Queries

**What to do:**
1. Browse your Knowledge Base page and identify ~25–30 Q&A pairs across different topics
2. For each, write a paraphrased version of the question — how a student would naturally ask it, not how it appears in the KB
3. Save as a CSV with `query` and `expected_question` columns
4. Run through the BM25 vs Semantic tool

**What to look for:**
- Queries where BM25 fails but Semantic succeeds — these are your strongest evidence
- Short queries (1–4 words) are the most compelling — include plenty of these
- Include at least some where both fail — intellectual honesty

**Why it matters:** This is the controlled experiment at the heart of your dissertation. The paraphrasing is essential — if you use the exact KB questions, semantic search will score near-perfect and the comparison is meaningless.

---

### Manual Test 2: Full System Correctness — Q&A Layer

**What to do:**
1. From the Full System Capture export, filter rows where `layer = "Q&A Match"`
2. For each, read the `Q&A answer` returned
3. Mark `Correct?` as: **Correct** / **Partially correct** / **Incorrect**

**What "correct" means here:** Does the answer actually address what the query was asking? A match can be semantically similar but still return a slightly off-topic answer.

**Why it matters:** Accuracy (did it match the right pair?) and correctness (is that answer actually right?) are different. The batch tester measures accuracy automatically; you measure correctness manually here.

---

### Manual Test 3: Full System Correctness — Passage Search (Grounded)

**What to do:**
1. From the Full System Capture export, filter rows where `layer = "Passage Search"`
2. For each, read the `Passage returned`
3. Mark `Correct?` as: **Relevant** / **Partially relevant** / **Not relevant**

**What "relevant" means here:** Does the passage contain information that could actually answer the query? It doesn't need to be a perfect answer — just relevant source material.

**Why it matters:** These are **grounded** responses — the LLM (if triggered) generates from retrieved document context, not from memory. Relevant passages = low hallucination risk. This feeds your grounded vs ungrounded analysis.

---

### Manual Test 4: Full System Correctness — AI Fallback (Ungrounded)

**What to do:**
1. From the Full System Capture export, filter rows where `layer = "AI Fallback"`
2. For each, read the `AI answer` returned
3. Mark `Correct?` as: **Correct** / **Generic (not Loughborough-specific)** / **Hallucinated (factually wrong)**

**What each category means:**
- **Correct** — the AI happened to give accurate information
- **Generic** — the answer is plausible but not specific to Loughborough (e.g. "most universities offer...")
- **Hallucinated** — the AI stated something factually wrong or invented

**Why it matters:** These are **ungrounded** responses — the LLM has no retrieved context. Compare the correctness rate here against Manual Test 3 (grounded). The gap between the two is your hallucination evidence. This directly validates your claim that LLMs hallucinate on domain-specific questions without retrieval.

**Dissertation framing:** *"Grounded responses (Passage Search + LLM with context) achieved X% correctness. Ungrounded responses (AI Fallback, no context) achieved Y% correctness. The gap demonstrates that retrieval-augmented generation reduces hallucination risk compared to pure LLM generation."*

---

### Manual Test 5: Threshold Justification

**What to do:**
1. Before finalising your threshold, run the Evaluation Feed at two or three different threshold values (e.g. 0.5, 0.6, 0.7)
2. Reset the query log between each run
3. Note the Q&A match rate and near-miss count from the Evaluation page each time
4. Choose the threshold that balances match rate against avoiding incorrect low-confidence matches

**Why it matters:** You need to justify your chosen threshold in the dissertation rather than just stating a number. This is your sensitivity analysis. Even a simple two-point comparison ("at 0.6, match rate was X% but borderline matches increased; at 0.7, match rate dropped to Y% but confidence was consistently high") is sufficient.

**Cite in:** Implementation chapter — system configuration and parameter choice.

---

## Grounded vs Ungrounded — How to Write It Up

This is your hallucination validation section.

**The argument structure:**
1. LLMs are known to hallucinate on domain-specific questions (cite literature)
2. Your system uses retrieval to ground responses where possible
3. Evidence: grounded responses (Manual Test 3) achieved higher correctness than ungrounded (Manual Test 4)
4. Conclusion: retrieval-augmented generation reduces hallucination risk — justifying the system's architecture

**The two modes to compare:**
- **Grounded** = Passage Search layer, where LLM generates from retrieved document context
- **Ungrounded** = AI Fallback layer, where LLM generates with no retrieved context

You do not need a separate "LLM-only" test mode. Your test set will naturally produce both types from the Full System Capture, and comparing them is sufficient.

---

## Evaluation Workflow — In Order

1. **Build and finalise the knowledge base** — all documents uploaded, Q&As approved, index built
2. **Write your test queries** — ~25–30 paraphrased questions with `expected_question` column
3. **Tune the threshold** — run Evaluation Feed at different settings, watch match rate on Evaluation page, pick the best value (Manual Test 5)
4. **Reset the query log** — Settings → Reset → Query log only
5. **Run Evaluation Feed** — your final query set at the chosen threshold, populates the dashboard
6. **Run BM25 vs Semantic** — same queries, generates the comparison table
7. **Run Full System Capture** — same queries, export CSV for manual annotation
8. **Complete manual tests** — fill in `Correct?` column for each layer (Tests 2, 3, 4)
9. **Screenshot the Evaluation page** — clean data, final threshold, all sections
10. **Export the comparison CSV** — from BM25 vs Semantic tool, use in dissertation

---

## Data Sources

| File | Contains |
|---|---|
| `data/processed/query_log.csv` | Every query: timestamp, mode, scores, source, threshold used |
| `data/processed/unanswered_log.csv` | Queries that failed all layers, with best attempted scores |
| `data/processed/document_chunks.csv` | Extracted text passages from PDFs |
| `data/processed/draft_qas.csv` | AI-generated Q&A pairs with review status |
| `data/raw/approved_qas.csv` | Admin-approved Q&A pairs |
| `data/processed/qa_dataset_clean.csv` | Final deduplicated knowledge base |
| `data/processed/meta.json` | Q&A metadata indexed by FAISS (used by both semantic and BM25 retrievers) |

---

## What Is Not Tracked (and Why)

| Thing | Why not tracked |
|---|---|
| Response latency | Not logged — would require changes to the query logger. Mention as a future work item. |
| User satisfaction | Out of scope — would require a formal user study. Acknowledge in limitations. |
| Precision / Recall (IR definitions) | Requires a fully labelled dataset. Your batch test gives Top-1 accuracy, which is the appropriate metric for a single-answer retrieval system. |
| BM25 score normalisation | BM25 scores are raw TF-IDF-based values, not comparable to cosine similarity. Only the ✓/✗ accuracy comparison is valid across methods. |
