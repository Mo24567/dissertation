# Architecture Diagrams

Paste each diagram into https://mermaid.live to render and export as PNG/SVG.

---

## Diagram 1: System Component Architecture

```mermaid
%%{init: {'theme': 'default', 'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    subgraph USER["User Interface"]
        UA["User App (app.py · port 8501)"]
    end

    subgraph ADMIN["Admin Interface"]
        AA["Admin App (admin.py · port 8502)"]
    end

    subgraph RETRIEVAL["Retrieval Layer"]
        HR["HybridRetriever"]
        QR["QARetriever (Semantic Q&A)"]
        CR["ChunkRetriever (Passage Search)"]
        LLM["LLMFallback (OpenAI GPT)"]
    end

    KR["KeywordRetriever (BM25, testing only)"]

    subgraph INDEXES["Vector Indexes (FAISS)"]
        QI["Q&A Index (index.faiss + meta.json)"]
        CI["Chunk Index (chunk_index.faiss + chunk_meta.json)"]
    end

    subgraph LOG["Logging"]
        QL["query_log.csv"]
        UL["unanswered_log.csv"]
    end

    INGEST["Ingestion Pipeline"]
    OPENAI[("OpenAI API")]

    UA -->|"query"| HR
    HR -->|"1st"| QR
    HR -->|"2nd"| CR
    HR -->|"3rd"| LLM
    HR -->|"all queries"| QL
    HR -->|"unanswered"| UL

    QR <-->|"search"| QI
    CR <-->|"search"| CI
    CR -.->|"passage context"| LLM
    LLM --> OPENAI

    AA -->|"manages"| INGEST
    INGEST -->|"builds"| QI
    INGEST -->|"builds"| CI
    INGEST --> OPENAI

    AA -->|"reads"| QL
    AA -->|"reads"| UL
    AA -.->|"testing only"| KR

    KR -.->|"reads"| QI
```

---

## Diagram 2: Query Flow (Runtime Sequence)

```mermaid
%%{init: {'theme': 'default', 'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    A([User submits query]) --> HR[HybridRetriever]
    HR -->|"all queries"| LOG["query_log.csv"]

    HR --> QR{QARetriever: Search Q&A index}

    QR -->|"score ≥ 0.60"| QA[/"Q&A Match: Stored answer returned"/]
    QR -->|"score < 0.60"| CR{ChunkRetriever: Search passage index}

    CR -->|"score ≥ 0.45"| CH[/"Passage Match: Document text returned"/]
    CR -->|"score < 0.45"| NONE[/"No Match: All layers exhausted"/]

    NONE -.->|"unanswered"| ULOG["unanswered_log.csv"]

    QA --> CONF{"Confidence check"}
    CONF -->|"score ≥ 0.75"| T1[/"Tier 1: Single high-confidence answer"/]
    CONF -->|"score < 0.75"| T2[/"Tier 2: Ranked candidate list"/]

    CH --> T3[/"Tier 3: Passages found, no direct Q&A match"/]
    NONE --> T4[/"Tier 4: No match found"/]

    T1 -->|"generate (ungrounded)"| GGEN["LLMFallback: generate(query)"]
    T2 -->|"generate (ungrounded)"| GGEN
    T4 -->|"user clicks button"| GGEN
    T3 -->|"user clicks button"| GGRD["LLMFallback: generate(query, context=passages)"]

    GGEN --> R1[/"General AI answer (not grounded in documents)"/]
    GGRD --> R2[/"Grounded AI answer (based on retrieved passages)"/]

    style QA fill:#dcfce7,stroke:#16a34a,color:#15803d
    style T1 fill:#dcfce7,stroke:#16a34a,color:#15803d
    style T2 fill:#fef9c3,stroke:#ca8a04,color:#92400e
    style CH fill:#ede9fe,stroke:#7c3aed,color:#6d28d9
    style T3 fill:#ede9fe,stroke:#7c3aed,color:#6d28d9
    style R2 fill:#ede9fe,stroke:#7c3aed,color:#6d28d9
    style GGRD fill:#ede9fe,stroke:#7c3aed,color:#6d28d9
    style R1 fill:#dbeafe,stroke:#2563eb,color:#1d4ed8
    style GGEN fill:#dbeafe,stroke:#2563eb,color:#1d4ed8
    style NONE fill:#f3f4f6,stroke:#6b7280,color:#374151
    style T4 fill:#f3f4f6,stroke:#6b7280,color:#374151
```

---

## Diagram 3: Ingestion Pipeline

```mermaid
%%{init: {'theme': 'default', 'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    PDF[/"PDF Documents (data/raw_docs/)"/]
    CSV[/"Hand-curated Q&A CSVs (data/raw/)"/]
    EXCL[/"excluded_qas.csv (soft-deletion list)"/]

    PDF --> EXT["extract_documents (800-char chunks per page)"]
    EXT --> CHUNKS[/"document_chunks.csv"/]

    CHUNKS --> BCI["build_chunk_index (SentenceTransformer + FAISS)"]
    CHUNKS --> GEN["generate_draft_qas (OpenAI API, resumable)"]

    BCI --> CI[/"Chunk FAISS Index (chunk_index.faiss + chunk_meta.json)"/]

    GEN --> DRAFT[/"draft_qas.csv (status: pending)"/]
    DRAFT --> REV["Admin Review (approve / reject / skip)"]
    REV --> APP[/"approved_qas.csv"/]

    APP --> LD["load_dataset (merge, deduplicate, filter exclusions)"]
    CSV --> LD
    EXCL -->|"filters out excluded questions"| LD

    LD --> CLEAN[/"qa_dataset_clean.csv"/]
    CLEAN --> BI["build_index (SentenceTransformer + FAISS)"]
    BI --> QI[/"Q&A FAISS Index (index.faiss + meta.json)"/]

    style PDF fill:#fef9c3,stroke:#ca8a04
    style CSV fill:#fef9c3,stroke:#ca8a04
    style CLEAN fill:#dcfce7,stroke:#16a34a
    style QI fill:#dbeafe,stroke:#2563eb
    style CI fill:#dbeafe,stroke:#2563eb
    style EXCL fill:#fee2e2,stroke:#dc2626
```

---

## How to Export

1. Go to **https://mermaid.live**
2. Paste one diagram block (the content between the triple backticks)
3. Click **Export → PNG** or **Export → SVG**
4. SVG is better for print/PDF dissertations — scales without pixelation

## Tips for Your Dissertation

- **Diagram 1** goes in the **System Architecture** section of your implementation chapter — gives the examiner the full picture
- **Diagram 2** goes in the **Retrieval Design** section — shows you understand the runtime behaviour and the layered fallback logic
- **Diagram 3** goes in the **Knowledge Base Construction** section — illustrates the ingestion methodology
- Label figures consistently: *Figure 1: System Component Architecture*, etc.
- Reference each diagram in your text — don't just drop them in without explanation
