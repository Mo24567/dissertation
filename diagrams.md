# Architecture Diagrams

Paste each diagram into https://mermaid.live to render and export as PNG/SVG.

---

## Diagram 1: System Component Architecture

```mermaid
flowchart TD
    subgraph USER["User Interface"]
        UA["User App\n(app.py / Streamlit)"]
    end

    subgraph ADMIN["Admin Interface"]
        AA["Admin App\n(admin.py / Streamlit)"]
    end

    subgraph RETRIEVAL["Retrieval Layer"]
        HR["HybridRetriever\norchestrator"]
        QR["QARetriever\nsemantic Q&A search"]
        CR["ChunkRetriever\npassage search"]
        LLM["LLMFallback\nOpenAI GPT"]
        KR["KeywordRetriever\nBM25 baseline"]
    end

    subgraph INDEXES["Vector Indexes (FAISS)"]
        QI["Q&A Index\nindex.faiss"]
        CI["Chunk Index\nchunk_index.faiss"]
    end

    subgraph STORE["Knowledge Base"]
        KB["Q&A Dataset\nqa_dataset_clean.csv"]
        CH["Document Chunks\ndocument_chunks.csv"]
    end

    subgraph INGEST["Ingestion Pipeline"]
        PDF["PDF Documents\ndata/raw_docs/"]
        EX["extract_documents\nchunk extraction"]
        GEN["generate_draft_qas\nOpenAI GPT"]
        REV["Admin Review\napprove / reject"]
        LD["load_dataset\nbuild_index"]
    end

    subgraph LOG["Logging"]
        QL["Query Logger\nquery_log.csv"]
        UL["Unanswered Log\nunanswered_log.csv"]
    end

    UA -->|"user query"| HR
    HR --> QR
    HR --> CR
    HR --> LLM
    HR -->|"logs result"| QL
    HR -->|"logs failures"| UL

    QR <-->|"vector search"| QI
    CR <-->|"vector search"| CI
    QI --- KB
    CI --- CH

    LLM -->|"OpenAI API"| OPENAI[("OpenAI\nAPI")]

    AA -->|"manages"| INGEST
    AA -->|"reads"| QL
    AA -->|"reads"| UL
    AA -->|"evaluation only"| KR
    KR --- KB

    PDF --> EX
    EX --> CH
    EX --> GEN
    GEN -->|"draft Q&As"| REV
    REV --> LD
    LD --> KB
    LD --> QI
    LD --> CI
```

---

## Diagram 2: Query Flow (Runtime Sequence)

```mermaid
flowchart TD
    A([User submits query]) --> B[HybridRetriever]

    B --> C{QARetriever\nSemantic search\nagainst Q&A index}

    C -->|"score ≥ threshold"| D[/"Q&A Match\nReturn stored answer"/]
    C -->|"score < threshold"| E{ChunkRetriever\nSemantic search\nagainst passage index}

    E -->|"score ≥ threshold"| F[/"Passage Search\nReturn relevant passage"/]
    E -->|"score < threshold"| G{AI Fallback\nenabled?}

    G -->|"yes"| H[LLMFallback\nOpenAI GPT]
    G -->|"no"| I[/"Unanswered\nNo result returned"/]

    H -->|"with passage context"| J[/"Grounded AI answer\ngenerated from documents"/]
    H -->|"no context"| K[/"Ungrounded AI answer\ngenerated from model knowledge"/]

    D --> L[Log to query_log.csv]
    F --> L
    J --> L
    K --> L
    I --> M[Log to unanswered_log.csv]

    style D fill:#dcfce7,stroke:#16a34a,color:#15803d
    style F fill:#ede9fe,stroke:#7c3aed,color:#6d28d9
    style J fill:#ede9fe,stroke:#7c3aed,color:#6d28d9
    style K fill:#dbeafe,stroke:#2563eb,color:#1d4ed8
    style I fill:#f3f4f6,stroke:#6b7280,color:#374151
```

---

## Diagram 3: Ingestion Pipeline

```mermaid
flowchart LR
    A[/"PDF Documents"/] --> B["extract_documents\nSplit into passages"]
    B --> C[/"Document Chunks\nCSV"/]
    C --> D["generate_draft_qas\nOpenAI GPT generates\nQ&A pairs per passage"]
    D --> E[/"Draft Q&As\nCSV"/]
    E --> F["Admin Review\nApprove / Reject"]
    F -->|"approved"| G[/"Approved Q&As\nCSV"/]
    G --> H["load_dataset\nDeduplicate + clean"]
    H --> I[/"qa_dataset_clean.csv\nFinal knowledge base"/]
    I --> J["build_index\nSentenceTransformer\n+ FAISS"]
    J --> K[/"Q&A FAISS Index\nready for search"/]

    C --> L["build_chunk_index\nSentenceTransformer\n+ FAISS"]
    L --> M[/"Chunk FAISS Index\nready for passage search"/]

    style A fill:#fef9c3,stroke:#ca8a04
    style I fill:#dcfce7,stroke:#16a34a
    style K fill:#dbeafe,stroke:#2563eb
    style M fill:#dbeafe,stroke:#2563eb
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
