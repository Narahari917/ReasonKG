# ReasonKG: Knowledge Graph-Grounded RAG with Logical Consistency Checking

> A financial question-answering system that combines Neo4j knowledge graphs, FAISS vector search, SEC EDGAR document retrieval, and LLM-based claim verification to reduce hallucinations by up to 90% compared to vanilla RAG.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Evaluation Results](#evaluation-results)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**ReasonKG** is a Retrieval-Augmented Generation (RAG) system built for financial fact-checking and question answering. Unlike standard RAG pipelines that blindly trust retrieved text, ReasonKG grounds every answer in a structured Neo4j Knowledge Graph and verifies LLM-generated claims against known financial metrics.

The system was evaluated on 20 queries across 4 major companies (Apple, Microsoft, Amazon, NVIDIA) spanning fiscal years 2022–2025, and achieved a **~90% reduction in hallucination rate** compared to vanilla RAG.

---

## System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Hybrid Retrieval Layer         │
│  ┌──────────────┐   ┌─────────────────┐ │
│  │  PageIndex   │   │   KG-FAISS      │ │
│  │  (SEC EDGAR) │   │  (Neo4j + FAISS)│ │
│  └──────────────┘   └─────────────────┘ │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│         LLM Answer Generation            │
│         (ASU CreateAI / GPT-4o)          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│      Logical Consistency Checker         │
│  1. Claim Extraction                     │
│  2. KG Verification (Neo4j lookup)       │
│  3. Verdict: VERIFIED / HALLUCINATED /   │
│              UNVERIFIABLE               │
└─────────────────────────────────────────┘
    │
    ▼
 Final Verified Answer
```

---

## Key Features

- **Hybrid Retrieval**: Automatically routes between PageIndex (for AAPL/MSFT with full SEC filings) and KG-FAISS (for other companies using EDGAR XBRL + Yahoo Finance data)
- **Neo4j Knowledge Graph**: Stores structured financial metrics (revenue, net income, gross profit, operating income, EPS) for AAPL, MSFT, AMZN, NVDA from 2015 to present
- **Dual Data Sources**: SEC EDGAR XBRL API for historical annual data + Yahoo Finance for recent gaps
- **Claim-Level Verification**: LLM extracts factual claims from answers; each claim is checked against Neo4j ground truth
- **Hallucination Rate Reporting**: Every query returns a hallucination rate, verification rate, and per-claim verdicts
- **Evaluation Framework**: 20-query benchmark suite across 3 systems (Vanilla RAG, KG+PageIndex RAG, ReasonKG) with difficulty tiers (simple/medium)
- **Publication-Ready Visualizations**: 5 matplotlib figures covering overall rates, difficulty breakdown, per-company breakdown, per-query heatmap, and a summary dashboard

---

## Tech Stack

| Component | Technology |
|---|---|
| Knowledge Graph | Neo4j (AuraDB or self-hosted) |
| Vector Search | FAISS (`faiss-cpu`) |
| Embeddings | `sentence-transformers` |
| LLM | GPT-4o via ASU CreateAI API |
| Financial Data | SEC EDGAR XBRL API + Yahoo Finance (`yfinance`) |
| Document Retrieval | PageIndex + BeautifulSoup4 |
| NLP | spaCy (`en_core_web_sm`) |
| Runtime | Google Colab |
| Visualization | matplotlib |
| Storage | Google Drive |

---

## Project Structure

```
ReasonKG/
│
├── ReasonKG_Knowledge_Graph_Grounded_RAG_with_Logical_Consistency_Checking.ipynb
│                              # Main notebook — all code in sequential blocks
│
├── README.md                  # This file
│
└── (outputs saved to Google Drive at runtime)
    ├── sec_bulk/              # Downloaded SEC EDGAR filings
    ├── faiss_index/           # Persisted FAISS vector index
    ├── pageindex_doc_map.json # PageIndex document ID map
    ├── eval_results_final.csv # Full per-query evaluation data
    ├── eval_summary_final.csv # Aggregated system comparison
    ├── fig1_hallucination_rate.png
    ├── fig2_by_difficulty.png
    ├── fig3_by_company.png
    ├── fig4_heatmap.png
    └── fig5_dashboard.png
```

---

## Setup & Installation

### Prerequisites

- Google account (for Google Colab + Google Drive)
- Neo4j instance — [Neo4j AuraDB Free Tier](https://neo4j.com/cloud/aura/) recommended
- ASU CreateAI API access (or substitute your own OpenAI-compatible endpoint)
- SEC EDGAR user agent string (any `Name email@domain.com` format per [SEC policy](https://www.sec.gov/os/accessing-edgar-data))

### Step 1 — Open the Notebook in Colab

Upload `ReasonKG_Knowledge_Graph_Grounded_RAG_with_Logical_Consistency_Checking.ipynb` to Google Drive, then open it with Google Colab.

### Step 2 — Set Up Colab Secrets

In Colab, click the **key icon** in the left sidebar and add the following secrets:

| Secret Name | Description |
|---|---|
| `CREATEAI_API_URL` | Your CreateAI (or OpenAI-compatible) endpoint URL |
| `CREATEAI_API_TOKEN` | API bearer token |
| `CREATEAI_PROJECT_ID` | Project ID for your LLM deployment |
| `NEO4J_URI` | Neo4j connection URI (e.g., `neo4j+s://xxxx.databases.neo4j.io`) |
| `NEO4J_USERNAME` | Neo4j username (default: `neo4j`) |
| `NEO4J_PASSWORD` | Neo4j password |
| `NEO4J_DATABASE` | Neo4j database name (default: `neo4j`) |
| `SEC_USER_AGENT` | SEC EDGAR user agent string (e.g., `YourName your@email.com`) |

### Step 3 — Run Blocks in Order

Execute the notebook blocks sequentially (BLOCK 1 → BLOCK 2 → … → STEP 5). Each block is self-contained with clear section headers.

---

## Configuration

### Companies & Tickers

Edit `TICKERS_TO_ENRICH` in Block 4 to add or remove companies:

```python
TICKERS_TO_ENRICH = ["AAPL", "MSFT", "AMZN", "NVDA"]
```

### Financial Metrics

Edit `XBRL_CONCEPT_MAP` (SEC EDGAR) and `YFINANCE_METRIC_MAP` (Yahoo Finance) in Block 4 to pull additional GAAP metrics.

### PageIndex Document Map

Add SEC accession numbers and their PageIndex document IDs in Block 2:

```python
doc_map.update({
    "AAPL_0000320193-25-000079": "pi-xxxx...",
    "MSFT_0000950170-25-100235": "pi-xxxx...",
})
```

### Evaluation Queries

Extend `EVAL_QUERIES_FINAL` in Step 4 to add new queries, companies, or fiscal years.

---

## Usage

### Ask a Single Question

```python
# Hybrid retrieval + LLM answer
answer, method, context = hybrid_answer(
    "What was Apple's total revenue in fiscal year 2024?",
    ticker="AAPL"
)
print(f"Method: {method}")
print(f"Answer: {answer}")
```

### Full Consistency Check (ReasonKG mode)

```python
result = consistency_check(
    query="What was NVIDIA's net income in fiscal year 2025?",
    context_ticker="NVDA",
    context_year=2025,
    correct_output=True
)

print(result["final_answer"])
print(f"Hallucination rate : {result['hallucination_rate']*100:.1f}%")
print(f"Verification rate  : {result['verification_rate']*100:.1f}%")
```

### Query Knowledge Graph Directly

```python
store = Neo4jStore()

# All metrics for a company
metrics = store.query_all_metrics("AAPL")

# Specific metric
result = store.query_metric("MSFT", "revenue", 2023)
print(f"MSFT Revenue 2023: ${result['value']:,.0f}")
```

---

## Evaluation Results

Evaluated on 20 queries × 3 systems = 60 total LLM calls.

| System | Hallucination Rate | Verification Coverage |
|---|---|---|
| Vanilla RAG | ~30–40% | ~60% |
| KG + PageIndex RAG | ~10–15% | ~80% |
| **ReasonKG** | **~3–5%** | **~90%** |

**~90% reduction in hallucinations** from Vanilla RAG → ReasonKG across all difficulty levels and all four companies.

### Query Breakdown

- **Simple queries** (e.g., "What was Apple's revenue in FY2024?"): ReasonKG achieves 0% hallucination on most companies
- **Medium queries** (e.g., older fiscal years, less-indexed companies): ReasonKG consistently outperforms both baselines

---

## How It Works

### 1. Knowledge Graph Population (Block 4)

Financial metrics are loaded from two sources:

- **SEC EDGAR XBRL API**: Annual 10-K filings from 2015 to present, mapped to internal metric keys (`revenue`, `net_income`, `gross_profit`, `operating_income`, `eps_basic`)
- **Yahoo Finance**: Recent 4 fiscal years, filling gaps the XBRL API misses

Both sources write to Neo4j nodes of type `FinancialMetric` linked to `Company` nodes via `HAS_METRIC` relationships.

### 2. Document Indexing (Block 5)

SEC EDGAR HTML filings are downloaded and chunked. Each chunk is embedded using `sentence-transformers` and indexed in FAISS. For companies with PageIndex API access (AAPL, MSFT), the richer PageIndex retrieval is used instead.

### 3. Hybrid Retrieval (Block 6)

At query time, the system checks whether a PageIndex document exists for the ticker. If yes, it retrieves context via PageIndex; otherwise it uses the KG-FAISS pipeline (Neo4j metric lookup + FAISS semantic search).

### 4. LLM Answer Generation (Block 7)

The retrieved context is passed to GPT-4o with a structured prompt. The LLM generates a natural language answer grounded in the retrieved data.

### 5. Claim Verification (Block 8)

The answer is parsed into atomic factual claims (e.g., "Apple's revenue in 2024 was $391 billion"). Each claim is matched against Neo4j ground truth:

- **VERIFIED**: Claim matches KG value within tolerance
- **HALLUCINATED**: Claim contradicts KG value
- **UNVERIFIABLE**: Claim references a metric/year not in the KG

The final answer is corrected when `correct_output=True`, replacing hallucinated figures with verified KG values.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project was developed as part of CSE 579 (Knowledge Representation and Reasoning) at Arizona State University. Please contact the authors before reusing for commercial purposes.

---

*Built with Neo4j · FAISS · SEC EDGAR · GPT-4o · Google Colab*
