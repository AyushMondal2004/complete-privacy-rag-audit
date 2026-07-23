# Evaluating RAG Frameworks in Context-Aware Semantic Auditing of Mobile App Privacy Disclosures

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.5-FF6F00)

This repository contains the complete implementation, experimental framework, and results for a Retrieval-Augmented Generation (RAG) system designed to audit mobile app privacy policies. It serves as the primary technical artefact for a master's level dissertation.

The system acts as a **decision-support tool** that automatically extracts privacy practices (e.g., data collection, third-party sharing, location tracking) from lengthy privacy disclosures and maps them to severity-based risk badges.

> [!WARNING]
> **Not Legal Advice**: This system highlights potential privacy risks based on automated text retrieval and LLM generation. It does not replace professional legal review or compliance auditing. 

---

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Installation & Setup](#installation--setup)
- [Running the Experiments](#running-the-experiments)
- [Using the Web UI](#using-the-web-ui)
- [Results & Visualisations](#results--visualisations)
- [Limitations](#limitations)

---

## 🎯 Project Overview

This project systematically investigates how different chunking strategies and retrieval thresholds affect the precision, recall, and F1-score of LLM-driven privacy practice identification.

**Key Research Objectives:**
1. Design a RAG architecture capable of context-aware semantic auditing.
2. Compare chunking approaches (Fixed Size vs. Semantic Boundaries).
3. Evaluate the sensitivity of cosine-similarity thresholds.
4. Convert unstructured privacy text into structured risk badges without introducing hallucinated claims.
5. Identify the technical, practical, and ethical limitations of using RAG for privacy auditing.

---

## 🏗 System Architecture

The pipeline processes raw HTML policies into structured JSON findings via local LLM execution:

```mermaid
graph TD
    A[Raw HTML Policies (APP-350)] --> B(Ingestion & Cleaning)
    B --> C{Chunking Strategy}
    C -->|Fixed 300 words| D[Embedding Model]
    C -->|Fixed 500 words| D
    C -->|Semantic Boundaries| D
    D -->|all-MiniLM-L6-v2| E[(ChromaDB Vector Store)]
    
    E -->|Cosine Similarity Retrieval| F[Multi-Query Retriever]
    F -->|Context-Aware Prompt| G[Local LLM via Ollama]
    G -->|qwen2.5:7b| H{Structured JSON Output}
    
    H --> I[FastAPI Backend]
    I --> J[Frontend UI with Risk Badges]
    H --> K[Evaluation & Error Analysis]
```

---

## 📂 Repository Structure

The project is logically divided into source code, experimental configs, analysis scripts, and documentation:

privacy-rag-audit/
├── .env                           # Environment variables (e.g., API keys, if any)
├── .gitignore                     # Specifies files to ignore in GitHub (.venv, data/raw, etc.)
├── README.md                      # Complete project documentation and setup guide
├── requirements.txt               # Python dependencies (FastAPI, ChromaDB, Matplotlib, Pandas, etc.)
│
├── data/
│   ├── raw/
│   │   └── APP-350_v1.1/          # The original dataset (annotations and HTML files)
│   └── processed/
│       └── chroma/                # The generated ChromaDB vector database (stored locally)
│
├── docs/                          # In-depth technical documentation
│   ├── architecture.md            # System design, retrieval strategy, and pipeline map
│   └── limitation.md              # Documented technical, practical, and ethical limitations
│
├── experiments/
│   ├── configs/                   # Hyperparameter configurations for chunking
│   │   ├── fixed_300.yaml
│   │   ├── fixed_500.yaml
│   │   └── semantic.yaml
│   └── threshold_sweep/           # Contains F1 score variations for different retrieval thresholds
│
├── frontend/                      # Web user interface
│   └── index.html                 # Single-page app with visual risk severity badges
│
├── notebooks/                     # Data visualisation
│   └── 01_results_visualisation.py # Script generating the Matplotlib charts
│
├── results/                       # Experimental output data
│   ├── figures/                   # PNG charts for the dissertation
│   │   ├── category_heatmap.png
│   │   ├── config_comparison_bar.png
│   │   ├── error_breakdown.png
│   │   ├── per_policy_f1_boxplot.png
│   │   └── threshold_sensitivity.png
│   ├── error_analysis_summary.md  # Detailed markdown report of missed risks/hallucinations
│   ├── error_analysis.csv         # Raw error categorisation data
│   ├── comparison_summary.csv     # Master summary of all configs (Precision, Recall, F1)
│   ├── category_analysis.csv      # Performance metrics split by privacy categories
│   ├── semantic_per_policy.csv    # Raw LLM output metrics for semantic chunking
│   ├── fixed_300_per_policy.csv   # Raw LLM output metrics for 300-word fixed chunking
│   └── fixed_500_per_policy.csv   # Raw LLM output metrics for 500-word fixed chunking
│
├── scripts/                       # Automation scripts
│   ├── 01_build_vector_db.py      # Cleans text, chunks it, embeds it, and stores in ChromaDB
│   ├── 02_run_experiment.py       # Audits the policies against ground truth using the LLM
│   ├── 03_compare_configs.py      # Compares results and generates summary CSVs
│   └── 04_error_analysis.py       # Reads results and categorises model errors
│
└── src/                           # Backend source code modules
    ├── api/
    │   └── main.py                # FastAPI endpoints for querying and ad-hoc auditing
    ├── chunking/
    │   ├── base.py                # Base abstract class for chunking
    │   ├── fixed_chunker.py       # Overlapping word-count chunking logic
    │   ├── semantic_chunker.py    # Sentence-embedding breakpoint chunking logic
    │   └── structural_chunker.py  # Heading-based chunking logic
    ├── embeddings/
    │   └── embedder.py            # Sentence-transformers wrapper (all-MiniLM-L6-v2)
    ├── evaluation/
    │   ├── label_matcher.py       # Maps raw LLM labels to APP-350 ground truth names
    │   └── metrics.py             # Calculates True Positives, False Negatives, F1 Score
    ├── generation/
    │   ├── llm_client.py          # HTTP wrapper to query the local Ollama LLM
    │   └── prompts.py             # Context-aware structured JSON extraction prompts
    ├── ingestion/
    │   ├── html_cleaner.py        # Strips irrelevant boilerplate (scripts, navbars, footers)
    │   └── loader.py              # Pairs raw HTML files with their YAML ground truth annotations
    ├── retrieval/
    │   └── retriever.py           # Multi-category cosine-similarity search engine
    └── vectorstore/
        └── chroma_store.py        # ChromaDB connection and query execution interface


---

## 📊 Dataset

This project utilises the **APP-350 Corpus** (Zimmeck et al., 2019), provided by the Usable Privacy Policy Project. 
- Contains 350 mobile app privacy policies (HTML format).
- Includes ground-truth YAML annotations for manual evaluation and F1 scoring.
- Ensure you have checked the dataset license and research-use conditions before utilizing the data.

---

## 🚀 Installation & Setup

1. **Clone the repository** (or extract the provided ZIP).
2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare the Data**: Ensure the APP-350 corpus is located at `data/raw/APP-350_v1.1/` (containing `annotations/` and `original_documents/`).
5. **Start Local LLM**: Ensure Ollama is running locally with your chosen model. The default configuration uses `qwen2.5:7b`.

---

## 🧪 Running the Experiments

To recreate the full 349-policy experimental evaluation:

1. **Index the Vector Database** (Run for each config):
   ```bash
   python scripts/01_build_vector_db.py --config experiments/configs/semantic.yaml
   ```
2. **Execute the Evaluation Pipeline**:
   ```bash
   python scripts/02_run_experiment.py --config experiments/configs/semantic.yaml
   ```
3. **Generate Error Analysis**:
   ```bash
   python scripts/04_error_analysis.py
   ```
4. **Generate Visualisations**:
   ```bash
   python notebooks/01_results_visualisation.py
   ```
   *(Charts will be output to `results/figures/`)*

---

## 🖥 Using the Web UI

The project features a clean, responsive frontend to visualize the RAG auditing process. 

1. **Start the FastAPI Backend**:
   ```bash
   uvicorn src.api.main:app --reload
   ```
2. **Launch the Frontend**: Open `frontend/index.html` in any modern web browser.
3. **Features**:
   - **Database Querying**: Enter a Policy ID from the APP-350 dataset to evaluate it against ground truth.
   - **Ad-Hoc Auditing**: Drag-and-drop a new privacy policy HTML file to perform a live semantic audit.
   - **Risk Badges**: Automatically categorises findings into **HIGH**, **MEDIUM**, and **LOW** severity.

---

## 📈 Results & Visualisations

The system has been fully evaluated against 349 valid HTML policies across three chunking configurations. 

**Key Findings:**
- **Conservative Predictions**: The pipeline exhibits a high False Negative rate relative to False Positives, indicating it rarely hallucinates risks that are not present (crucial for a decision-support tool).
- **Semantic Dominance**: Semantic chunking (`F1: 0.339`) significantly outperformed both fixed size chunks (`fixed_300 F1: 0.166`, `fixed_500 F1: 0.164`) by preserving contextual boundaries.
- **LLM Limitations**: Formatting failures (JSON parse errors) and timeout limits on long context windows are measurable failure modes in local RAG deployments.

*For complete tables and analysis, refer to `results/error_analysis_summary.md` and the charts in `results/figures/`.*

---

## ⚠️ Limitations

A detailed exploration of technical, ethical, and practical limitations is available in [`docs/limitation.md`](docs/limitation.md). Major constraints include:
- Incomplete LLM extraction over long, multi-excerpt context windows.
- Attribution drift when reading repetitive structural text.
- Hardware constraints dictating single-model evaluation (Qwen 2.5 7B) within the project timeframe.

---

**AI-Use Note**: This system was developed with the assistance of Generative AI for architectural scaffolding, debugging, and code structuring, consistent with project proposals. No dissertation prose or experimental results were generated by AI.
#   c o m p l e t e - p r i v a c y - r a g - a u d i t  
 