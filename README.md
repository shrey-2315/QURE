# QURE: Query Understanding via Retrieval Ensembles

*A Modular and Interpretable Text-to-SQL Framework with STaR Reasoning*

---

## 🧠 Overview

**QURE** is a **modular** and **interpretable** framework for **Text-to-SQL generation**, designed to overcome critical limitations of current end-to-end LLM systems in real-world NLIDB (Natural Language Interface to Databases) scenarios.

Instead of directly mapping a question to SQL, QURE introduces an **explicit reasoning stage** that generates **natural language rationales** using a **STaR-trained LLaMA model**. These rationales act as validated blueprints for SQL generation using fine-tuned **SQLCoder** models specialized for different SQL structures (e.g., CTEs, subqueries).

---

## 🔍 Motivation

Despite advances in LLM-based Text-to-SQL synthesis, existing systems face key challenges:

* ❌ Poor semantic alignment between NL queries and database schemas
* ❌ Low generalization to unseen databases
* ❌ Fragile SQL generation for complex structures (CTEs, subqueries)
* ❌ Lack of interpretability in enterprise settings

**QURE** addresses these through:

* ✨ **Modularity** (reasoning + generation separation)
* 🔄 **Hybrid retrieval (BM25 + Dense)** for context and schema grounding
* 📜 **Natural language rationales** (via STaR) for transparency
* 🧹 **Structure-aware generation** with SQLCoder routing
* 🧪 **Execution-free metrics** for robust evaluation




## 🧠 Key Concepts Used

### ✅ Self-Taught Reasoner (STaR)

Iteratively teaches the model to generate better rationales using its own past mistakes and corrected generations.

### 🧠 Natural Language Rationale Generation

Instead of SQL-first generation, QURE explains the **thought process** in plain English before synthesizing SQL.

### 📚 Hybrid Retrieval

Combines **BM25 sparse** search with **dense embeddings** (e.g., from BGE) to fetch both:

* Relevant **schema elements**
* Few-shot **example prompts**

### ⚙️ Structural Routing

Uses validated rationale to classify the instruction into **CTE** or **subquery**, and routes it to the **appropriate SQLCoder model**.



## 📊 Evaluation

QURE is evaluated using **execution-free** and **structure-aware** metrics:

* 🧹 **AST(TE)**: Tree Edit Distance between Abstract Syntax Trees
* 🧠 **ETM**: Enhanced Tree Matching (schema-aware)
* 📏 **CodeBERTScore**: Semantic match of generated SQL
* 🔀 **RelPM**, **ASTPM** (reproduced from Spider/BIRD benchmarks)

---

## 📦 Requirements

* Python ≥ 3.10
* Transformers ≥ 4.39
* `peft`, `bitsandbytes`, `datasets`, `accelerate`, `tqdm`
* GPU with ≥16GB VRAM recommended
* Hugging Face token (set via `HF_TOKEN` in `config.py`)

Install all dependencies:

```bash
pip install -r requirements.txt
```




This project is licensed under the MIT License.
