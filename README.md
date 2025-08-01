# QURE: A Reasoning-Enhanced RAG Pipeline for Text-to-SQL Generation

QURE is a modular **Retrieval-Augmented Generation (RAG)** based framework for converting natural language questions into accurate SQL queries. It features structured reasoning, complexity-aware model routing, and feedback integration.

---

## 🧠 Architecture Overview

The system follows a structured multi-stage pipeline:

### 1. **NLP Question Input**
User inputs a natural language question.

### 2. **QURE RAG Module**
Retrieves:
- Top-K semantically similar solved examples
- Associated database schema

### 3. **Reasoning Module**
Processes:
- Retrieved examples
- Schema
- Input question

Generates:
- Rationale Output
- Query Plan
- Complexity Decision

### 4. **Complexity Decision**
Determines whether the question needs a simple or complex SQL query.

### 5. **Model Routing**
- If **complex**, routes to `SQLCoder-70B-alpha (CTEs)`
- If **simple**, routes to `SQLCoder-7B (Subqueries)`

### 6. **SQL Execution**
The selected model generates SQL, which is then executed on the **target database**.

### 7. **Error Check & Feedback Loop**
- If there's an error in SQL execution, it routes back to the reasoning module.
- If successful, the result is shown to the **user**.
- Optional feedback is collected to improve future performance.

---

## 🔁 Feedback Loop
The pipeline supports continuous improvement by:
- Detecting SQL execution failures
- Looping back to the Reasoning Module for correction
- Collecting user feedback post successful execution

---

## 💡 Components

| Module              | Description                                      |
|---------------------|--------------------------------------------------|
| QURE RAG Module     | Hybrid retriever (dense + BM25) for examples and schema |
| Reasoning Module    | LLaMA-3.1-8B fine-tuned on rationale generation |
| SQLCoder-7B         | Generates subqueries for simple instructions     |
| SQLCoder-70B-alpha  | Handles CTEs and complex SQL structures          |
| Feedback System     | Collects user feedback for iterative learning    |

---

## 📦 Technologies Used

- Python, Transformers (Hugging Face)
- LLaMA-3.1-8B (reasoning)
- SQLCoder models (Fine-tuned 7B & 70B variants)
- Retrieval (BM25 + Dense)
- Flask / FastAPI (optional API backend)
- SQLite / Postgres (as database engine)

---

## 📌 Use Cases

- Complex question-to-SQL translation
- Interactive data querying with reasoning trace
- Multi-step query generation with control


[MIT License](LICENSE)

