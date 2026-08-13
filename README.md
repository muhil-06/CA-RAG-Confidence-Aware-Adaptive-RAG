# CA-RAG: Confidence-Aware Adaptive Retrieval-Augmented Generation

## The idea

Standard RAG pipelines retrieve documents on *every single query*, even when
the LLM already knows the answer (e.g. "What is the capital of France?").
This wastes embedding calls, vector search time, and prompt tokens.

**CA-RAG adds a confidence gate in front of retrieval.** The LLM first
attempts a direct answer and self-rates its confidence. Only when confidence
is low — or the question clearly needs document-specific facts — does the
system retrieve. This cuts unnecessary retrieval calls while keeping
document-grounded answers accurate.

## Architecture

```
                     User Query
                         │
                         ▼
        ┌────────────────────────────────┐
        │ 1. Direct Answer Attempt        │
        │    LLM answers + self-rates     │
        │    confidence (0-100) and       │
        │    flags "needs external info"  │
        └────────────────┬────────────────┘
                          │
                          ▼
        ┌────────────────────────────────┐
        │ 2. Confidence Gate               │
        │    confidence >= threshold AND   │
        │    no external info needed?      │
        └───────┬───────────────┬─────────┘
                 │ YES           │ NO
                 ▼               ▼
        ┌────────────────┐  ┌───────────────────────────┐
        │ Return direct   │  │ 3. Retrieve top-k chunks   │
        │ answer           │  │    from vector store        │
        │ (retrieval        │  └──────────────┬──────────────┘
        │  SKIPPED)          │                 ▼
        └────────────────┘  ┌───────────────────────────┐
                              │ 4. Filter chunks below     │
                              │    similarity threshold      │
                              └──────────────┬──────────────┘
                                              ▼
                              ┌───────────────────────────┐
                              │ 5. Generate grounded answer │
                              │    with citations             │
                              └───────────────────────────┘
```

Every run is logged to `logs/run_log.jsonl` with confidence score, whether
retrieval was triggered, latency, and estimated cost — so the savings are
measurable, not just claimed.

## Project structure

```
ca_rag/
├── data/documents/          # sample knowledge base (leave policy, FAQ, manual)
├── src/
│   ├── config.py              # all settings, env-driven
│   ├── llm_client.py           # Anthropic API or local Ollama backend
│   ├── embeddings.py            # sentence-transformers wrapper
│   ├── vector_store.py           # ChromaDB wrapper
│   ├── confidence_gate.py         # the core novelty: retrieval decision logic
│   ├── retriever.py                # retrieval + relevance filtering
│   └── pipeline.py                  # orchestrates the full flow
├── evaluation/
│   ├── benchmark_questions.json      # 25 questions: general + document-specific
│   ├── metrics.py                      # accuracy / cost / latency scoring
│   └── run_evaluation.py                # CA-RAG vs baseline comparison
├── app.py                     # Streamlit chat demo
├── requirements.txt
└── .env.example
```

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   - For the cheapest cloud option, set `MODE=api` and add your
     `ANTHROPIC_API_KEY` (Claude Haiku is inexpensive and fast).
   - For a **fully free, local** option, set `MODE=local`, install
     [Ollama](https://ollama.com), run `ollama pull llama3.2`, and start
     it with `ollama serve`.

3. **Run the demo app**
   ```bash
   streamlit run app.py
   ```
   The first run automatically indexes the sample documents in
   `data/documents/` into a local ChromaDB store.

4. **Run the evaluation benchmark**
   ```bash
   python -m evaluation.run_evaluation
   ```
   This runs all 25 benchmark questions through both CA-RAG and a baseline
   "always retrieve" pipeline, then prints a comparison table and saves
   `evaluation/results_chart.png`.

## How to demo it

Ask general-knowledge questions first (e.g. *"What is the capital of Japan?"*)
— you'll see the "⚡ Answered directly (no retrieval)" badge. Then ask a
document-specific question (e.g. *"How many days of sick leave do I get?"*)
— you'll see "📄 Retrieved N document chunk(s)" with a grounded, cited answer.

## Results

*(Paste your benchmark output here after running `run_evaluation.py`.)*

| Metric | CA-RAG | Baseline (always-retrieve) |
|---|---|---|
| Retrieval skip rate | — | 0% |
| Accuracy | — | — |
| Avg latency (ms) | — | — |
| Total estimated cost (USD) | — | — |

## Tuning

- `CONFIDENCE_THRESHOLD` (default 70) — lower this to retrieve more
  aggressively (safer, less savings); raise it to skip retrieval more
  often (cheaper, riskier).
- `SIMILARITY_THRESHOLD` (default 0.35) — controls how strict the
  relevance filter is on retrieved chunks.
- `TOP_K` (default 5) — how many chunks to retrieve before filtering.

## Notes

- Cost estimates in this project use placeholder per-token rates for
  *relative* comparison between CA-RAG and the baseline — they are not
  exact billing figures. Check current Anthropic pricing at
  https://docs.claude.com for accurate rates.
- The relevance filter uses plain cosine similarity — no extra model or
  training required, keeping the whole system free/low-cost to run.
