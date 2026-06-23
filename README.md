# 📈 Nifty Market Intelligence Chatbot

A RAG-based conversational AI system for querying Indian equity market data (2023–2026). Built as part of a finance + data science portfolio targeting Analyst and Portfolio Manager roles.

---

## 🧠 Architecture

```
GOOGLEFINANCE → CSV → SQLite → Weekly/Monthly Summaries → FAISS Embeddings → Gemma 2B → Streamlit
                                          ↕
                                 FinBERT Sentiment Layer
```

---

## 🚀 Features

- **Historical queries** — "How did Nifty perform in January 2024?"
- **VIX regime analysis** — "What was the VIX regime in Q3 2023?"
- **Sectoral analysis** — "Which sector led in mid 2025?"
- **FinBERT sentiment** — Weekly market sentiment scored via ProsusAI/finbert
- **Prediction mode** — "What is the outlook for next week?" using momentum + VIX + sentiment signals
- **Conversation memory** — Follow-up questions with context retention
- **Market snapshot** — Latest Nifty close and VIX displayed in sidebar
- **Source citations** — Confidence score and chunk count shown per answer

---

## 🗃️ Data Sources

| Data | Source | Frequency |
|---|---|---|
| Nifty 50 OHLCV | GOOGLEFINANCE | Daily |
| India VIX | GOOGLEFINANCE | Daily |
| Bank Nifty | GOOGLEFINANCE | Daily |
| Nifty IT | GOOGLEFINANCE | Daily |
| Nifty Auto | GOOGLEFINANCE | Daily |
| Nifty FMCG | GOOGLEFINANCE | Daily |

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Data collection | Google Sheets (GOOGLEFINANCE) |
| Storage | SQLite via sqlite3 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS (Facebook AI Similarity Search) |
| Sentiment | FinBERT (ProsusAI/finbert) |
| LLM | Gemma 2B Instruct (runs locally on GPU) |
| UI | Streamlit |
| Infra | Google Colab T4 GPU + ngrok |

---

## ⚙️ How to Run

1. Clone the repo
```bash
git clone https://github.com/your-username/nifty-market-intelligence-chatbot
```

2. Install dependencies
```bash
pip install streamlit sentence-transformers faiss-cpu transformers torch pyngrok
```

3. Set HuggingFace token
```bash
export HF_TOKEN=your_token_here
```

4. Place `nifty_intelligence.db` in the same folder as `app.py`

5. Run the app
```bash
streamlit run app.py
```

---

## 📊 RAG Design Decisions

This project uses **Naive RAG** — deliberately chosen because the knowledge base is structured, uniform, and small (~225 chunks). Adding retrieval complexity would be over-engineering without measurable accuracy gain.

Key decisions:

**Keyword pre-filtering before vector search**
Prevents year/period confusion — "January 2024" vs "January 2026" would confuse pure semantic search. We filter by year and month first, then run FAISS on the filtered subset.

**Python aggregator before LLM**
Gemma 2B struggles to synthesize across 10+ chunks. We aggregate retrieved chunks into a single structured summary line before passing to the model — reduces hallucination significantly.

**Direct answer engine**
For factual queries (VIX regime, sector led, Nifty range), we bypass the LLM entirely and use regex extraction on the aggregated summary. Faster and more accurate than LLM generation for structured lookups.

**Dual granularity summaries**
183 weekly summaries + 42 monthly summaries = 225 total chunks. Monthly summaries handle broad queries ("How was 2024?"), weekly summaries handle specific queries ("What happened in January 2024?").

**FinBERT integration**
Each weekly summary includes a FinBERT sentiment label derived from price return and VIX. Monthly summaries aggregate weekly sentiment counts.

---

## 🗄️ Database Schema

```sql
-- Raw tables
nifty_ohlcv      -- date, open, high, low, close, volume
india_vix        -- date, close
sectoral         -- date, open, high, low, close, sector
weekly_sentiment -- week, sentiment, score, weekly_return, vix

-- Views
weekly_nifty   -- week, week_start, avg_close, weekly_range_pct
vix_regime     -- date, vix, regime (LOW/MEDIUM/HIGH)
sector_weekly  -- week, sector, avg_close, weekly_range_pct
```

---

## 💬 Sample Questions

- "What was the VIX regime in Q3 2023?"
- "How did Nifty perform in January 2024?"
- "Which sector led in mid 2025?"
- "How did markets perform in second half of 2024?"
- "Which sectors were most volatile during high VIX periods?"
- "What is the outlook for next week?"
- "Compare sector performance in 2023 vs 2024"

---

## 🎯 Target Roles

Analyst | Risk Manager | Portfolio Manager
JPMorgan | BlackRock | Indiabulls Securities
