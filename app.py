
import streamlit as st
import os
import sqlite3
import pandas as pd
import faiss
import numpy as np
import torch
import datetime
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN", "")
login(token=HF_TOKEN)

st.set_page_config(page_title="Nifty Market Intelligence", page_icon="📈", layout="wide")
st.title("📈 Nifty Market Intelligence Chatbot")
st.caption("Ask questions about Nifty, VIX regimes, sectoral performance, and get next week predictions (2023-2026)")

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text-generation", model="google/gemma-2b-it", device_map="auto", torch_dtype=torch.float16)
    finbert = pipeline("text-classification", model="ProsusAI/finbert", device=0)
    return embedder, generator, finbert

@st.cache_resource
def load_data():
    conn = sqlite3.connect("nifty_intelligence.db")
    weekly = pd.read_sql("SELECT * FROM weekly_nifty", conn)
    yearly_stats = pd.read_sql("""SELECT strftime("%Y", date) as year, MAX(close) as yearly_high, MIN(close) as yearly_low FROM nifty_ohlcv GROUP BY year""", conn)
    yearly_dict = {row["year"]: (row["yearly_high"], row["yearly_low"]) for _, row in yearly_stats.iterrows()}

    # Sentiment
    sentiment_rows = []
    for _, row in weekly.iterrows():
        week = row["week"]
        prev = weekly[weekly["week"] < week]
        weekly_return = 0.0
        if len(prev) > 0:
            prev_close = prev.iloc[-1]["avg_close"]
            weekly_return = round((row["avg_close"] - prev_close) / prev_close * 100, 2)
        vix_row = pd.read_sql(f"""SELECT ROUND(AVG(close), 2) as avg_vix FROM india_vix WHERE strftime("%Y-W%W", date) = "{week}" """, conn)
        avg_vix = vix_row["avg_vix"].values[0] if len(vix_row) > 0 else 15.0
        text = (f"Nifty weekly return was {weekly_return:+.2f}%. India VIX was {avg_vix:.1f}. "
                f"Market {'rose' if weekly_return > 0 else 'fell'} this week with {'low' if avg_vix < 13 else 'high' if avg_vix > 20 else 'moderate'} volatility.")
        sentiment_rows.append({"week": week, "weekly_return": weekly_return, "vix": avg_vix, "text": text})
    sent_texts = [r["text"] for r in sentiment_rows]
    sent_results = finbert(sent_texts, truncation=True, max_length=512)
    for i, r in enumerate(sentiment_rows):
        r["sentiment"] = sent_results[i]["label"]
        r["score"] = round(sent_results[i]["score"], 3)
    sentiment_df = pd.DataFrame(sentiment_rows)

    # Weekly summaries
    summaries = []
    for _, row in weekly.iterrows():
        week = row["week"]
        week_start = row["week_start"]
        date_obj = datetime.datetime.strptime(str(week_start), "%Y-%m-%d")
        month_name = date_obj.strftime("%B")
        year = date_obj.strftime("%Y")
        quarter = f"Q{(date_obj.month - 1) // 3 + 1}"
        vix_rows = pd.read_sql(f"""SELECT close as vix FROM india_vix WHERE strftime("%Y-W%W", date) = "{week}" ORDER BY date""", conn)
        regime = "UNKNOWN"
        vix_trend = "stable"
        if len(vix_rows) > 0:
            avg_vix = vix_rows["vix"].mean()
            regime = "LOW" if avg_vix < 13 else ("HIGH" if avg_vix > 20 else "MEDIUM")
            if len(vix_rows) >= 2:
                vix_trend = "rising" if vix_rows["vix"].iloc[-1] > vix_rows["vix"].iloc[0] else "falling"
        sec_rows = pd.read_sql(f"""SELECT sector, avg_close, weekly_range_pct FROM sector_weekly WHERE week = "{week}" ORDER BY weekly_range_pct DESC""", conn)
        sector_text = ", ".join([f"{r['sector']} (close {r['avg_close']}, range {r['weekly_range_pct']}%)" for _, r in sec_rows.iterrows()])
        top_sector = sec_rows.iloc[0]["sector"] if len(sec_rows) > 0 else "UNKNOWN"
        prev = weekly[weekly["week"] < week]
        weekly_return = 0.0
        direction = "bullish"
        if len(prev) > 0:
            prev_close = prev.iloc[-1]["avg_close"]
            weekly_return = round((row["avg_close"] - prev_close) / prev_close * 100, 2)
            direction = "bullish" if weekly_return > 0 else "bearish"
        hl_tag = ""
        if year in yearly_dict:
            yh, yl = yearly_dict[year]
            if row["avg_close"] >= yh * 0.98: hl_tag = " Nifty was near yearly HIGH."
            elif row["avg_close"] <= yl * 1.02: hl_tag = " Nifty was near yearly LOW."
        sent_row = sentiment_df[sentiment_df["week"] == week]
        sentiment = sent_row["sentiment"].values[0] if len(sent_row) > 0 else "neutral"
        sent_score = sent_row["score"].values[0] if len(sent_row) > 0 else 0.5
        summaries.append({"week": week, "text": (
            f"Week {week} of {month_name} {year} ({quarter} {year}, starting {week_start}): "
            f"Nifty average close was {row['avg_close']}, weekly return was {weekly_return:+.2f}%, "
            f"weekly price range was {row['weekly_range_pct']}%, overall market trend was {direction}. "
            f"VIX regime was {regime} and VIX was {vix_trend}. "
            f"FinBERT sentiment was {sentiment} (confidence {sent_score}). "
            f"Sector details: {sector_text}. Most volatile sector was {top_sector}.{hl_tag}"
        )})

    # Monthly summaries
    monthly_summaries = []
    months_df = pd.read_sql("""SELECT strftime("%Y-%m", date) AS month, MIN(date) AS month_start, ROUND(AVG(close), 2) AS avg_close, ROUND((MAX(close) - MIN(close)) / MIN(close) * 100, 2) AS monthly_range_pct FROM nifty_ohlcv GROUP BY month ORDER BY month""", conn)
    for _, row in months_df.iterrows():
        month = row["month"]
        date_obj = datetime.datetime.strptime(str(row["month_start"]), "%Y-%m-%d")
        month_name = date_obj.strftime("%B")
        year = date_obj.strftime("%Y")
        quarter = f"Q{(date_obj.month - 1) // 3 + 1}"
        vix_row = pd.read_sql(f"""SELECT regime, COUNT(*) as cnt FROM vix_regime WHERE strftime("%Y-%m", date) = "{month}" GROUP BY regime ORDER BY cnt DESC LIMIT 1""", conn)
        regime = vix_row["regime"].values[0] if len(vix_row) > 0 else "UNKNOWN"
        vix_trend_row = pd.read_sql(f"""SELECT MIN(close) as vix_start, MAX(close) as vix_end FROM india_vix WHERE strftime("%Y-%m", date) = "{month}" """, conn)
        vix_trend = "rising" if len(vix_trend_row) > 0 and vix_trend_row["vix_end"].values[0] > vix_trend_row["vix_start"].values[0] else "falling"
        sec_rows = pd.read_sql(f"""SELECT s.sector, ROUND(AVG(s.avg_close), 2) AS avg_close, ROUND(AVG(s.weekly_range_pct), 2) AS avg_range FROM sector_weekly s WHERE s.week IN (SELECT DISTINCT strftime("%Y-W%W", date) FROM nifty_ohlcv WHERE strftime("%Y-%m", date) = "{month}") GROUP BY s.sector ORDER BY avg_range DESC""", conn)
        sector_text = ", ".join([f"{r['sector']} (avg close {r['avg_close']}, avg range {r['avg_range']}%)" for _, r in sec_rows.iterrows()])
        top_sector = sec_rows.iloc[0]["sector"] if len(sec_rows) > 0 else "UNKNOWN"
        month_sent = sentiment_df[sentiment_df["week"].str.startswith(month[:4])]
        pos = len(month_sent[month_sent["sentiment"] == "positive"])
        neg = len(month_sent[month_sent["sentiment"] == "negative"])
        dominant_sent = "positive" if pos > neg else "negative"
        monthly_summaries.append({"month": month, "text": (
            f"Month {month_name} {year} ({quarter} {year}): "
            f"Nifty average close was {row['avg_close']}, monthly price range was {row['monthly_range_pct']}%. "
            f"Dominant VIX regime was {regime}, VIX was {vix_trend}. "
            f"FinBERT dominant sentiment was {dominant_sent} ({pos} positive, {neg} negative weeks). "
            f"Sector details: {sector_text}. Most volatile sector was {top_sector}."
        )})

    all_summaries = summaries + monthly_summaries
    latest = pd.read_sql("""SELECT n.date, n.close as nifty_close, v.close as vix FROM nifty_ohlcv n JOIN india_vix v ON n.date = v.date ORDER BY n.date DESC LIMIT 1""", conn)
    return all_summaries, conn, latest, sentiment_df

with st.spinner("Loading models and data... (may take 2-3 mins)"):
    embedder, generator, finbert = load_models()
    all_summaries, conn, latest, sentiment_df = load_data()
    embeddings = embedder.encode([s["text"] for s in all_summaries])
    main_index = faiss.IndexFlatL2(embeddings.shape[1])
    main_index.add(np.array(embeddings))

def predict_next_week():
    conn = sqlite3.connect("nifty_intelligence.db")
    closes = pd.read_sql("""SELECT close FROM nifty_ohlcv ORDER BY date DESC LIMIT 20""", conn)["close"].values
    recent_return = round((closes[0] - closes[4]) / closes[4] * 100, 2)
    longer_return = round((closes[0] - closes[19]) / closes[19] * 100, 2)
    latest_vix = pd.read_sql("""SELECT close FROM india_vix ORDER BY date DESC LIMIT 1""", conn)["close"].values[0]
    vix_regime = "LOW" if latest_vix < 13 else ("HIGH" if latest_vix > 20 else "MEDIUM")
    if recent_return > 1.5 and longer_return > 3: momentum = "strong bullish"
    elif recent_return > 0 and longer_return > 0: momentum = "mild bullish"
    elif recent_return < -1.5 and longer_return < -3: momentum = "strong bearish"
    elif recent_return < 0 and longer_return < 0: momentum = "mild bearish"
    else: momentum = "mixed"
    latest_sent = sentiment_df.iloc[-1]
    signals = []
    if "bullish" in momentum: signals.append("bullish momentum")
    else: signals.append("bearish momentum")
    if vix_regime == "LOW": signals.append("low fear environment")
    elif vix_regime == "HIGH": signals.append("high fear — caution advised")
    if latest_sent["sentiment"] == "positive": signals.append("positive FinBERT sentiment")
    else: signals.append("negative FinBERT sentiment")
    return (
        f"4-week return is {recent_return:+.2f}%, 20-day return is {longer_return:+.2f}%. "
        f"Momentum is {momentum}. VIX is {latest_vix:.1f} ({vix_regime} regime). "
        f"FinBERT sentiment is {latest_sent['sentiment']} (score {latest_sent['score']}). "
        f"Combined signals: {', '.join(signals)}."
    )

def aggregate_context(retrieved):
    bull_count = sum(1 for r in retrieved if "bullish" in r)
    bear_count = sum(1 for r in retrieved if "bearish" in r)
    trend = "mostly bullish" if bull_count > bear_count else "mostly bearish"
    regimes = [m.group(1) for r in retrieved for m in [re.search(r"VIX regime was (\w+)", r)] if m]
    dominant_regime = max(set(regimes), key=regimes.count) if regimes else "UNKNOWN"
    sector_counts = {"IT": 0, "AUTO": 0, "BANKNIFTY": 0, "FMCG": 0}
    for r in retrieved:
        m = re.search(r"Most (?:frequently )?volatile sector was (\w+)", r)
        if m and m.group(1) in sector_counts:
            sector_counts[m.group(1)] += 1
    top_sector = max(sector_counts, key=sector_counts.get)
    closes = [float(m.group(1)) for r in retrieved for m in [re.search(r"Nifty average close was ([\d.]+)", r)] if m]
    nifty_range = f"{min(closes):.0f} to {max(closes):.0f}" if closes else "N/A"
    dates = [m.group(1) for r in retrieved for m in [re.search(r"starting ([\d-]+)", r)] if m]
    date_range = f"{min(dates)} to {max(dates)}" if dates else "N/A"
    chunks_found = len(retrieved)
    return (
        f"Period covered: {date_range}. Nifty ranged from {nifty_range}. "
        f"Overall trend was {trend} ({bull_count} bullish weeks, {bear_count} bearish weeks). "
        f"Dominant VIX regime was {dominant_regime}. "
        f"Most frequently volatile sector was {top_sector} ({sector_counts[top_sector]} out of {chunks_found} weeks). "
        f"Sector counts: " + ", ".join([f"{k}: {v} weeks" for k,v in sorted(sector_counts.items(), key=lambda x: -x[1])])
    ), chunks_found

def search(query, top_k=8):
    conn = sqlite3.connect("nifty_intelligence.db")
    period_map = {
        "early": ["January","February","March"],
        "mid": ["April","May","June","July","August"],
        "late": ["September","October","November","December"],
        "q1": ["January","February","March"], "q2": ["April","May","June"],
        "q3": ["July","August","September"], "q4": ["October","November","December"],
        "first half": ["January","February","March","April","May","June"],
        "second half": ["July","August","September","October","November","December"],
        "beginning": ["January","February","March"], "end": ["October","November","December"]
    }
    query_lower = query.lower()
    year_match = re.search(r"20\d{2}", query)
    month_match = re.search(r"January|February|March|April|May|June|July|August|September|October|November|December", query, re.IGNORECASE)
    matched_months = [m for kw, months in period_map.items() if kw in query_lower for m in months]
    filtered = all_summaries
    if year_match:
        y = [s for s in filtered if year_match.group() in s["text"]]
        if y: filtered = y
    if month_match:
        mo = [s for s in filtered if month_match.group().capitalize() in s["text"]]
        if mo: filtered = mo
    elif matched_months:
        p = [s for s in filtered if any(m in s["text"] for m in matched_months)]
        if p: filtered = p
    top_k = 15 if len(filtered) > 20 else (10 if len(filtered) > 10 else min(8, len(filtered)))
    emb = embedder.encode([s["text"] for s in filtered])
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb))
    _, indices = idx.search(np.array(embedder.encode([query])), min(top_k, len(filtered)))
    return [filtered[i]["text"] for i in indices[0]]

def direct_answer(question, summary_line):
    q = question.lower()
    period = re.search(r"Period covered: (.+?)\.", summary_line)
    nifty_range = re.search(r"Nifty ranged from (.+?)\.", summary_line)
    trend = re.search(r"Overall trend was (.+?)\(", summary_line)
    vix = re.search(r"Dominant VIX regime was (\w+)", summary_line)
    sector = re.search(r"Most (?:frequently )?volatile sector was (\w+) \((\d+) out of (\d+)", summary_line)
    sector_counts = re.search(r"Sector counts: (.+)$", summary_line)
    period_str = period.group(1) if period else ""
    nifty_str = nifty_range.group(1).strip() if nifty_range else ""
    trend_str = trend.group(1).strip() if trend else ""
    vix_str = vix.group(1) if vix else ""
    sector_str = sector.group(1) if sector else ""
    sector_weeks = f"{sector.group(2)} out of {sector.group(3)} weeks" if sector else ""
    sector_counts_str = sector_counts.group(1) if sector_counts else ""
    if "sector" in q and any(w in q for w in ["vix", "volatile", "high vix"]):
        if sector_str:
            return f"During {period_str} (dominant VIX: {vix_str}), the most volatile sector was {sector_str} ({sector_weeks}). Breakdown: {sector_counts_str}."
    if any(w in q for w in ["which sector", "sector led", "leading sector", "most volatile sector"]):
        if sector_str:
            return f"The most volatile sector during {period_str} was {sector_str} ({sector_weeks}). Breakdown: {sector_counts_str}."
    if any(w in q for w in ["vix regime", "vix", "fear", "regime"]) and "sector" not in q:
        return f"The dominant VIX regime during {period_str} was {vix_str}."
    if any(w in q for w in ["nifty", "index", "perform", "market", "close", "level", "how did"]):
        return f"During {period_str}, Nifty ranged from {nifty_str} with a {trend_str} trend. VIX regime was {vix_str}."
    return None

def ask(question, chat_history):
    q = question.lower()
    if any(w in q for w in ["next week", "predict", "outlook", "forecast", "what to expect", "next month"]):
        prediction = predict_next_week()
        prompt = f"""<start_of_turn>user
You are a financial analyst. Based on the signals below, give a 3-4 sentence market outlook.
Be specific about direction, risk level, and which sectors to watch.

Signals: {prediction}

Question: {question}<end_of_turn>
<start_of_turn>model
Based on current market signals, """
        output = generator(prompt, max_new_tokens=200, do_sample=False)
        answer = output[0]["generated_text"].split("<start_of_turn>model")[-1].strip()
        return answer, "🔮 Prediction", "live signals", 3

    retrieved = search(question)
    summary_line, chunks_found = aggregate_context(retrieved)
    direct = direct_answer(question, summary_line)
    if direct:
        confidence = "🟢 High" if chunks_found >= 8 else ("🟡 Medium" if chunks_found >= 4 else "🔴 Low")
        sources = ", ".join(list(dict.fromkeys([re.search(r"starting (\d{4}-\d{2})", r).group(1) if re.search(r"starting (\d{4}-\d{2})", r) else "" for r in retrieved]))[:3])
        return direct, confidence, sources, chunks_found
    history_text = "".join([f"Q: {t['user']}\nA: {t['assistant']}\n" for t in chat_history[-3:]])
    prompt = f"""<start_of_turn>user
Data: {summary_line}
{f"Previous conversation:{chr(10)}{history_text}" if history_text else ""}
Question: {question}
Give a direct 2-3 sentence answer using only the data.<end_of_turn>
<start_of_turn>model
Based on the data, """
    output = generator(prompt, max_new_tokens=200, do_sample=False)
    answer = output[0]["generated_text"].split("<start_of_turn>model")[-1].strip()
    confidence = "🟢 High" if chunks_found >= 8 else ("🟡 Medium" if chunks_found >= 4 else "🔴 Low")
    sources = ", ".join(list(dict.fromkeys([re.search(r"starting (\d{4}-\d{2})", r).group(1) if re.search(r"starting (\d{4}-\d{2})", r) else "" for r in retrieved]))[:3])
    return answer, confidence, sources, chunks_found

for key, val in [("messages", []), ("chat_history", []), ("pending_question", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.header("📊 Market Snapshot")
    if len(latest) > 0:
        col1, col2 = st.columns(2)
        col1.metric("Nifty", f"{latest['nifty_close'].values[0]:,.0f}")
        vix_val = latest["vix"].values[0]
        vix_regime = "LOW" if vix_val < 13 else ("MEDIUM" if vix_val <= 20 else "HIGH")
        col2.metric("VIX", f"{vix_val:.1f}", vix_regime)
        st.caption(f"As of {latest['date'].values[0]}")
    st.divider()
    st.header("💡 Sample Questions")
    for s in [
        "Which sector led in mid 2025?",
        "How did Nifty perform in early 2024?",
        "Which sectors were most volatile during high VIX?",
        "How did markets perform in second half of 2024?",
        "What was the VIX regime in Q3 2023?",
        "What is the outlook for next week?",
        "Compare sector performance in 2023 vs 2024"
    ]:
        if st.button(s, use_container_width=True):
            st.session_state.pending_question = s
    st.divider()
    st.header("🧠 Conversation Memory")
    if st.session_state.chat_history:
        for i, turn in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {turn['user'][:35]}..."):
                st.write(f"**Q:** {turn['user']}")
                st.write(f"**A:** {turn['assistant']}")
        if st.button("🗑️ Clear Memory", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()
    else:
        st.caption("No conversation history yet.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg:
            st.caption(msg["meta"])

typed = st.chat_input("Ask about Nifty, VIX, sectors, or get next week prediction...")
question = st.session_state.pending_question if st.session_state.pending_question else typed

if question:
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, confidence, sources, chunks = ask(question, st.session_state.chat_history)
        st.markdown(answer)
        meta = f"Confidence: {confidence} | Sources: {sources} | Chunks: {chunks}"
        st.caption(meta)
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
    st.session_state.chat_history.append({"user": question, "assistant": answer})
