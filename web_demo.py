import os
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# 基础配置
st.set_page_config(page_title="AI News Researcher", layout="wide", page_icon="🧠")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 1. 数据加载逻辑
@st.cache_data
def load_data():
    file_name = 'bbc_news.csv' 
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df.columns = [c.lower() for c in df.columns]
        if 'pubdate' in df.columns:
            df['pubdate'] = pd.to_datetime(df['pubdate'], errors='coerce')
        
        # 分类与情感逻辑
        def analyze_sentiment(text):
            text = str(text).lower()
            if any(w in text for w in ['win', 'success', 'growth', 'rise']): return "Positive"
            if any(w in text for w in ['loss', 'fail', 'crisis', 'drop']): return "Negative"
            return "Neutral"
        df['sentiment'] = df['description'].apply(analyze_sentiment)
        return df
    return pd.DataFrame()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

df = load_data()
model = load_model()

# --- 2. Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/48/BBC_News_2019.svg", width=120)
    st.title("📊 Analysis Insights")
    if not df.empty:
        st.metric("Total Articles", f"{len(df):,}")
        st.write("📈 **Sentiment Distribution**")
        st.bar_chart(df['sentiment'].value_counts())
    st.divider()
    num_results = st.slider("Results to Analyze", 1, 10, 3)
    search_depth = st.select_slider("Search Depth", options=[100, 200, 300, 500], value=300)

# --- 3. Main Area ---
st.title("🧠 AI News Researcher")
st.markdown("Transforming BBC News into **Active Intelligence**.")

if "search_word" not in st.session_state:
    st.session_state.search_word = ""

st.write("🔥 **Research Hotspots:**")
hot_keywords = ["Technology", "Economy", "Climate", "Politics", "Health"]
cols = st.columns(len(hot_keywords))

for i, word in enumerate(hot_keywords):
    if cols[i].button(word, key=f"btn_{word}"):
        st.session_state.search_word = word
        st.rerun()

def sync_input():
    st.session_state.search_word = st.session_state.user_input_key

query = st.text_input("🔍 Search topic:", value=st.session_state.search_word, key="user_input_key", on_change=sync_input)
query = st.session_state.search_word

# --- 4. Retrieval & Analysis ---
if query and not df.empty:
    with st.status("AI Analyzing...", expanded=False):
        query_vec = model.encode([query])
        sub_df = df.iloc[:search_depth].copy()
        content_vecs = model.encode(sub_df['description'].fillna("").tolist())
        scores = np.dot(content_vecs, query_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:num_results]
        top_results = sub_df.iloc[top_indices].copy()

    with st.chat_message("assistant"):
        st.write(f"### 📝 Executive Summary: {query}")
        sentiment_mode = top_results['sentiment'].mode()[0]
        st.write(f"Key finding: The top reports focus on **{top_results.iloc[0]['title']}**. Overall mood is **{sentiment_mode}**.")

    st.write("📅 **Reporting Trend**")
    if 'pubdate' in top_results.columns:
        trend_df = top_results.set_index('pubdate').resample('D').size()
        st.line_chart(trend_df)

    for idx in top_indices:
        row = sub_df.iloc[idx]
        with st.container(border=True):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                display_title = re.sub(f"({query})", r"<mark style='background:#FFD700'>\1</mark>", row['title'], flags=re.IGNORECASE)
                st.markdown(f"#### {display_title}", unsafe_allow_html=True)
                st.write(row['description'])
                st.caption(f"Sentiment: {row['sentiment']}")
            with col_b:
                st.metric("Match", f"{scores[idx]*100:.1f}%")
                st.progress(float(scores[idx]) if scores[idx] > 0 else 0.0)
else:
    st.info("Input a topic to start.")