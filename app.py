# app.py

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ----------------- Page Config ----------------- #
st.set_page_config(page_title="Malay Sentiment Analyzer", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #FFF5F8; }
.stButton>button { background-color: #A07DFE; color: white; font-weight: bold; }
.stSidebar { background-color: #FDE2FF; }
</style>
""", unsafe_allow_html=True)

# ----------------- Sidebar ----------------- #
st.sidebar.markdown("## Navigation")
st.sidebar.button("Home")
st.sidebar.button("History")
st.sidebar.button("About")

# ----------------- SQLite Database ----------------- #
conn = sqlite3.connect("history.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    aspect TEXT,
    sentiment TEXT,
    confidence REAL,
    timestamp TEXT
)
''')
conn.commit()

# ----------------- Load mBERT ----------------- #
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()
labels = ["negative", "neutral", "positive"]

# ----------------- Aspect Detection ----------------- #
aspect_keywords = {
    "Taste": ["rasa", "manis", "masin", "pedas", "asam", "lembut", "gurih"],
    "General": ["bagus", "best", "ok", "menarik", "simple", "mudah"],
    "Cooking Steps": ["rebus", "goreng", "bakar", "panaskan", "campur", "masukkan", "adun"],
    "Ingredients": ["bahan", "tepung", "gula", "garam", "telur", "minyak", "santan"]
}

def detect_aspect(text):
    text_lower = text.lower()
    for aspect, keywords in aspect_keywords.items():
        if any(k in text_lower for k in keywords):
            return aspect
    return "General"

# ----------------- Prediction ----------------- #
def predict_sentiment(texts):
    results, confidences = [], []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            pred_label = labels[torch.argmax(probs)]
            confidence = float(torch.max(probs))
        results.append(pred_label)
        confidences.append(confidence)
    return results, confidences

# ----------------- History Management ----------------- #
def save_history(texts, aspects, sentiments, confidences):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for text, aspect, sentiment, conf in zip(texts, aspects, sentiments, confidences):
        c.execute('INSERT INTO history (text, aspect, sentiment, confidence, timestamp) VALUES (?,?,?,?,?)',
                  (text, aspect, sentiment, conf, timestamp))
    conn.commit()

def get_history():
    return pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)

# ----------------- Visualization ----------------- #
def plot_sentiment_distribution(df, title="Sentiment Distribution"):
    plt.figure(figsize=(6,3))
    sns.countplot(x='sentiment', data=df, palette=['#FFB3BA','#BAE1FF','#BAFFC9'])
    plt.title(title)
    st.pyplot(plt)

def generate_wordcloud(df, title="WordCloud"):
    text = " ".join(df["text"].tolist())
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    st.pyplot(plt)

# ----------------- Streamlit Tabs ----------------- #
tab1, tab2, tab3 = st.tabs(["Text Input", "CSV Upload", "History"])

# ---------- Tab 1: Text Input ----------
with tab1:
    input_text = st.text_area("Paste your comment or text here:")
    if st.button("Analyze Text"):
        if input_text.strip() != "":
            aspect = detect_aspect(input_text)
            sentiments, confidences = predict_sentiment([input_text])
            save_history([input_text], [aspect], sentiments, confidences)

            df_result = pd.DataFrame({
                "text":[input_text],
                "aspect":[aspect],
                "sentiment":sentiments,
                "confidence":confidences
            })
            st.subheader("Analysis Result")
            st.write(f"**Text:** {input_text}")
            st.write(f"**Aspect:** {aspect}")
            st.write(f"**Sentiment:** {sentiments[0].capitalize()}")
            st.write(f"**Confidence:** {confidences[0]:.2f}")
            plot_sentiment_distribution(df_result, f"Sentiment - {aspect}")
            generate_wordcloud(df_result, f"WordCloud - {aspect}")
        else:
            st.warning("Please paste some text!")

# ---------- Tab 2: CSV Upload ----------
with tab2:
    uploaded_file = st.file_uploader("Upload CSV (columns: text, optional aspect):", type=["csv"])
    if uploaded_file:
        df_csv = pd.read_csv(uploaded_file)
        if df_csv.shape[1] == 1:
            df_csv.columns = ["text"]
            df_csv["aspect"] = df_csv["text"].apply(detect_aspect)
        else:
            df_csv.columns = ["text","aspect"]
        sentiments, confidences = predict_sentiment(df_csv["text"].tolist())
        save_history(df_csv["text"].tolist(), df_csv["aspect"].tolist(), sentiments, confidences)
        df_csv["sentiment"] = sentiments
        df_csv["confidence"] = confidences
        st.subheader("CSV Analysis Results")
        st.dataframe(df_csv)
        for aspect in df_csv["aspect"].unique():
            df_aspect = df_csv[df_csv["aspect"]==aspect]
            st.write(f"**Aspect:** {aspect}")
            plot_sentiment_distribution(df_aspect, f"Sentiment - {aspect}")
            generate_wordcloud(df_aspect, f"WordCloud - {aspect}")

# ---------- Tab 3: History ----------
with tab3:
    df_history = get_history()
    if not df_history.empty:
        st.subheader("Recent History")
        st.dataframe(df_history)
        for aspect in df_history["aspect"].unique():
            df_aspect = df_history[df_history["aspect"]==aspect]
            st.write(f"**Aspect:** {aspect}")
            plot_sentiment_distribution(df_aspect, f"Sentiment - {aspect}")
            generate_wordcloud(df_aspect, f"WordCloud - {aspect}")
    else:
        st.info("No history available yet.")
