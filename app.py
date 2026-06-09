import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import sqlite3
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ------------------ Page Config ------------------ #
st.set_page_config(
    page_title="YouTube Comment Sentiment Analyzer",
    layout="wide"
)

# ------------------ Sidebar ------------------ #
st.sidebar.markdown("## Dashboard")
st.sidebar.button("Home")
st.sidebar.button("History")
st.sidebar.button("About")

# ------------------ Title ------------------ #
st.markdown("<h1 style='color: #6A0DAD;'>Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #444;'>Analyze the sentiment of your text instantly</p>", unsafe_allow_html=True)

# ------------------ Database ------------------ #
conn = sqlite3.connect("history.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    sentiment TEXT,
    aspect TEXT,
    timestamp TEXT
)
''')
conn.commit()

# ------------------ Load mBERT ------------------ #
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()
labels = ["negative","neutral","positive"]

# ------------------ Aspect Detection ------------------ #
aspect_keywords = {
    "Taste": ["rasa","manis","masin","pedas","asam","gurih","lembut"],
    "General": ["bagus","best","ok","menarik","mudah","simple"],
    "Cooking Steps": ["rebus","goreng","bakar","panaskan","campur","masukkan","adun"],
    "Ingredients": ["bahan","tepung","gula","garam","telur","minyak","santan"]
}

def detect_aspect(text):
    text_lower = text.lower()
    for aspect, keywords in aspect_keywords.items():
        if any(k in text_lower for k in keywords):
            return aspect
    return "General"

# ------------------ Functions ------------------ #
def predict_sentiment(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            pred_label = labels[torch.argmax(probs)]
        results.append(pred_label)
    return results

def save_history(texts, sentiments, aspects):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for text, sentiment, aspect in zip(texts, sentiments, aspects):
        c.execute('INSERT INTO history (text,sentiment,aspect,timestamp) VALUES (?,?,?,?)',
                  (text, sentiment, aspect, timestamp))
    conn.commit()

def get_history():
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    if 'aspect' not in df.columns:
        df['aspect'] = 'General'
    return df

def plot_sentiment_bar(df):
    plt.figure(figsize=(6,3))
    sns.countplot(x='sentiment', data=df, palette=['#FFB3BA','#BAFFC9','#BAE1FF'])
    plt.title("Sentiment Distribution")
    st.pyplot(plt)

def plot_wordcloud(df):
    text = " ".join(df["text"].tolist())
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# ------------------ Main Panel ------------------ #
col1, col2 = st.columns([2,1])

with col1:
    st.markdown("### Enter YouTube comment or text")
    input_text = st.text_area("Enter text here:")
    if st.button("Analyze"):
        if input_text.strip() != "":
            sentiment = predict_sentiment([input_text])[0]
            aspect = detect_aspect(input_text)
            save_history([input_text],[sentiment],[aspect])
            
            st.markdown(f"**Sentiment:** {sentiment.capitalize()}")
            st.markdown(f"**Aspect:** {aspect}")
            
            df_display = pd.DataFrame({"text":[input_text],"sentiment":[sentiment],"aspect":[aspect]})
            plot_sentiment_bar(df_display)
            plot_wordcloud(df_display)
        else:
            st.warning("Please enter a comment or URL!")

with col2:
    st.markdown("### Latest Result")
    history_df = get_history()
    if not history_df.empty:
        latest = history_df.iloc[0]
        st.markdown(f"**Text:** {latest['text']}")
        st.markdown(f"**Sentiment:** {latest['sentiment'].capitalize()}")
        st.markdown(f"**Aspect:** {latest['aspect']}")
        st.markdown(f"**Timestamp:** {latest['timestamp']}")

st.markdown("### Recent History")
history_table = get_history()
st.dataframe(history_table)
