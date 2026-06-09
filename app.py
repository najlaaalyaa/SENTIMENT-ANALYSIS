# ------------------ Full Streamlit App: Malay Sentiment Analyzer ------------------ #

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

# ------------------ Page & Style ------------------ #
st.set_page_config(page_title="Malay Sentiment Analyzer", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #FFF5F8; }
.stButton>button { background-color: #A7C7E7; color: black; }
</style>
""", unsafe_allow_html=True)

st.title("Malay Sentiment Analyzer")
st.write("Analyze Malay YouTube comments with mBERT, store history, and view visualizations.")

# ------------------ Database ------------------ #
conn = sqlite3.connect("comment_history.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment TEXT,
    aspect TEXT,
    sentiment TEXT,
    timestamp TEXT
)
''')
conn.commit()

# ------------------ Load mBERT ------------------ #
@st.cache_resource
def load_model():
    model_name = "ipankamal/mb-bert-sentiment-malay"  # replace with your fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()
labels = ['negative', 'neutral', 'positive']

# ------------------ Prediction ------------------ #
def predict_sentiment(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            pred_label = labels[torch.argmax(probs)]
        results.append(pred_label)
    return results

# ------------------ Database Helpers ------------------ #
def save_to_db(comments, sentiments, aspects):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for comment, sentiment, aspect in zip(comments, sentiments, aspects):
        c.execute('INSERT INTO history (comment, aspect, sentiment, timestamp) VALUES (?,?,?,?)',
                  (comment, aspect, sentiment, timestamp))
    conn.commit()

def get_history():
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    return df

# ------------------ Visualizations ------------------ #
def plot_sentiment_distribution(df, title='Sentiment Distribution'):
    plt.figure(figsize=(6,4))
    sns.countplot(x='sentiment', data=df, palette=['#FFB3BA','#BAE1FF','#BAFFC9'])
    plt.title(title)
    st.pyplot(plt)

def generate_wordcloud(df, title='WordCloud'):
    text = ' '.join(df['comment'].astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

# ------------------ Streamlit Tabs ------------------ #
tab1, tab2, tab3 = st.tabs(["Text Input", "Upload CSV", "History"])

# ---------- Tab 1: Text Input ----------
with tab1:
    user_text = st.text_area("Enter your Malay comment here:")
    user_aspect = st.selectbox("Aspect", ["general","ingredients","steps","taste","cooking","challenge","education","mukbang","QnA","vlog"])
    
    if st.button("Analyze Text"):
        if user_text.strip() != "":
            sentiment = predict_sentiment([user_text])
            save_to_db([user_text], sentiment, [user_aspect])
            df_result = pd.DataFrame({'comment':[user_text],'aspect':[user_aspect],'sentiment':sentiment})
            
            st.subheader("Analysis Result")
            st.dataframe(df_result)
            plot_sentiment_distribution(df_result, f'Sentiment - {user_aspect}')
            generate_wordcloud(df_result, f'WordCloud - {user_aspect}')
        else:
            st.error("Please enter a comment!")

# ---------- Tab 2: CSV Upload ----------
with tab2:
    uploaded_file = st.file_uploader("Upload CSV (Columns: comment, optional aspect)", type=["csv"])
    if uploaded_file:
        df_csv = pd.read_csv(uploaded_file)
        if df_csv.shape[1] == 1:
            df_csv.columns = ['comment']
            df_csv['aspect'] = 'general'
        else:
            df_csv.columns = ['comment','aspect']
        
        sentiments = predict_sentiment(df_csv['comment'].astype(str).tolist())
        save_to_db(df_csv['comment'].tolist(), sentiments, df_csv['aspect'].tolist())
        df_csv['sentiment'] = sentiments
        
        st.subheader("Analysis Result")
        st.dataframe(df_csv)
        for aspect in df_csv['aspect'].unique():
            df_aspect = df_csv[df_csv['aspect']==aspect]
            st.write(f"**Aspect:** {aspect}")
            plot_sentiment_distribution(df_aspect, f'Sentiment - {aspect}')
            generate_wordcloud(df_aspect, f'WordCloud - {aspect}')

# ---------- Tab 3: History ----------
with tab3:
    df_history = get_history()
    if not df_history.empty:
        st.subheader("History of Comments")
        st.dataframe(df_history)
        for aspect in df_history['aspect'].unique():
            df_aspect = df_history[df_history['aspect']==aspect]
            st.write(f"**Aspect:** {aspect}")
            plot_sentiment_distribution(df_aspect, f'Sentiment - {aspect}')
            generate_wordcloud(df_aspect, f'WordCloud - {aspect}')
    else:
        st.info("No history yet.")
