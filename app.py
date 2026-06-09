import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ------------------ Visualization Functions ------------------ #
def plot_sentiment_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='sentimen', data=df, palette=['red','gray','green'])
    plt.title('Taburan Sentimen')
    st.pyplot(plt)

def generate_wordcloud(df):
    text = ' '.join(df['komen'].astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# ------------------ Streamlit GUI ------------------ #
st.title("Sistem Analisis Sentimen Bahasa Melayu (Front-End)")
st.write("Masukkan teks atau muat naik CSV untuk analisis sentimen.")

tab1, tab2 = st.tabs(["Input Teks", "Muat Naik CSV"])

# ---------- Tab 1: Text Input ----------
with tab1:
    user_text = st.text_area("Masukkan komen Bahasa Melayu di sini:")
    user_sentiment = st.selectbox("Pilih Sentimen (untuk contoh/testing)", ["positive","negative","neutral"])
    
    if st.button("Tunjuk Hasil Teks"):
        if user_text.strip() == "":
            st.error("Sila masukkan teks terlebih dahulu!")
        else:
            df_text = pd.DataFrame({'komen':[user_text], 'sentimen':[user_sentiment]})
            st.subheader("Keputusan Analisis:")
            st.dataframe(df_text)
            plot_sentiment_distribution(df_text)
            st.subheader("WordCloud:")
            generate_wordcloud(df_text)

# ---------- Tab 2: CSV Upload ----------
with tab2:
    uploaded_file = st.file_uploader("Muat naik fail CSV (satu lajur teks + optional satu lajur sentimen):", type=["csv"])
    if uploaded_file:
        try:
            df_csv = pd.read_csv(uploaded_file)
            if df_csv.shape[1] == 1:
                df_csv.columns = ['komen']
                df_csv['sentimen'] = 'neutral'  # placeholder
            else:
                df_csv.columns = ['komen','sentimen']
            
            st.subheader("Keputusan Analisis:")
            st.dataframe(df_csv)
            plot_sentiment_distribution(df_csv)
            st.subheader("WordCloud:")
            generate_wordcloud(df_csv)
        except Exception as e:
            st.error(f"Ralat semasa membaca CSV: {e}")
