import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from model import load_model, analyse_comment

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SentiMalay — YouTube Comment Analyser",
    page_icon="🎬",
    layout="wide",
)

# ── CSS — Green & Cream theme, light mode forced ───────────────
st.markdown("""
<style>
    /* Force light cream background everywhere */
    .stApp {
        background-color: #FAFAF5 !important;
        color: #1a1a18 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F0F4E8 !important;
    }
    [data-testid="stSidebar"] * {
        color: #1a1a18 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #1a1a18 !important;
        font-size: 14px !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] span {
        color: #5a5a5a !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #1B4D1B !important;
    }

    /* Main content text */
    .stMarkdown, .stMarkdown p, .stMarkdown li, label, .stText {
        color: #1a1a18 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a18 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #2E7D32 !important;
    }

    /* Text area & inputs — cream background */
    textarea, input, [data-baseweb="textarea"] textarea {
        background-color: #ffffff !important;
        color: #1a1a18 !important;
        border: 1px solid #cdd8c0 !important;
    }

    /* Buttons */
    .stButton > button {
        background: #2E7D32 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background: #225C25 !important;
        color: #ffffff !important;
    }

    /* Code blocks */
    .stCode, code, pre {
        background-color: #F0F4E8 !important;
        color: #1B4D1B !important;
    }

    /* Dataframe — Streamlit renders this as a canvas (glide-data-grid), so we
       theme the wrapper/toolbar chrome; cell text colour is controlled via
       the dataframe theme passed from Python, not CSS. */
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 1px solid #d8e0cc !important;
        border-radius: 8px !important;
    }
    [data-testid="stElementToolbar"] {
        background-color: #ffffff !important;
    }
    [data-testid="stElementToolbarButton"] svg {
        fill: #1a1a18 !important;
    }

    /* Caption */
    .stCaption, .stCaption p {
        color: #666 !important;
    }

    /* Download / generic buttons rendered as <a> or <button> with kind="secondary" */
    .stDownloadButton > button,
    [data-testid="stBaseButton-secondary"] {
        background: #2E7D32 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stDownloadButton > button:hover,
    [data-testid="stBaseButton-secondary"]:hover {
        background: #225C25 !important;
        color: #ffffff !important;
    }
    .stDownloadButton > button *,
    [data-testid="stBaseButton-secondary"] * {
        color: #ffffff !important;
    }

    /* Alert boxes (info / warning / success / error) — force readable text */
    [data-testid="stAlert"] {
        background-color: #FFF6D9 !important;
    }
    [data-testid="stAlert"] * {
        color: #5a4600 !important;
        background-color: transparent !important;
    }
    [data-testid="stAlertContentInfo"] *,
    [data-testid="stAlertContentSuccess"] *,
    [data-testid="stAlertContentWarning"] *,
    [data-testid="stAlertContentError"] * {
        color: inherit !important;
    }

    /* Plotly chart axis labels & tick text */
    .js-plotly-plot .xtick text,
    .js-plotly-plot .ytick text,
    .js-plotly-plot .xtitle,
    .js-plotly-plot .ytitle {
        fill: #1a1a18 !important;
    }

    /* Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border: 1px solid rgba(46,125,50,0.15);
        text-align: center;
    }
    .metric-num   { font-size: 2rem; font-weight: 700; color: #1a1a18; }
    .metric-label { font-size: 12px; color: #666; margin-top: 2px; }

    /* Sentiment badges */
    .badge-Positive { background:#E1EFD9; color:#1B4D1B; padding:3px 12px; border-radius:99px; font-size:12px; font-weight:600; }
    .badge-Neutral  { background:#FAEEDA; color:#633806; padding:3px 12px; border-radius:99px; font-size:12px; font-weight:600; }
    .badge-Negative { background:#FCEBEB; color:#791F1F; padding:3px 12px; border-radius:99px; font-size:12px; font-weight:600; }

    /* Aspect tags */
    .aspect-tag {
        display: inline-block;
        background: #E1EFD9;
        color: #2E7D32;
        padding: 3px 10px;
        border-radius: 99px;
        font-size: 11px;
        margin: 2px;
        font-weight: 500;
    }

    /* Result history boxes */
    .result-box {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid rgba(46,125,50,0.12);
        margin-bottom: 0.75rem;
        color: #1a1a18;
    }
    .result-box * { color: #1a1a18; }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #2E7D32 !important;
    }

    /* Info / warning / success boxes */
    .stAlert { border-radius: 10px !important; }

    /* Select box */
    [data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #1a1a18 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Model and analysis functions are imported from model.py ───────────

# ── Demo comment helper ────────────────────────────────────────
DEMO_COMMENT = (
    "Resepi ni sedap sangat! Bahan-bahannya mudah didapati. "
    "Tapi cara masak dia agak susah sikit untuk orang baru."
)


def set_demo_comment():
    """Place the demo sentence inside the comment text area."""
    st.session_state.comment_input = DEMO_COMMENT


# ── Session state ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 SentiMalay")
    st.caption("Sentiment Analysis of Malay YouTube Comments")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Analyse Comment", "📂 Batch Analysis", "📊 Dashboard", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small>Powered by custom fine-tuned mBERT<br>(Hugging Face: nvjlaa/mBERT — joint aspect-sentiment model)"
        "<br><br>UiTM Final Year Project 2026"
        "<br>Nur Najlaa' Alyaa' Binti Roslan</small>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — ANALYSE COMMENT
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Analyse Comment":
    st.markdown("### 🏠 Analyse YouTube Comment")
    st.caption("Enter a Malay or mixed Malay YouTube comment for sentiment and aspect analysis.")
    st.markdown("")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("**Enter YouTube comment:**")
        comment = st.text_area(
            label="comment",
            label_visibility="collapsed",
            placeholder='e.g. "Resepi ni sedap sangat! Tapi masa memasak agak lama sikit."',
            height=150,
            key="comment_input",
        )

        st.button(
            "🎲 Try demo comment",
            on_click=set_demo_comment,
        )

        analyse_btn = st.button(
            "🔍 Analyse Comment",
            use_container_width=True,
        )

    with col_out:
        if analyse_btn and comment.strip():
            with st.spinner("Analysing with mBERT…"):
                try:
                    clf = load_model()
                    res = analyse_comment(comment, clf)
                    res["id"]   = len(st.session_state.history) + 1
                    res["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.history.insert(0, res)

                    s = res["sentiment"]
                    icons  = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
                    colors = {"Positive": "#E1EFD9", "Neutral": "#FAEEDA", "Negative": "#FCEBEB"}
                    tcols  = {"Positive": "#1B4D1B", "Neutral": "#633806", "Negative": "#791F1F"}

                    st.markdown(
                        f"""<div style='background:{colors[s]};border-radius:12px;
                        padding:1rem 1.25rem;margin-bottom:0.75rem'>
                        <span style='font-size:2rem'>{icons[s]}</span>
                        <span style='font-size:1.3rem;font-weight:700;color:{tcols[s]};
                        margin-left:10px'>{s}</span>
                        <div style='font-size:12px;color:#555;margin-top:4px'>
                        Confidence: {res['confidence']:.2%}</div></div>""",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Detected cooking aspects:**")
                    aspects_html = "".join(f'<span class="aspect-tag">{a}</span>' for a in res["aspects"])
                    st.markdown(aspects_html, unsafe_allow_html=True)
                    st.markdown("<br>**Preprocessed text:**", unsafe_allow_html=True)
                    st.code(res["clean"], language=None)

                except Exception as e:
                    st.error(f"Model loading or prediction error: {e}")

        elif analyse_btn:
            st.warning("Please enter a comment first.")
        else:
            st.markdown(
                "<div style='text-align:center;padding:3rem 0;color:#aaa'>"
                "🔍<br><small>Result will appear here</small></div>",
                unsafe_allow_html=True,
            )

    if st.session_state.history:
        st.markdown("---")
        st.markdown("#### 🕐 Recent analyses")
        for h in st.session_state.history[:5]:
            s    = h["sentiment"]
            badge = f'<span class="badge-{s}">{s}</span>'
            atags = " ".join(f'<span class="aspect-tag">{a}</span>' for a in h["aspects"])
            st.markdown(
                f"""<div class='result-box'>
                <div style='display:flex;justify-content:space-between;align-items:center'>
                  <div style='font-size:13px;color:#1a1a18;max-width:65%'>
                  {h['original'][:120]}{'…' if len(h['original'])>120 else ''}</div>
                  <div>{badge}&nbsp;<span style='font-size:12px;color:#888'>
                  {h['confidence']:.0%}</span></div>
                </div>
                <div style='margin-top:6px'>{atags}</div></div>""",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "📂 Batch Analysis":
    st.markdown("### 📂 Batch Comment Analysis")
    st.caption("Upload a CSV file of YouTube comments for bulk analysis.")

    tab1, tab2 = st.tabs(["📤 Upload CSV", "✏️ Paste Comments"])

    with tab1:
        st.markdown("**CSV must have a column named `comment`.**")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            if "comment" not in df.columns:
                st.error("CSV must contain a column named `comment`.")
            else:
                st.success(f"Loaded {len(df)} comments.")
                st.dataframe(df.head(5), use_container_width=True, hide_index=True)
                if st.button("🚀 Run Batch Analysis"):
                    clf = load_model()
                    results, bar = [], st.progress(0, text="Analysing…")
                    for i, row in df.iterrows():
                        results.append(analyse_comment(str(row["comment"]), clf))
                        bar.progress((i+1)/len(df), text=f"Analysing {i+1}/{len(df)}…")
                    bar.empty()
                    st.session_state["batch_results"] = pd.DataFrame(results)
                    st.success("Done!")

    with tab2:
        pasted = st.text_area(
            "Paste one comment per line:", height=200,
            placeholder="Sedap sangat resepi ni!\nLangkah memasak agak susah sikit.",
        )
        if st.button("🚀 Analyse Pasted Comments"):
            lines = [l.strip() for l in pasted.splitlines() if l.strip()]
            if lines:
                clf = load_model()
                results, bar = [], st.progress(0)
                for i, line in enumerate(lines):
                    results.append(analyse_comment(line, clf))
                    bar.progress((i+1)/len(lines))
                bar.empty()
                st.session_state["batch_results"] = pd.DataFrame(results)
            else:
                st.warning("Please paste at least one comment.")

    if "batch_results" in st.session_state:
        out = st.session_state["batch_results"]
        st.markdown("---")
        st.markdown(f"#### Results — {len(out)} comments")
        pos = (out["sentiment"]=="Positive").sum()
        neu = (out["sentiment"]=="Neutral").sum()
        neg = (out["sentiment"]=="Negative").sum()
        c1,c2,c3 = st.columns(3)
        c1.markdown(f"<div class='metric-card'><div class='metric-num' style='color:#1B4D1B'>{pos}</div><div class='metric-label'>Positive</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-num' style='color:#854F0B'>{neu}</div><div class='metric-label'>Neutral</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-num' style='color:#A32D2D'>{neg}</div><div class='metric-label'>Negative</div></div>", unsafe_allow_html=True)
        st.markdown("")
        display = out[["original","sentiment","confidence","aspects"]].copy()
        display.columns = ["Comment","Sentiment","Confidence","Aspects"]
        display["Confidence"] = display["Confidence"].apply(lambda x: f"{x:.0%}")
        display["Aspects"]    = display["Aspects"].apply(lambda x: ", ".join(x))
        st.dataframe(display, use_container_width=True, height=300, hide_index=True)
        st.download_button("⬇️ Download results CSV", data=out.to_csv(index=False).encode("utf-8"),
                           file_name="sentiment_results.csv", mime="text/csv")

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — DASHBOARD  (Confidence histogram & Aspect×Sentiment heatmap removed)
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown("### 📊 Analytics Dashboard")
    all_data = st.session_state.history.copy()
    batch    = st.session_state.get("batch_results", None)
    if batch is not None:
        all_data = batch.to_dict("records") + all_data

    if not all_data:
        st.info("No data yet — analyse some comments first on the Home or Batch page.")
    else:
        df    = pd.DataFrame(all_data)
        total = len(df)
        pos   = (df["sentiment"]=="Positive").sum()
        neu   = (df["sentiment"]=="Neutral").sum()
        neg   = (df["sentiment"]=="Negative").sum()

        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><div class='metric-num'>{total}</div><div class='metric-label'>Total comments</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-num' style='color:#1B4D1B'>{pos}</div><div class='metric-label'>Positive</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-num' style='color:#854F0B'>{neu}</div><div class='metric-label'>Neutral</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><div class='metric-num' style='color:#A32D2D'>{neg}</div><div class='metric-label'>Negative</div></div>", unsafe_allow_html=True)

        st.markdown("")
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Sentiment distribution**")
            pie = go.Figure(go.Pie(
                labels=["Positive","Neutral","Negative"], values=[pos,neu,neg],
                marker_colors=["#2E7D32","#BA7517","#E24B4A"], hole=0.45, textinfo="label+percent",
            ))
            pie.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=280,
                              showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#1a1a18"))
            st.plotly_chart(pie, use_container_width=True)

        with col_r:
            st.markdown("**Top detected aspects**")
            aspect_counts = {}
            for row in all_data:
                for a in row.get("aspects", []):
                    aspect_counts[a] = aspect_counts.get(a, 0) + 1
            aspect_df = pd.DataFrame(sorted(aspect_counts.items(), key=lambda x:-x[1]), columns=["Aspect","Count"])
            bar_fig = px.bar(aspect_df, x="Count", y="Aspect", orientation="h", color_discrete_sequence=["#2E7D32"])
            bar_fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=280,
                                  yaxis_title="", xaxis_title="",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#1a1a18"))
            bar_fig.update_xaxes(tickfont=dict(color="#1a1a18"))
            bar_fig.update_yaxes(tickfont=dict(color="#1a1a18"))
            st.plotly_chart(bar_fig, use_container_width=True)

        st.download_button("⬇️ Download all data CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="dashboard_data.csv", mime="text/csv")

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ═══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("### ℹ️ About This System")
    st.markdown("""
    **Sentiment Analysis of YouTube Comments**
    — Nur Najlaa' Alyaa' Binti Roslan (2023436326), UiTM, July 2026

    ---

    #### System overview
    This system analyses Malay YouTube cooking comments to classify their sentiment and extract
    cooking-related aspects, helping content creators understand audience feedback at scale.

    | Component | Technology |
    |---|---|
    | UI Framework | Streamlit |
    | Sentiment Model | Custom fine-tuned mBERT (`nvjlaa/mBERT`) |
    | Aspect Extraction | Keyword-based (Malay cooking vocabulary) |
    | Data Collection | Apify YouTube Comments Scraper |
    | Visualisation | Plotly |

    #### Sentiment labels
    | Label | Description |
    |---|---|
    | ✅ Positive | Viewer liked the video or recipe |
    | 😐 Neutral | Mixed or no clear opinion |
    | ❌ Negative | Viewer disliked or criticised |

    #### Cooking aspects detected
    Each comment is assigned to exactly one aspect:
    - **Taste / Rasa**
    - **Ingredients / Bahan**
    - **Cooking Steps / Langkah**
    - **Time / Masa**
    - **Presentation / Persembahan**
    - **Texture / Tekstur**
    - **General** when no aspect keyword is found

    #### How to run locally
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """)
