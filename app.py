import streamlit as st
import anthropic
import json
import re
from datetime import datetime

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SentiAI — Sentiment Analysis",
    page_icon="💜",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f7ff;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Main background */
    .stApp {
        background-color: #f4f3fc;
    }

    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        border: 0.5px solid rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 14px;
        font-weight: 600;
        color: #1a1a18;
        margin-bottom: 0.75rem;
    }

    /* Sentiment badges */
    .badge-positive {
        background: #EAF3DE;
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1rem;
    }
    .badge-neutral {
        background: #FAEEDA;
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1rem;
    }
    .badge-negative {
        background: #FCEBEB;
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1rem;
    }

    .sent-positive { color: #27500A; font-size: 20px; font-weight: 600; }
    .sent-neutral  { color: #633806; font-size: 20px; font-weight: 600; }
    .sent-negative { color: #791F1F; font-size: 20px; font-weight: 600; }

    /* Score breakdown */
    .breakdown-pos {
        background: #EAF3DE;
        border-radius: 8px;
        text-align: center;
        padding: 10px;
        color: #27500A;
        font-weight: 600;
    }
    .breakdown-neu {
        background: #FAEEDA;
        border-radius: 8px;
        text-align: center;
        padding: 10px;
        color: #633806;
        font-weight: 600;
    }
    .breakdown-neg {
        background: #FCEBEB;
        border-radius: 8px;
        text-align: center;
        padding: 10px;
        color: #791F1F;
        font-weight: 600;
    }

    /* Pill badges in table */
    .pill-positive { background:#EAF3DE; color:#27500A; padding:2px 10px; border-radius:99px; font-size:12px; font-weight:500; }
    .pill-neutral  { background:#FAEEDA; color:#633806; padding:2px 10px; border-radius:99px; font-size:12px; font-weight:500; }
    .pill-negative { background:#FCEBEB; color:#791F1F; padding:2px 10px; border-radius:99px; font-size:12px; font-weight:500; }

    /* Analyze button */
    .stButton > button {
        background: #7F77DD !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.5rem !important;
        transition: background 0.2s !important;
    }
    .stButton > button:hover {
        background: #534AB7 !important;
    }

    /* Hide default streamlit header */
    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "next_id" not in st.session_state:
    st.session_state.next_id = 1
if "result" not in st.session_state:
    st.session_state.result = None

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💜 SentiAI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🕐 History", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#aaa'>Understanding emotions,<br>one text at a time.</small>",
        unsafe_allow_html=True,
    )

# ── Anthropic client ──────────────────────────────────────────
def get_client():
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        api_key = None
    if not api_key:
        st.error("⚠️ No API key found. Add `ANTHROPIC_API_KEY` to your Streamlit secrets.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)

# ── Analyze function ──────────────────────────────────────────
def analyze_sentiment(text: str, analysis_type: str) -> dict:
    type_map = {
        "Overall Sentiment":      "overall sentiment analysis",
        "Emotion Detection":      "emotion detection",
        "Aspect-based Sentiment": "aspect-based sentiment analysis",
    }
    client = get_client()

    prompt = f"""Analyze the sentiment of the following text using {type_map[analysis_type]}.

Text: "{text}"

Respond ONLY with a valid JSON object — no preamble, no markdown:
{{"sentiment":"Positive","confidence":0.92,"positive":92,"neutral":6,"negative":2,"description":"This text has a positive sentiment."}}

Rules:
- sentiment must be exactly "Positive", "Neutral", or "Negative"
- confidence is a float between 0 and 1
- positive, neutral, negative are integers that sum to 100"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)

# ══════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":

    # Header
    col_icon, col_title = st.columns([0.07, 0.93])
    with col_icon:
        st.markdown("### 💜")
    with col_title:
        st.markdown("### Sentiment Analysis")
        st.caption("Analyze the sentiment of your text instantly")

    st.markdown("")

    col_left, col_right = st.columns(2, gap="medium")

    # ── Left: Input ──
    with col_left:
        st.markdown('<div class="card"><div class="card-title">Analyze new text</div>', unsafe_allow_html=True)

        input_text = st.text_area(
            "Enter your text below:",
            placeholder="Type or paste your text here…",
            height=130,
        )

        analysis_type = st.selectbox(
            "Choose an analysis option:",
            ["Overall Sentiment", "Emotion Detection", "Aspect-based Sentiment"],
        )

        analyze_clicked = st.button("✨ Analyze", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Right: Result ──
    with col_right:
        st.markdown('<div class="card"><div class="card-title">Result</div>', unsafe_allow_html=True)

        if analyze_clicked:
            if not input_text.strip():
                st.warning("Please enter some text first.")
            else:
                with st.spinner("Analyzing…"):
                    try:
                        result = analyze_sentiment(input_text.strip(), analysis_type)
                        st.session_state.result = result

                        # Save to history
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.history.insert(0, {
                            "id":        st.session_state.next_id,
                            "text":      input_text.strip(),
                            "sentiment": result["sentiment"],
                            "score":     f"{result['confidence']:.2f}",
                            "date":      now,
                        })
                        st.session_state.next_id += 1

                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        st.session_state.result = None

        result = st.session_state.result

        if result is None:
            st.markdown(
                "<div style='text-align:center;padding:3rem 0;color:#aaa'>"
                "🔍<br><small>Your result will appear here</small></div>",
                unsafe_allow_html=True,
            )
        else:
            s = result["sentiment"].lower()
            icons = {"positive": "😊", "neutral": "😐", "negative": "😞"}
            icon  = icons.get(s, "🔍")

            # Badge
            st.markdown(
                f'<div class="badge-{s}">'
                f'<span style="font-size:2rem">{icon}</span>'
                f'<div>'
                f'<div class="sent-{s}">{result["sentiment"]}</div>'
                f'<div style="font-size:12px;color:#666">{result["description"]}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # Confidence bar
            conf = result["confidence"]
            st.markdown(f"**Confidence score** — `{conf:.2f}`")
            st.progress(conf)

            # Breakdown
            st.markdown("**Score breakdown**")
            b1, b2, b3 = st.columns(3)
            with b1:
                st.markdown(
                    f'<div class="breakdown-pos"><div style="font-size:1.4rem">{result["positive"]}%</div>'
                    f'<div style="font-size:11px">Positive</div></div>',
                    unsafe_allow_html=True,
                )
            with b2:
                st.markdown(
                    f'<div class="breakdown-neu"><div style="font-size:1.4rem">{result["neutral"]}%</div>'
                    f'<div style="font-size:11px">Neutral</div></div>',
                    unsafe_allow_html=True,
                )
            with b3:
                st.markdown(
                    f'<div class="breakdown-neg"><div style="font-size:1.4rem">{result["negative"]}%</div>'
                    f'<div style="font-size:11px">Negative</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # ── History table ──
    st.markdown("---")
    st.markdown("#### Recent history")

    if not st.session_state.history:
        st.info("No analyses yet — run your first one above.")
    else:
        pills = {
            "Positive": '<span class="pill-positive">Positive</span>',
            "Neutral":  '<span class="pill-neutral">Neutral</span>',
            "Negative": '<span class="pill-negative">Negative</span>',
        }

        table_html = """
        <table style="width:100%;border-collapse:collapse;font-size:13px">
          <thead>
            <tr style="border-bottom:1px solid #eee;color:#888;font-weight:500">
              <th style="padding:8px;text-align:left">ID</th>
              <th style="padding:8px;text-align:left">Text</th>
              <th style="padding:8px;text-align:left">Sentiment</th>
              <th style="padding:8px;text-align:left">Score</th>
              <th style="padding:8px;text-align:left">Date &amp; time</th>
            </tr>
          </thead>
          <tbody>
        """
        for h in st.session_state.history:
            snippet = (h["text"][:60] + "…") if len(h["text"]) > 60 else h["text"]
            pill    = pills.get(h["sentiment"], h["sentiment"])
            table_html += f"""
            <tr style="border-bottom:0.5px solid #f0f0f0">
              <td style="padding:10px 8px;color:#aaa">{h['id']}</td>
              <td style="padding:10px 8px;color:#555;max-width:220px">{snippet}</td>
              <td style="padding:10px 8px">{pill}</td>
              <td style="padding:10px 8px">{h['score']}</td>
              <td style="padding:10px 8px;color:#aaa">{h['date']}</td>
            </tr>
            """
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════════════════════
elif page == "🕐 History":
    st.markdown("### 🕐 Analysis history")
    st.caption("All your previous sentiment analyses")

    if not st.session_state.history:
        st.info("No analyses yet. Head to Home and run your first one!")
    else:
        for h in st.session_state.history:
            with st.expander(f"#{h['id']} — {h['text'][:60]}{'…' if len(h['text'])>60 else ''}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Sentiment", h["sentiment"])
                c2.metric("Confidence", h["score"])
                c3.metric("Date", h["date"].split(" ")[0])
                st.caption(f"Full text: {h['text']}")

        if st.button("🗑️ Clear all history"):
            st.session_state.history = []
            st.rerun()

