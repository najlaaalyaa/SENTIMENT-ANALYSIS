import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from model import load_model, analyse_comment, preprocess

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
    "Resepi ni sangat sedap dan rasanya memang terbaik!"
)


def set_demo_comment():
    """Place the demo sentence inside the comment text area."""
    st.session_state.comment_input = DEMO_COMMENT


# ── Batch data preprocessing ───────────────────────────────────
def preprocess_dataframe(
    dataframe: pd.DataFrame,
    row_offset: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Clean comments before batch prediction.

    Steps:
    1. Detect missing and blank comments.
    2. Apply the same preprocess() function used by model.py.
    3. Remove comments that become empty after cleaning.
    4. Remove duplicate comments based on cleaned text.

    row_offset=2 is used for CSV files because row 1 is the header.
    row_offset=1 is used for pasted comments.
    """
    working = dataframe.copy().reset_index(drop=True)

    if "comment" not in working.columns:
        raise KeyError("The data must contain a column named 'comment'.")

    valid_rows = []
    skipped_rows = []
    seen_clean_comments = set()

    missing_or_blank = 0
    empty_after_cleaning = 0
    duplicates_removed = 0

    for index, raw_value in working["comment"].items():
        source_row = index + row_offset

        # Missing cell such as NaN or None.
        if pd.isna(raw_value):
            missing_or_blank += 1
            skipped_rows.append({
                "Row": source_row,
                "Comment": "",
                "Cleaned Comment": "",
                "Reason": "Missing or blank comment",
            })
            continue

        original_comment = str(raw_value).strip()

        # Empty string or whitespace-only string.
        if not original_comment:
            missing_or_blank += 1
            skipped_rows.append({
                "Row": source_row,
                "Comment": "",
                "Cleaned Comment": "",
                "Reason": "Missing or blank comment",
            })
            continue

        clean_comment = preprocess(original_comment)

        # Emoji-only, URL-only, mention-only, or symbol-only comments may
        # become empty after the preprocessing function removes them.
        if not clean_comment.strip():
            empty_after_cleaning += 1
            skipped_rows.append({
                "Row": source_row,
                "Comment": original_comment,
                "Cleaned Comment": "",
                "Reason": "Empty after preprocessing",
            })
            continue

        # Remove duplicate comments using their cleaned form, so comments
        # with differences only in casing, links, mentions, or spacing are
        # treated as duplicates.
        duplicate_key = clean_comment.casefold()

        if duplicate_key in seen_clean_comments:
            duplicates_removed += 1
            skipped_rows.append({
                "Row": source_row,
                "Comment": original_comment,
                "Cleaned Comment": clean_comment,
                "Reason": "Duplicate cleaned comment",
            })
            continue

        seen_clean_comments.add(duplicate_key)

        valid_rows.append({
            "Source Row": source_row,
            "original_comment": original_comment,
            "clean_comment": clean_comment,
        })

    prepared_df = pd.DataFrame(
        valid_rows,
        columns=["Source Row", "original_comment", "clean_comment"],
    )

    skipped_df = pd.DataFrame(
        skipped_rows,
        columns=["Row", "Comment", "Cleaned Comment", "Reason"],
    )

    statistics = {
        "Original comments": len(working),
        "Valid comments": len(prepared_df),
        "Missing or blank": missing_or_blank,
        "Empty after cleaning": empty_after_cleaning,
        "Duplicates removed": duplicates_removed,
        "Total removed": len(skipped_df),
    }

    return prepared_df, skipped_df, statistics


def show_preprocessing_summary(
    prepared_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    statistics: dict,
    key_prefix: str,
) -> None:
    """Display preprocessing totals, a preview, and download controls."""
    st.markdown("#### 🧹 Data preprocessing summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Original",
        statistics["Original comments"],
    )
    col2.metric(
        "Valid",
        statistics["Valid comments"],
    )
    col3.metric(
        "Blank removed",
        statistics["Missing or blank"],
    )
    col4.metric(
        "Invalid removed",
        statistics["Empty after cleaning"],
    )
    col5.metric(
        "Duplicates",
        statistics["Duplicates removed"],
    )

    if not prepared_df.empty:
        st.markdown("**Preview after preprocessing:**")

        preview = prepared_df[
            ["Source Row", "original_comment", "clean_comment"]
        ].head(20).copy()

        preview.columns = [
            "Source Row",
            "Original Comment",
            "Cleaned Comment",
        ]

        st.dataframe(
            preview,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download preprocessed comments",
            data=prepared_df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="preprocessed_comments.csv",
            mime="text/csv",
            key=f"{key_prefix}_download_preprocessed",
        )
    else:
        st.warning(
            "No valid comments remain after preprocessing."
        )

    if not skipped_df.empty:
        with st.expander(
            f"View {len(skipped_df)} removed comments"
        ):
            st.dataframe(
                skipped_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "⬇️ Download removed comments",
                data=skipped_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="removed_comments.csv",
                mime="text/csv",
                key=f"{key_prefix}_download_removed",
            )


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
                    st.markdown("**Detected cooking aspect:**")
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
    st.caption(
        "Upload or paste comments. The app cleans the data before "
        "sending valid comments to the trained model."
    )

    tab1, tab2 = st.tabs([
        "📤 Upload CSV",
        "✏️ Paste Comments",
    ])

    # ----------------------------------------------------------
    # TAB 1: CSV UPLOAD
    # ----------------------------------------------------------
    with tab1:
        st.markdown(
            "**The CSV file must contain a column named `comment`.**"
        )
        st.caption(
            "The app removes missing, blank, emoji-only, URL-only, "
            "mention-only, symbol-only, and duplicate comments."
        )

        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="batch_csv_uploader",
        )

        if uploaded is not None:
            try:
                source_df = pd.read_csv(uploaded)
            except UnicodeDecodeError:
                uploaded.seek(0)
                source_df = pd.read_csv(
                    uploaded,
                    encoding="latin-1",
                )
            except Exception as error:
                st.error(f"Unable to read the CSV file: {error}")
                source_df = None

            if source_df is not None:
                if "comment" not in source_df.columns:
                    st.error(
                        "CSV must contain a column named `comment`. "
                        f"Available columns: {list(source_df.columns)}"
                    )
                else:
                    prepared_df, initial_skipped_df, stats = (
                        preprocess_dataframe(
                            source_df,
                            row_offset=2,
                        )
                    )

                    st.success(
                        f"Loaded {len(source_df)} original comments."
                    )

                    show_preprocessing_summary(
                        prepared_df,
                        initial_skipped_df,
                        stats,
                        key_prefix="csv",
                    )

                    if st.button(
                        "🚀 Run Batch Analysis",
                        key="run_csv_batch",
                        disabled=prepared_df.empty,
                    ):
                        try:
                            clf = load_model()
                        except Exception as error:
                            st.error(
                                f"Unable to load the trained model: {error}"
                            )
                        else:
                            results = []
                            prediction_skipped = []

                            total = len(prepared_df)

                            bar = st.progress(
                                0,
                                text="Starting analysis…",
                            )

                            for position, (_, row) in enumerate(
                                prepared_df.iterrows(),
                                start=1,
                            ):
                                try:
                                    prediction = analyse_comment(
                                        row["original_comment"],
                                        clf,
                                    )

                                    prediction["source_row"] = int(
                                        row["Source Row"]
                                    )
                                    results.append(prediction)

                                except Exception as error:
                                    prediction_skipped.append({
                                        "Row": int(row["Source Row"]),
                                        "Comment": row["original_comment"],
                                        "Cleaned Comment": row["clean_comment"],
                                        "Reason": (
                                            f"Prediction error: {error}"
                                        ),
                                    })

                                bar.progress(
                                    position / total,
                                    text=(
                                        f"Analysing "
                                        f"{position}/{total}…"
                                    ),
                                )

                            bar.empty()

                            result_df = pd.DataFrame(results)

                            all_skipped_df = pd.concat(
                                [
                                    initial_skipped_df,
                                    pd.DataFrame(
                                        prediction_skipped,
                                        columns=[
                                            "Row",
                                            "Comment",
                                            "Cleaned Comment",
                                            "Reason",
                                        ],
                                    ),
                                ],
                                ignore_index=True,
                            )

                            if not result_df.empty:
                                st.session_state[
                                    "batch_results"
                                ] = result_df
                            else:
                                st.session_state.pop(
                                    "batch_results",
                                    None,
                                )

                            st.session_state[
                                "skipped_comments"
                            ] = all_skipped_df

                            st.success(
                                f"Completed: {len(result_df)} analysed, "
                                f"{len(all_skipped_df)} removed or skipped."
                            )

                            if not all_skipped_df.empty:
                                with st.expander(
                                    "View all removed or skipped comments"
                                ):
                                    st.dataframe(
                                        all_skipped_df,
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                                    st.download_button(
                                        "⬇️ Download skipped comments CSV",
                                        data=all_skipped_df.to_csv(
                                            index=False
                                        ).encode("utf-8-sig"),
                                        file_name=(
                                            "skipped_comments.csv"
                                        ),
                                        mime="text/csv",
                                        key="csv_prediction_skipped",
                                    )

    # ----------------------------------------------------------
    # TAB 2: PASTED COMMENTS
    # ----------------------------------------------------------
    with tab2:
        pasted = st.text_area(
            "Paste one comment per line:",
            height=220,
            placeholder=(
                "Sedap sangat resepi ni!\n"
                "Langkah memasak agak susah sikit.\n"
                "👏👏👏"
            ),
            key="pasted_batch_comments",
        )

        pasted_lines = pasted.splitlines()

        if any(line.strip() for line in pasted_lines):
            pasted_source_df = pd.DataFrame({
                "comment": pasted_lines
            })

            pasted_prepared_df, pasted_skipped_df, pasted_stats = (
                preprocess_dataframe(
                    pasted_source_df,
                    row_offset=1,
                )
            )

            show_preprocessing_summary(
                pasted_prepared_df,
                pasted_skipped_df,
                pasted_stats,
                key_prefix="pasted",
            )
        else:
            pasted_prepared_df = pd.DataFrame(
                columns=[
                    "Source Row",
                    "original_comment",
                    "clean_comment",
                ]
            )
            pasted_skipped_df = pd.DataFrame(
                columns=[
                    "Row",
                    "Comment",
                    "Cleaned Comment",
                    "Reason",
                ]
            )

        if st.button(
            "🚀 Analyse Pasted Comments",
            key="run_pasted_batch",
            disabled=pasted_prepared_df.empty,
        ):
            try:
                clf = load_model()
            except Exception as error:
                st.error(
                    f"Unable to load the trained model: {error}"
                )
            else:
                results = []
                prediction_skipped = []
                total = len(pasted_prepared_df)

                bar = st.progress(
                    0,
                    text="Starting analysis…",
                )

                for position, (_, row) in enumerate(
                    pasted_prepared_df.iterrows(),
                    start=1,
                ):
                    try:
                        prediction = analyse_comment(
                            row["original_comment"],
                            clf,
                        )
                        prediction["source_row"] = int(
                            row["Source Row"]
                        )
                        results.append(prediction)
                    except Exception as error:
                        prediction_skipped.append({
                            "Row": int(row["Source Row"]),
                            "Comment": row["original_comment"],
                            "Cleaned Comment": row["clean_comment"],
                            "Reason": f"Prediction error: {error}",
                        })

                    bar.progress(
                        position / total,
                        text=(
                            f"Analysing {position}/{total}…"
                        ),
                    )

                bar.empty()

                result_df = pd.DataFrame(results)

                all_skipped_df = pd.concat(
                    [
                        pasted_skipped_df,
                        pd.DataFrame(
                            prediction_skipped,
                            columns=[
                                "Row",
                                "Comment",
                                "Cleaned Comment",
                                "Reason",
                            ],
                        ),
                    ],
                    ignore_index=True,
                )

                if not result_df.empty:
                    st.session_state["batch_results"] = result_df
                else:
                    st.session_state.pop(
                        "batch_results",
                        None,
                    )

                st.session_state[
                    "skipped_comments"
                ] = all_skipped_df

                st.success(
                    f"Completed: {len(result_df)} analysed, "
                    f"{len(all_skipped_df)} removed or skipped."
                )

    # ----------------------------------------------------------
    # BATCH RESULTS
    # ----------------------------------------------------------
    if "batch_results" in st.session_state:
        out = st.session_state["batch_results"]

        if not out.empty:
            st.markdown("---")
            st.markdown(
                f"#### Results — {len(out)} comments"
            )

            positive_count = (
                out["sentiment"] == "Positive"
            ).sum()
            neutral_count = (
                out["sentiment"] == "Neutral"
            ).sum()
            negative_count = (
                out["sentiment"] == "Negative"
            ).sum()

            col1, col2, col3 = st.columns(3)

            col1.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-num' "
                f"style='color:#1B4D1B'>"
                f"{positive_count}</div>"
                f"<div class='metric-label'>Positive</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            col2.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-num' "
                f"style='color:#854F0B'>"
                f"{neutral_count}</div>"
                f"<div class='metric-label'>Neutral</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            col3.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-num' "
                f"style='color:#A32D2D'>"
                f"{negative_count}</div>"
                f"<div class='metric-label'>Negative</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("")

            display_columns = [
                "original",
                "clean",
                "sentiment",
                "confidence",
                "aspects",
            ]

            if "source_row" in out.columns:
                display_columns.insert(0, "source_row")

            display = out[display_columns].copy()

            rename_map = {
                "source_row": "Source Row",
                "original": "Original Comment",
                "clean": "Cleaned Comment",
                "sentiment": "Sentiment",
                "confidence": "Confidence",
                "aspects": "Aspect",
            }

            display = display.rename(columns=rename_map)

            display["Confidence"] = display[
                "Confidence"
            ].apply(lambda value: f"{value:.2%}")

            display["Aspect"] = display[
                "Aspect"
            ].apply(
                lambda values: (
                    values[0]
                    if isinstance(values, list) and values
                    else str(values)
                )
            )

            st.dataframe(
                display,
                use_container_width=True,
                height=360,
                hide_index=True,
            )

            export_df = out.copy()

            export_df["aspects"] = export_df[
                "aspects"
            ].apply(
                lambda values: (
                    values[0]
                    if isinstance(values, list) and values
                    else str(values)
                )
            )

            st.download_button(
                "⬇️ Download analysis results CSV",
                data=export_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="sentiment_results.csv",
                mime="text/csv",
                key="download_batch_results",
            )

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
            st.markdown("**Detected aspect distribution**")
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
    - **General** when no aspect keyword is found

    #### How to run locally
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """)
