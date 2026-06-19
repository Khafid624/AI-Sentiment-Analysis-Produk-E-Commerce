import streamlit as st
import pandas as pd
import plotly.express as px

from utils.predict import predict_sentiment
from utils.generative_ai import generate_insight


# =====================================
# CONFIG
# =====================================
st.set_page_config(
    page_title="AI Sentiment Analysis Dashboard",
    page_icon="🤖 ",
    layout="wide"
)

# =====================================
# SESSION STATE
# =====================================
if "df" not in st.session_state:
    st.session_state.df = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "insight" not in st.session_state:
    st.session_state.insight = ""


# =====================================
# HEADER
# =====================================
st.title("🤖 AI Sentiment Analysis Dashboard")
st.markdown("---")


# =====================================
# UPLOAD DATASET
# =====================================
uploaded_file = st.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset berhasil diupload")

    text_columns = df.select_dtypes(include="object").columns.tolist()

    if len(text_columns) == 0:
        st.error("Tidak ditemukan kolom teks")
        st.stop()

    col1, col2 = st.columns([3, 1])

    with col1:
        text_col = st.selectbox(
            "Pilih Kolom Review",
            text_columns
        )

    with col2:
        run = st.button("🚀 Analisis")

    df[text_col] = df[text_col].fillna("").astype(str)

    # =====================================
    # ANALISIS SENTIMEN
    # =====================================
    if run:

        with st.spinner("Sedang menganalisis..."):

            df["sentiment"] = df[text_col].apply(
                predict_sentiment
            )

            def kategori(sentiment):

                if sentiment == "Positif":
                    return "Pelanggan Puas"

                elif sentiment == "Negatif":
                    return "Perlu Perbaikan"

                else:
                    return "Pendapat Netral"

            df["kategori"] = df["sentiment"].apply(
                kategori
            )

            df["panjang_teks"] = (
                df[text_col]
                .astype(str)
                .apply(len)
            )

        st.session_state.df = df
        st.session_state.processed = True

        st.success("Analisis selesai")


# =====================================
# HASIL ANALISIS
# =====================================
if st.session_state.processed:

    df = st.session_state.df

    total = len(df)

    pos = (df["sentiment"] == "Positif").sum()
    net = (df["sentiment"] == "Netral").sum()
    neg = (df["sentiment"] == "Negatif").sum()

    positive_rate = round((pos / total) * 100, 2)

    # =====================================
    # FILTER
    # =====================================
    st.subheader("Filter")

    col1, col2 = st.columns(2)

    with col1:
        sentiment_filter = st.multiselect(
            "Sentiment",
            ["Positif", "Netral", "Negatif"],
            default=["Positif", "Netral", "Negatif"]
        )

    with col2:
        kategori_filter = st.multiselect(
            "Kategori",
            df["kategori"].unique(),
            default=df["kategori"].unique()
        )

    filtered_df = df[
        (df["sentiment"].isin(sentiment_filter))
        &
        (df["kategori"].isin(kategori_filter))
    ]

    st.markdown("---")

    # =====================================
    # KPI CARD
    # =====================================
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Total Opinions",
        total
    )

    c2.metric(
        "Positive",
        pos
    )

    c3.metric(
        "Negative",
        neg
    )

    c4.metric(
        "Neutral",
        net
    )

    c5.metric(
        "Positive Rate",
        f"{positive_rate}%"
    )

    st.markdown("---")

    # =====================================
    # DATA CHART
    # =====================================
    chart_data = pd.DataFrame({
        "Sentiment": ["Positif", "Netral", "Negatif"],
        "Jumlah": [pos, net, neg]
    })

    color_map = {
        "Positif": "#22c55e",
        "Netral": "#f59e0b",
        "Negatif": "#ef4444"
    }

    left, right = st.columns([1.2, 1.8])

    # =====================================
    # HORIZONTAL BAR
    # =====================================
    with left:

        st.subheader(
            "Distribution of Sentiment"
        )

        fig_bar = px.bar(
            chart_data,
            x="Jumlah",
            y="Sentiment",
            orientation="h",
            color="Sentiment",
            color_discrete_map=color_map
        )

        fig_bar.update_layout(
            showlegend=False,
            height=500
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    # =====================================
    # TABLE
    # =====================================
    with right:

        st.subheader(
            "Dataset Result"
        )

        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=500
        )

    st.markdown("---")

    # =====================================
    # PIE CHART + AI INSIGHT
    # =====================================
    colA, colB = st.columns(2)

    with colA:

        st.subheader(
            "Products by Sentiment"
        )

        fig_pie = px.pie(
            chart_data,
            names="Sentiment",
            values="Jumlah",
            color="Sentiment",
            color_discrete_map=color_map
        )

        fig_pie.update_layout(
            height=450
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with colB:

        st.subheader(
            "AI Insight"
        )

        if st.button(
            "✨ Generate Insight"
        ):

            with st.spinner(
                "AI sedang menganalisis..."
            ):

                st.session_state.insight = (
                    generate_insight(df)
                )

        if st.session_state.insight != "":
            st.info(
                st.session_state.insight
            )

    st.markdown("---")

    # =====================================
    # DATASET RESULT
    # =====================================
    st.subheader(
        "Dataset Result"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    # =====================================
    # DOWNLOAD
    # =====================================
    csv = (
        filtered_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Hasil",
        data=csv,
        file_name="sentiment_result.csv",
        mime="text/csv"
    )

else:

    st.info(
        ""
    )