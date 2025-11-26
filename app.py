import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 1. Define the Z-Score calculation function
def calculate_rolling_zscore(df: pd.DataFrame, column_name: str, window: int) -> pd.Series:
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    shifted_series = df[column_name].shift(1)

    rolling_mean = shifted_series.rolling(window=window, min_periods=1).mean()
    rolling_std = shifted_series.rolling(window=window, min_periods=1).std()

    z_score = (df[column_name] - rolling_mean) / rolling_std
    z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0)

    return z_score


# 2. App Setup
st.set_page_config(page_title="Анализ временного ряда по Z-score", layout="wide")
st.title("Приложение позволяет используя данные из FedNet подсчитывать Z-score с разными окнами")

# 3. File Uploader
uploaded_file = st.file_uploader("Загрузи Excel файл сюда", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_excel(uploaded_file).iloc[3:,:]

        # 4. Settings Section (Widgets)
        st.subheader("Настройки")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Widget for the Window parameter
            window = st.number_input("Ширина окна", min_value=1, value=360, step=1)

        with col2:
            # Select which column represents the Date
            date_col = st.selectbox("Выбор колонки с датой", df.columns)

        with col3:
            # Select which column represents the Value (Raw)
            val_col = st.selectbox("Выбор колонки со значением", df.columns)

        with col4:
            # Select which column represents YoY
            yoy_col = st.selectbox("Выбор колонки YoY", df.columns)

        # 5. Data Processing
        # Sort by date
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)

        # Handle numeric conversion (replace comma with dot if string)
        for col in [val_col, yoy_col]:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate Z-Score
        df['zscore'] = calculate_rolling_zscore(df, yoy_col, window=window)

        # 6. Visualization (Plotly)
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                "Исходное значение во времени",
                "Изменение год-к-году (YoY), %",
                f"{window}-недельное скользящее Z-значение YoY",
            )
        )

        # Trace 1: Original Value
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[val_col],
                mode="lines",
                name="Исходное значение",
                line=dict(color="navy"),
            ),
            row=1, col=1
        )

        # Trace 2: YoY
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[yoy_col],
                mode="lines",
                name="YoY изменение",
                line=dict(color="green"),
            ),
            row=2, col=1
        )
        fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dash"), row=2, col=1)

        # Trace 3: Z-Score
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df['zscore'],
                mode="lines",
                name="Z-Score",
                line=dict(color="purple"),
            ),
            row=3, col=1
        )

        # Reference lines for Z-Score
        fig.add_hline(
            y=2, line=dict(color="red", width=1, dash="dash"), row=3, col=1,
            annotation_text="+2σ", annotation_position="top left"
        )
        fig.add_hline(
            y=-2, line=dict(color="red", width=1, dash="dash"), row=3, col=1,
            annotation_text="-2σ", annotation_position="bottom left"
        )
        fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dash"), row=3, col=1)

        # Layout updates
        fig.update_layout(
            height=900,
            title_text="Анализ временного ряда с использованием Z-значений",
            showlegend=True,
            template="plotly_white",
        )

        # Set X-axis range dynamically based on data
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        fig.update_xaxes(range=[min_date, max_date])

        fig.update_yaxes(title_text="Значение", row=1, col=1)
        fig.update_yaxes(title_text="Изменение YoY (%)", row=2, col=1)
        fig.update_yaxes(title_text="Z-значение (стандартные отклонения)", row=3, col=1)

        # Render Plot
        st.plotly_chart(fig, use_container_width=True)

        # Optional: Show Data
        with st.expander("Данные"):
            st.dataframe(df)

    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
else:
    st.info("Нужно вставить Excel файл")