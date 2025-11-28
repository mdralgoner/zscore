import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def calculate_rolling_zscore(df: pd.DataFrame, column_name: str, window: int) -> pd.Series:
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    shifted_series = df[column_name].shift(1)

    rolling_mean = shifted_series.rolling(window=window, min_periods=1).mean()
    rolling_std = shifted_series.rolling(window=window, min_periods=1).std()

    z_score = (df[column_name] - rolling_mean) / rolling_std
    z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0)

    return z_score



st.set_page_config(page_title="Анализ временного ряда по Z-score", layout="wide")
st.title("Приложение позволяет используя данные из FedNet подсчитывать Z-score с разными окнами")

col_input, col_upload = st.columns([1, 3])

with col_input:
    window = st.number_input("Ширина окна", min_value=1, value=360, step=1)

with col_upload:
    uploaded_file = st.file_uploader("Данные для вставки", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_excel(uploaded_file).iloc[4:,:]
        df.rename(columns={
            df.columns[0]: 'date',
            df.columns[1]: 'value',
            df.columns[2]: 'yoy'
        }, inplace=True)

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')


        for col in ['value', 'yoy']:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')


        df['zscore'] = calculate_rolling_zscore(df, 'yoy', window=window)


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

        fig.add_trace(
            go.Scatter(
                x=df['date'].tolist(),
                y=df['value'].tolist(),
                mode="lines",
                name="Исходное значение",
                line=dict(color="navy"),
            ),
            row=1, col=1
        )


        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['yoy'].tolist(),
                mode="lines",
                name="YoY изменение",
                line=dict(color="green"),
            ),
            row=2, col=1
        )
        fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dash"), row=2, col=1)


        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['zscore'].tolist(),
                mode="lines",
                name="Z-Score",
                line=dict(color="purple"),
            ),
            row=3, col=1
        )


        fig.add_hline(
            y=2, line=dict(color="red", width=1, dash="dash"), row=3, col=1,
            annotation_text="+2σ", annotation_position="top left"
        )
        fig.add_hline(
            y=-2, line=dict(color="red", width=1, dash="dash"), row=3, col=1,
            annotation_text="-2σ", annotation_position="bottom left"
        )
        fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dash"), row=3, col=1)


        fig.update_layout(
            height=900,
            title_text="Анализ временного ряда с использованием Z-значений",
            showlegend=True,
            template="plotly_white",
        )


        min_date = df['date'].min()
        max_date = df['date'].max()
        fig.update_xaxes(range=[min_date, max_date])

        fig.update_yaxes(title_text="Значение", row=1, col=1)
        fig.update_yaxes(title_text="Изменение YoY (%)", row=2, col=1)
        fig.update_yaxes(title_text="Z-значение (стандартные отклонения)", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)


        with st.expander("Данные"):
            st.dataframe(df)

    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
else:
    st.info("Нужно вставить Excel файл")
