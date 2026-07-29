import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------------------
# 1. 頁面基本配置
# ---------------------------------------------------------
st.set_page_config(
    page_title="LBMS 流動性與泡沫預警系統",
    page_icon="📈",
    layout="wide"
)

st.title("🛡️ LBMS 自動化流動性與泡沫預警系統 (含歷史回測)")
st.caption("透過微觀結構、波動率及信用利差，客觀監測資產類高點與流動性風險。")

# ---------------------------------------------------------
# 2. 側邊欄控制項
# ---------------------------------------------------------
st.sidebar.header("⚙️ 系統參數設定")

target_symbol = st.sidebar.text_input("監控標的代碼 (Ticker)", value="SPY")
hyg_symbol = "HYG"
tlt_symbol = "TLT"

st.sidebar.subheader("🎯 警戒閾值調整")
b_wave_min = st.sidebar.slider("類高點 (B浪) 下限 (ATH %)", 50, 95, 80) / 100.0
b_wave_max = st.sidebar.slider("類高點 (B浪) 上限 (ATH %)", 60, 100, 92) / 100.0
vol_quantile = st.sidebar.slider("VaR 波動率高位分位數 (%)", 75, 99, 90) / 100.0

# ---------------------------------------------------------
# 3. 數據抓取與計算
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(ticker, period="max", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

with st.spinner("正在讀取歷史市場數據..."):
    df_target = load_data(target_symbol)
    df_hyg = load_data(hyg_symbol)
    df_tlt = load_data(tlt_symbol)

if df_target.empty:
    st.error(f"無法取得標的 '{target_symbol}' 數據，請確認代碼是否正確。")
    st.stop()

# 數據時間對齊
common_index = df_target.index.intersection(df_hyg.index).intersection(df_tlt.index)
df_t = df_target.loc[common_index].copy()
df_h = df_hyg.loc[common_index].copy()
df_l = df_tlt.loc[common_index].copy()

# 指標計算
df_t['ATH'] = df_t['High'].cummax()
df_t['ATH_Ratio'] = df_t['Close'] / df_t['ATH']

df_t['Returns'] = df_t['Close'].pct_change()
df_t['Vol_20d'] = df_t['Returns'].rolling(window=20).std() * (252 ** 0.5)
vol_threshold_hist = df_t['Vol_20d'].expanding().quantile(vol_quantile)

credit_ratio = df_h['Close'] / df_l['Close']
credit_mavg = credit_ratio.rolling(20).mean()
credit_threshold = credit_mavg * 0.97

# 訊號歷史判定 (0:綠, 1:黃, 2:橘, 3:紅)
cond_b_wave = (df_t['ATH_Ratio'] >= b_wave_min) & (df_t['ATH_Ratio'] <= b_wave_max)
cond_vol = df_t['Vol_20d'] > vol_threshold_hist
cond_credit = credit_ratio < credit_threshold

signal_score = cond_b_wave.astype(int) + cond_vol.astype(int) + cond_credit.astype(int)
df_t['Signal'] = signal_score # 0=綠, 1=黃, 2=橘, >=3=紅

# ---------------------------------------------------------
# 4. 頁面分頁結構 (Tabs)
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📊 即時風控儀表板", "📈 歷史數據回測分析"])

# =========================================================
# TAB 1: 即時儀表板
# =========================================================
with tab1:
    current_price = float(df_t['Close'].iloc[-1])
    ath_price = float(df_t['ATH'].iloc[-1])
    ath_ratio = float(df_t['ATH_Ratio'].iloc[-1])
    current_vol = float(df_t['Vol_20d'].iloc[-1])
    vol_thresh_now = float(vol_threshold_hist.iloc[-1])
    current_credit = float(credit_ratio.iloc[-1])
    thresh_credit_now = float(credit_threshold.iloc[-1])

    triggers = []
    if b_wave_min <= ath_ratio <= b_wave_max:
        triggers.append(f"進入類高點危險區 (當前為 ATH 的 {ath_ratio*100:.1f}%)")
    if current_vol > vol_thresh_now:
        triggers.append(f"波動率爆表 (當前 {current_vol*100:.1f}% > 門檻 {vol_thresh_now*100:.1f}%)")
    if current_credit < thresh_credit_now:
        triggers.append("信用利差惡化 (高收益債相對強度偏弱)")

    trigger_count = len(triggers)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("當前價格", f"${current_price:.2f}", f"ATH: ${ath_price:.2f}")
    col2.metric("相對於 ATH 比例", f"{ath_ratio*100:.1f}%")
    col3.metric("20日年化波動率", f"{current_vol*100:.1f}%", f"門檻: {vol_thresh_now*100:.1f}%", delta_color="inverse")
    col4.metric("信用利差 Proxy", f"{current_credit:.2f}", f"門檻: {thresh_credit_now:.2f}", delta_color="normal")

    st.divider()

    if trigger_count == 0:
        st.success("🟢 **當前燈號：綠燈 (系統安全)**\n\n市場結構與流動性正常，可維持原配置。")
    elif trigger_count == 1:
        st.warning("🟡 **當前燈號：黃燈 (高度警戒)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 停止開槓桿，取消追高買單。")
    elif trigger_count == 2:
        st.error("🟠 **當前燈號：橘燈 (逃生區/減碼)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 現貨減碼 50%，清空槓桿部位。")
    else:
        st.error("🔴 **當前燈號：紅燈 (極限離場)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 執行無條件清倉 (Market Sell) 並轉入現金避險。")

    st.subheader("📊 價格走勢與類高點警戒區間")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df_t.index[-500:], y=df_t['Close'].iloc[-500:], name="收盤價", line=dict(color='skyblue', width=2)))
    fig_price.add_hline(y=ath_price, line_dash="dash", line_color="gray", annotation_text="歷史最高點 (ATH)")
