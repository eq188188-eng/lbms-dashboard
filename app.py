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

st.title("🛡️ LBMS 自動化流動性與泡沫預警系統 (精準低接版)")
st.caption("減碼純看流動性危機（B浪、歷史波動、信用利差、VIX），加碼則結合 IV/VIX 恐慌解除與多維均線（MA20/60/240）支撐。")

# ---------------------------------------------------------
# 2. 數據抓取與預處理
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(ticker, period="max", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

st.sidebar.header("⚙️ 系統參數設定")
target_symbol = st.sidebar.text_input("監控標的代碼 (Ticker)", value="SOXL")
hyg_symbol = "HYG"
tlt_symbol = "TLT"
vix_symbol = "^VIX"

with st.spinner(f"正在讀取 {target_symbol}、HYG、TLT 與 VIX 市場數據..."):
    df_target = load_data(target_symbol)
    df_hyg = load_data(hyg_symbol)
    df_tlt = load_data(tlt_symbol)
    df_vix = load_data(vix_symbol)

if df_target.empty:
    st.error(f"無法取得標的 '{target_symbol}' 數據，請確認代碼是否正確。")
    st.stop()

# 數據時間對齊與空值清理
common_index = df_target.index.intersection(df_hyg.index).intersection(df_tlt.index).intersection(df_vix.index)
df_t = df_target.loc[common_index].copy()
df_h = df_hyg.loc[common_index].copy()
df_l = df_tlt.loc[common_index].copy()
df_v = df_vix.loc[common_index].copy()

# 基礎指標與多維均線計算
df_t['ATH'] = df_t['High'].cummax()
df_t['ATH_Ratio'] = df_t['Close'] / df_t['ATH']
df_t['Returns'] = df_t['Close'].pct_change()
df_t['Vol_20d'] = df_t['Returns'].rolling(window=20).std() * (252 ** 0.5)

df_t['MA20'] = df_t['Close'].rolling(window=20).mean()
df_t['MA60'] = df_t['Close'].rolling(window=60).mean()
df_t['MA240'] = df_t['Close'].rolling(window=240).mean()

# 信用利差 Proxy
credit_ratio = df_h['Close'] / df_l['Close']
credit_mavg = credit_ratio.rolling(20).mean()
credit_threshold = credit_mavg * 0.97
cond_credit = credit_ratio < credit_threshold

# VIX 指標邏輯
vix_close = df_v['Close']
vix_mavg = vix_close.rolling(60).mean()
cond_vix = (vix_close > vix_mavg * 1.2) | (vix_close > 25.0)

# ---------------------------------------------------------
# 3. 回測計算核心函數 (減碼不看MA，加碼結合IV/VIX與MA支撐)
# ---------------------------------------------------------
def run_backtest(df, b_min, b_max, v_quant):
    vol_thresh = df['Vol_20d'].expanding().quantile(v_quant)
    cond_b = (df['ATH_Ratio'] >= b_min) & (df['ATH_Ratio'] <= b_max)
    cond_vol = df['Vol_20d'] > vol_thresh
    
    # 風險評分（用於減碼/倉位控制：不看 MA）
    score = (
        cond_b.astype(int) + 
        cond_vol.astype(int) + 
        cond_credit.loc[df.index].astype(int) + 
        cond_vix.loc[df.index].astype(int)
    )
    
    prev_score = score.shift(1).fillna(0)
    
    # 加碼條件：流動性危機解除 (score == 0)，且 VIX 降溫，且價格位於三條均線（MA20/MA60/MA240）附近或下方安全區
    vix_cooling = vix_close.loc[df.index] < vix_mavg.loc[df.index]
    price_near_support = (df['Close'] < df['MA20']) | (df['Close'] < df['MA60']) | (df['Close'] < df['MA240'] * 1.05)
    
    add_signal = (prev_score >= 1) & (score == 0) & vix_cooling & price_near_support
    
    # 動態倉位控制
    position = np.where(score >= 3, 0.0, np.where(score == 2, 0.3, np.where(score == 1, 0.7, 1.0)))
    pos_series = pd.Series(position, index=df.index).shift(1).fillna(1.0)
    
    strat_ret = df['Returns'] * pos_series
    cum_strat = (1 + strat_ret.fillna(0)).cumprod()
    mdd = ((cum_strat / cum_strat.cummax()) - 1).min()
    total_ret = cum_strat.iloc[-1] - 1
    return total_ret, mdd, score, add_signal

df_clean = df_t.dropna(subset=['Vol_20d', 'ATH_Ratio', 'Returns', 'MA20', 'MA60', 'MA240']).copy()

# Session State 預設參數
if "best_params" not in st.session_state:
    st.session_state["best_params"] = {"b_min": 0.65, "b_max": 0.97, "v_quant": 0.85}

st.sidebar.subheader("🎯 參數與一鍵最佳化")

if st.sidebar.button("⚡ 尋找該標的歷史最佳參數"):
    with st.spinner("正在執行網格搜尋 (Grid Search) 尋找最佳風控參數..."):
        best_score_metric = -999.0
        best_combo = (0.65, 0.97, 0.85)
        
        for b_min in np.arange(0.60, 0.85, 0.05):
            for b_max in np.arange(0.85, 0.98, 0.03):
                for v_q in np.arange(0.75, 0.95, 0.05):
                    ret, mdd, _, _ = run_backtest(df_clean, b_min, b_max, v_q)
                    score_metric = ret + (mdd * 2.0)
                    if score_metric > best_score_metric:
                        best_score_metric = score_metric
                        best_combo = (round(b_min, 2), round(b_max, 2), round(v_q, 2))
        
        st.session_state["best_params"] = {
            "b_min": best_combo[0],
            "b_max": best_combo[1],
            "v_quant": best_combo[2]
        }
        st.sidebar.success(f"已套用最佳參數！\nB浪: {int(best_combo[0]*100)}%~{int(best_combo[1]*100)}%, VaR: {int(best_combo[2]*100)}%")

b_wave_min = st.sidebar.slider("類高點 (B浪) 下限 (ATH %)", 50, 95, int(st.session_state["best_params"]["b_min"] * 100)) / 100.0
b_wave_max = st.sidebar.slider("類高點 (B浪) 上限 (ATH %)", 60, 100, int(st.session_state["best_params"]["b_max"] * 100)) / 100.0
vol_quantile = st.sidebar.slider("VaR 波動率高位分位數 (%)", 75, 99, int(st.session_state["best_params"]["v_quant"] * 100)) / 100.0

total_ret, mdd_strat, signal_score, add_signals = run_
