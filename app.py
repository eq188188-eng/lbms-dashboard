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

st.title("🛡️ LBMS 自動化流動性與泡沫預警系統 (含加碼點與回測)")
st.caption("透過微觀結構、波動率及信用利差，客觀監測資產風險並自動尋找風控與加碼買點。")

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

with st.spinner(f"正在讀取 {target_symbol} 與歷史市場數據..."):
    df_target = load_data(target_symbol)
    df_hyg = load_data(hyg_symbol)
    df_tlt = load_data(tlt_symbol)

if df_target.empty:
    st.error(f"無法取得標的 '{target_symbol}' 數據，請確認代碼是否正確。")
    st.stop()

# 數據時間對齊與空值清理
common_index = df_target.index.intersection(df_hyg.index).intersection(df_tlt.index)
df_t = df_target.loc[common_index].copy()
df_h = df_hyg.loc[common_index].copy()
df_l = df_tlt.loc[common_index].copy()

# 基礎指標計算
df_t['ATH'] = df_t['High'].cummax()
df_t['ATH_Ratio'] = df_t['Close'] / df_t['ATH']
df_t['Returns'] = df_t['Close'].pct_change()
df_t['Vol_20d'] = df_t['Returns'].rolling(window=20).std() * (252 ** 0.5)
df_t['MA20'] = df_t['Close'].rolling(window=20).mean()

credit_ratio = df_h['Close'] / df_l['Close']
credit_mavg = credit_ratio.rolling(20).mean()
credit_threshold = credit_mavg * 0.97
cond_credit = credit_ratio < credit_threshold

# ---------------------------------------------------------
# 3. 回測計算與加碼點判定核心函數
# ---------------------------------------------------------
def run_backtest(df, b_min, b_max, v_quant):
    vol_thresh = df['Vol_20d'].expanding().quantile(v_quant)
    cond_b = (df['ATH_Ratio'] >= b_min) & (df['ATH_Ratio'] <= b_max)
    cond_v = df['Vol_20d'] > vol_thresh
    
    score = cond_b.astype(int) + cond_v.astype(int) + cond_credit.loc[df.index].astype(int)
    
    # 判斷加碼點邏輯：從警示燈號(>=1)轉為綠燈(0) 且 價格站在20日均線之上
    prev_score = score.shift(1).fillna(0)
    add_signal = (prev_score >= 1) & (score == 0) & (df['Close'] > df['MA20'])
    
    position = np.where(score >= 3, 0.0, np.where(score == 2, 0.5, 1.0))
    pos_series = pd.Series(position, index=df.index).shift(1).fillna(1.0)
    
    strat_ret = df['Returns'] * pos_series
    cum_strat = (1 + strat_ret.fillna(0)).cumprod()
    mdd = ((cum_strat / cum_strat.cummax()) - 1).min()
    total_ret = cum_strat.iloc[-1] - 1
    return total_ret, mdd, score, add_signal

df_clean = df_t.dropna(subset=['Vol_20d', 'ATH_Ratio', 'Returns', 'MA20']).copy()

# 預設最佳化 Session State
if "best_params" not in st.session_state:
    st.session_state["best_params"] = {"b_min": 0.65, "b_max": 0.97, "v_quant": 0.85}

# 一鍵尋找最佳參數按鈕
st.sidebar.subheader("🎯 參數與最佳化控制")

if st.sidebar.button("⚡ 尋找該標的歷史最佳參數"):
    with st.spinner("正在尋找最佳風控參數中 (Grid Search)..."):
        best_mdd = -999.0
        best_ret = -999.0
        best_combo = (0.65, 0.97, 0.85)
        best_score_metric = -999.0
        
        for b_min in np.arange(0.60, 0.85, 0.05):
            for b_max in np.arange(0.85, 0.98, 0.03):
                for v_q in np.arange(0.75, 0.95, 0.05):
                    ret, mdd, _, _ = run_backtest(df_clean, b_min, b_max, v_q)
                    score_metric = ret + (mdd * 2.0)
                    if score_metric > best_score_metric:
                        best_score_metric = score_metric
                        best_ret = ret
                        best_mdd = mdd
                        best_combo = (round(b_min, 2), round(b_max, 2), round(v_q, 2))
        
        st.session_state["best_params"] = {
            "b_min": best_combo[0],
            "b_max": best_combo[1],
            "v_quant": best_combo[2]
        }
        st.sidebar.success(f"已套用最佳參數！\nB浪: {int(best_combo[0]*100)}%~{int(best_combo[1]*100)}%, VaR: {int(best_
