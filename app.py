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

st.title("🛡️ LBMS 自動化流動性與泡沫預警系統 (一鍵最佳化旗艦版)")
st.caption("結合 B浪警戒、VIX 恐慌、信用利差、多維均線（MA20/60/240）與一鍵最佳化參數搜尋。")

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
# 3. 回測計算與加碼點判定核心函數
# ---------------------------------------------------------
def run_backtest(df, b_min, b_max, v_quant):
    vol_thresh = df['Vol_20d'].expanding().quantile(v_quant)
    cond_b = (df['ATH_Ratio'] >= b_min) & (df['ATH_Ratio'] <= b_max)
    cond_vol = df['Vol_20d'] > vol_thresh
    
    score = (
        cond_b.astype(int) + 
        cond_vol.astype(int) + 
        cond_credit.loc[df.index].astype(int) + 
        cond_vix.loc[df.index].astype(int)
    )
    
    prev_score = score.shift(1).fillna(0)
    add_signal = (prev_score >= 1) & (score == 0) & (df['Close'] < df['MA20'] * 0.98)
    
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

total_ret, mdd_strat, signal_score, add_signals = run_backtest(df_clean, b_wave_min, b_wave_max, vol_quantile)
df_clean['Signal'] = signal_score
df_clean['Add_Signal'] = add_signals

# ---------------------------------------------------------
# 4. 頁面分頁結構
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📊 即時風控與加碼儀表板", "📈 歷史數據回測分析"])

# =========================================================
# TAB 1: 即時儀表板
# =========================================================
with tab1:
    vol_threshold_hist = df_clean['Vol_20d'].expanding().quantile(vol_quantile)
    current_price = float(df_clean['Close'].iloc[-1])
    ath_price = float(df_clean['ATH'].iloc[-1])
    ath_ratio = float(df_clean['ATH_Ratio'].iloc[-1])
    current_vol = float(df_clean['Vol_20d'].iloc[-1])
    vol_thresh_now = float(vol_threshold_hist.iloc[-1])
    current_credit = float(credit_ratio.loc[df_clean.index[-1]])
    thresh_credit_now = float(credit_threshold.loc[df_clean.index[-1]])
    current_vix = float(vix_close.loc[df_clean.index[-1]])
    vix_mavg_now = float(vix_mavg.loc[df_clean.index[-1]])
    is_add_today = bool(df_clean['Add_Signal'].iloc[-1])

    triggers = []
    if b_wave_min <= ath_ratio <= b_wave_max:
        triggers.append(f"進入 B浪類高點危險區 (當前 ATH 比例: {ath_ratio*100:.1f}%)")
    if current_vol > vol_thresh_now:
        triggers.append(f"歷史波動率爆表 (當前 {current_vol*100:.1f}%)")
    if current_credit < thresh_credit_now:
        triggers.append("信用利差惡化 (高收益債相對強度偏弱)")
    if current_vix > vix_mavg_now * 1.2 or current_vix > 25.0:
        triggers.append(f"VIX 恐慌指數飆升 (當前 VIX: {current_vix:.2f})")

    trigger_count = len(triggers)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("當前價格", f"${current_price:.2f}", f"ATH: ${ath_price:.2f}")
    col2.metric("相對於 ATH 比例", f"{ath_ratio*100:.1f}%")
    col3.metric("VIX 恐慌指數", f"{current_vix:.2f}", f"均線: {vix_mavg_now:.2f}", delta_color="inverse")
    col4.metric("信用利差 Proxy", f"{current_credit:.2f}", f"門檻: {thresh_credit_now:.2f}", delta_color="normal")

    st.divider()

    if is_add_today:
        st.info("🔵 **當前訊號：拉回低接加碼點！ (Low-Risk Re-entry)**\n\n流動性危機解除且價格拉回至月線（MA20）下方具備安全邊際，建議分批低接加碼。")
    elif trigger_count == 0:
        st.success("🟢 **當前燈號：綠燈 (系統安全)**\n\n各項泡沫、波動與流動性指標正常，可維持原持倉。")
    elif trigger_count == 1:
        st.warning("🟡 **當前燈號：黃燈 (高度警戒)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 停止開槓桿，提高防備。")
    elif trigger_count == 2:
        st.error("🟠 **當前燈號：橘燈 (逃生區/減碼)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 現貨減碼 50%，清空槓桿部位。")
    else:
        st.error("🔴 **當前燈號：紅燈 (流動性危機/極限離場)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 執行無條件清倉 (Market Sell) 並轉入現金避險。")

    st.subheader(f"📊 {target_symbol} 近期價格走勢與 MA20
