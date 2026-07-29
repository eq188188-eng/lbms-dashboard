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
        b_min_pct = int(best_combo[0] * 100)
        b_max_pct = int(best_combo[1] * 100)
        var_pct = int(best_combo[2] * 100)
        st.sidebar.success(f"已套用最佳參數！\nB浪: {b_min_pct}%~{b_max_pct}%, VaR: {var_pct}%")

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
    is_add_today = bool(df_clean['Add_Signal'].iloc[-1])

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

    if is_add_today:
        st.info("🔵 **當前訊號：安全加碼點！ (Re-entry / Add Signal)**\n\n風險警報解除且站穩 20 日均線，建議分批加碼/補回滿倉部位。")
    elif trigger_count == 0:
        st.success("🟢 **當前燈號：綠燈 (系統安全)**\n\n市場結構與流動性正常，可維持原持倉。")
    elif trigger_count == 1:
        st.warning("🟡 **當前燈號：黃燈 (高度警戒)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 停止開槓桿，取消追高買單。")
    elif trigger_count == 2:
        st.error("🟠 **當前燈號：橘燈 (逃生區/減碼)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 現貨減碼 50%，清空槓桿部位。")
    else:
        st.error("🔴 **當前燈號：紅燈 (極限離場)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 執行無條件清倉 (Market Sell) 並轉入現金避險。")

    st.subheader(f"📊 {target_symbol} 近期價格走勢、警戒區間與加碼點")
    
    recent_df = df_clean.iloc[-500:]
    add_pts = recent_df[recent_df['Add_Signal']]

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Close'], name="收盤價", line=dict(color='skyblue', width=2)))
    fig_price.add_trace(go.Scatter(x=recent_df.index, y=recent_df['MA20'], name="20日均線", line=dict(color='yellow', width=1, dash='dot')))
    
    # 繪製加碼點買入標記 (藍色向上箭頭)
    if not add_pts.empty:
        fig_price.add_trace(go.Scatter(
            x=add_pts.index, y=add_pts['Close'],
            mode='markers',
            name='🔵 藍燈加碼訊號',
            marker=dict(symbol='triangle-up', size=12, color='cyan')
        ))

    fig_price.add_hline(y=ath_price, line_dash="dash", line_color="gray", annotation_text="歷史最高點 (ATH)")
    fig_price.add_hrect(y0=ath_price*b_wave_min, y1=ath_price*b_wave_max, fillcolor="orange", opacity=0.15, line_width=0, annotation_text="B 浪警戒區間")
    fig_price.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_price, use_container_width=True)

# =========================================================
# TAB 2: 歷史回測分析
# =========================================================
with tab2:
    st.header(f"📈 {target_symbol} 風控與加碼策略歷史回測模擬")
    st.caption("模擬規則：綠燈 100% 持倉；橘燈 50% 減碼；紅燈 100% 清倉；轉綠燈且站上 MA20 觸發藍燈加碼。")

    position = np.where(df_clean['Signal'] >= 3, 0.0, np.where(df_clean['Signal'] == 2, 0.5, 1.0))
    position_series = pd.Series(position, index=df_clean.index).shift(1).fillna(1.0)
    strategy_returns = df_clean['Returns'] * position_series
    
    cum_bh = (1 + df_clean['Returns'].fillna(0)).cumprod()
    cum_strat = (1 + strategy_returns.fillna(0)).cumprod()
    
    mdd_bh = ((cum_bh / cum_bh.cummax()) - 1).min()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("買入持有 (B&H) 累積報酬", f"{(cum_bh.iloc[-1]-1)*100:.1f}%")
    c2.metric("LBMS 避險策略 累積報酬", f"{(cum_strat.iloc[-1]-1)*100:.1f}%")
    c3.metric("B&H 最大回撤 (MDD)", f"{mdd_bh*100:.1f}%", delta_color="inverse")
    c4.metric("LBMS 策略最大回撤 (MDD)", f"{mdd_strat*100:.1f}%", f"改善 {abs(mdd_bh-mdd_strat)*1
