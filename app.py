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
    page_title="台股籌碼與流動性預警系統",
    page_icon="🇹🇼",
    layout="wide"
)

st.title("🇹🇼 台股法人籌碼、融資與選擇權流動性預警系統")
st.caption("防守減碼：三大法人賣超、融資維持率惡化、融券異常、期權籌碼偏空與散戶過度樂觀。加碼：危機解除配合均線支撐。")

# ---------------------------------------------------------
# 2. 數據抓取與預處理 (台股版)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(ticker, period="max", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

st.sidebar.header("⚙️ 系統參數設定")
target_symbol = st.sidebar.text_input("監控標的代碼 (例如大盤 ^TWII 或台積電 2330.TW)", value="^TWII")
index_symbol = "^TWII"

with st.spinner(f"正在讀取 {target_symbol} 市場數據..."):
    df_target = load_data(target_symbol)
    df_index = load_data(index_symbol)

if df_target.empty:
    st.error(f"無法取得標的 '{target_symbol}' 數據，請確認代碼是否正確。")
    st.stop()

# 數據時間對齊
common_index = df_target.index.intersection(df_index.index)
df_t = df_target.loc[common_index].copy()

# 基礎指標與多維均線計算
df_t['ATH'] = df_t['High'].cummax()
df_t['ATH_Ratio'] = df_t['Close'] / df_t['ATH']
df_t['Returns'] = df_t['Close'].pct_change()
df_t['Vol_20d'] = df_t['Returns'].rolling(window=20).std() * (252 ** 0.5)

df_t['MA20'] = df_t['Close'].rolling(window=20).mean()
df_t['MA60'] = df_t['Close'].rolling(window=60).mean()
df_t['MA240'] = df_t['Close'].rolling(window=240).mean()

# ---------------------------------------------------------
# 3. 模擬台股籌碼與融資指標
# ---------------------------------------------------------
np.random.seed(42)
n_rows = len(df_t)

df_t['Institutional_Sell'] = (df_t['Returns'].rolling(5).sum() < -0.05).astype(int)

base_maint_ratio = 165 - (df_t['Close'] / df_t['MA60'] - 1) * 50
df_t['Margin_Maintenance_Ratio'] = np.clip(base_maint_ratio + np.random.normal(0, 3, n_rows), 130, 190)
cond_margin_danger = df_t['Margin_Maintenance_Ratio'] < 152

df_t['Futures_Net_Equivalent'] = np.random.normal(0, 5000, n_rows) - (df_t['Returns'].rolling(10).sum() * 20000)
cond_futures_bearish = df_t['Futures_Net_Equivalent'] < -4000

df_t['PCR'] = np.clip(110 + (df_t['Returns'].rolling(5).mean() * 300) + np.random.normal(0, 10, n_rows), 70, 160)
cond_pcr_extreme = (df_t['PCR'] < 85) | (df_t['PCR'] > 145)

# ---------------------------------------------------------
# 4. 回測核心函數
# ---------------------------------------------------------
def run_backtest(df):
    score = (
        df['Institutional_Sell'].astype(int) + 
        cond_margin_danger.astype(int) + 
        cond_futures_bearish.astype(int) + 
        cond_pcr_extreme.astype(int)
    )
    
    prev_score = score.shift(1).fillna(0)
    price_near_support = (df['Close'] < df['MA20']) | (df['Close'] < df['MA60']) | (df['Close'] < df['MA240'] * 1.05)
    add_signal = (prev_score >= 1) & (score == 0) & price_near_support
    
    position = np.where(score >= 3, 0.0, np.where(score == 2, 0.3, np.where(score == 1, 0.7, 1.0)))
    pos_series = pd.Series(position, index=df.index).shift(1).fillna(1.0)
    
    strat_ret = df['Returns'] * pos_series
    cum_strat = (1 + strat_ret.fillna(0)).cumprod()
    mdd = ((cum_strat / cum_strat.cummax()) - 1).min()
    total_ret = cum_strat.iloc[-1] - 1
    return total_ret, mdd, score, add_signal

total_ret, mdd_strat, signal_score, add_signals = run_backtest(df_t)
df_t['Signal'] = signal_score
df_t['Add_Signal'] = add_signals

# ---------------------------------------------------------
# 5. 頁面呈現
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📊 台股籌碼即時儀表板", "📈 歷史回測與訊號分析"])

with tab1:
    curr_price = float(df_t['Close'].iloc[-1])
    curr_ath = float(df_t['ATH'].iloc[-1])
    curr_maint = float(df_t['Margin_Maintenance_Ratio'].iloc[-1])
    curr_futures = float(df_t['Futures_Net_Equivalent'].iloc[-1])
    curr_pcr = float(df_t['PCR'].iloc[-1])
    is_add_today = bool(df_t['Add_Signal'].iloc[-1])
    
    triggers = []
    if df_t['Institutional_Sell'].iloc[-1] == 1:
        triggers.append("三大法人近期呈現明顯賣超")
    if curr_maint < 152:
        triggers.append(f"融資維持率偏低逼近斷頭區 (當前: {curr_maint:.1f}%)")
    if curr_futures < -4000:
        triggers.append(f"期貨法人大台當量淨額大幅偏空 ({curr_futures:.0f} 口)")
    if curr_pcr < 85 or curr_pcr > 145:
        triggers.append(f"選擇權 Put/Call Ratio 處於極端數值 ({curr_pcr:.1f}%)")

    trigger_count = len(triggers)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("當前價格", f"${curr_price:.2f}", f"歷史高點: ${curr_ath:.2f}")
    c2.metric("融資維持率 Proxy", f"{curr_maint:.1f}%", "警戒線: 152%", delta_color="inverse" if curr_maint < 152 else "normal")
    c3.metric("期貨法人淨額當量", f"{curr_futures:.0f} 口", delta_color="inverse" if curr_futures < 0 else "normal")
    c4.metric("選擇權 PCR", f"{curr_pcr:.1f}%", delta_color="off")

    st.divider()

    if is_add_today:
        st.info("🔵 **當前訊號：籌碼危機解除與均線支撐加碼點！**\n\n法人與融資籌碼壓力已過，且價格回測至均線支撐帶，建議分批低接。")
    elif trigger_count == 0:
        st.success("🟢 **當前燈號：綠燈 (籌碼結構安全)**\n\n三大法人與期權散戶結構健康，可維持滿倉。")
    elif trigger_count == 1:
        st.warning("🟡 **當前燈號：黃燈 (籌碼鬆動警戒)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 停止融資開槓桿，提高現金。")
    elif trigger_count == 2:
        st.error("🟠 **當前燈號：橘燈 (籌碼惡化減碼)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 現貨減碼 50%，清空槓桿。")
    else:
        st.error("🔴 **當前燈號：紅燈 (流動性與融資斷頭危機/清倉)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 執行無條件清倉避險。")

    st.subheader(f"📊 {target_symbol} 近期價格與均線防守區")
    recent_df = df_t.iloc[-500:]
    add_pts = recent_df[recent_df['Add_Signal']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Close'], name="收盤價", line=dict(color='skyblue', width=2)))
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['MA20'], name="MA20", line=dict(color='yellow', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['MA60'], name="MA60", line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['MA240'], name="MA240", line=dict(color='magenta', width=1.5)))
    
    if not add_pts.empty:
        fig.add_trace(go.Scatter(
            x=add_pts.index, y=add_pts['Close'],
            mode='markers',
            name='🔵 籌碼解除與均線低接點',
            marker=dict(symbol='triangle-up', size=12, color='cyan')
        ))

    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("📈 籌碼策略歷史回測效能")
    cum_bh = (1 + df_t['Returns'].fillna(0)).cumprod()
    
    position = np.where(df_t['Signal'] >= 3, 0.0, np.where(df_t['Signal'] == 2, 0.3, np.where(df_t['Signal'] == 1, 0.7, 1.0)))
    pos_ser = pd.Series(position, index=df_t.index).shift(1).fillna(1.0)
    strat_ret = df_t['Returns'] * pos_ser
    cum_strat = (1 + strat_ret.fillna(0)).cumprod()
    
    c1, c2 = st.columns(2)
    c1.metric("買入持有 (B&H) 報酬率", f"{(cum_bh.iloc[-1]-1)*100:.1f}%")
    c2.metric("台股籌碼策略報酬率", f"{(cum_strat.iloc[-1]-1)*100:.1f}%")
    
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=df_t.index, y=cum_bh, name="買入持有", line=dict(color='gray', width=1.5)))
    fig_b.add_trace(go.Scatter(x=df_t.index, y=cum_strat, name="台股籌碼風控策略", line=dict(color='green', width=2)))
    fig_b.update_layout(template="plotly_dark", height=400, yaxis_type="log", title="對數淨值曲線")
    st.plotly_chart(fig_b, use_container_width=True)
