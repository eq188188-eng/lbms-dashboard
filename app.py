app.py
import streamlit as st
import yfinance as yf
import pandas as pd
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

st.title("🛡️ LBMS 自動化流動性與泡沫預警系統 (網頁版)")
st.caption("透過微觀結構、波動率及信用利差，客觀監測資產類高點與流動性風險。")

# ---------------------------------------------------------
# 2. 側邊欄控制項 (Parameters)
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
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

with st.spinner("正在讀取全球市場數據..."):
    df_target = load_data(target_symbol)
    df_hyg = load_data(hyg_symbol)
    df_tlt = load_data(tlt_symbol)

if df_target.empty:
    st.error(f"無法取得標的 '{target_symbol}' 數據，請確認代碼是否正確。")
    st.stop()

# --- 計算指標 ---
current_price = float(df_target['Close'].iloc[-1])
ath_price = float(df_target['High'].max())
ath_ratio = current_price / ath_price

# 波動率計算
df_target['Returns'] = df_target['Close'].pct_change()
df_target['Vol_20d'] = df_target['Returns'].rolling(window=20).std() * (252 ** 0.5)
current_vol = float(df_target['Vol_20d'].iloc[-1])
vol_threshold = float(df_target['Vol_20d'].quantile(vol_quantile))

# 信用利差 Proxy 計算 (HYG / TLT)
credit_ratio = df_hyg['Close'] / df_tlt['Close']
credit_mavg = credit_ratio.rolling(20).mean()
current_credit = float(credit_ratio.iloc[-1])
threshold_credit = float(credit_mavg.iloc[-1] * 0.97)

# ---------------------------------------------------------
# 4. 風控訊號判定邏輯
# ---------------------------------------------------------
triggers = []
if b_wave_min <= ath_ratio <= b_wave_max:
    triggers.append(f"進入類高點 (B浪) 危險區 (當前為 ATH 的 {ath_ratio*100:.1f}%)")

if current_vol > vol_threshold:
    triggers.append(f"波動率爆表 (當前 {current_vol*100:.1f}% > 門檻 {vol_threshold*100:.1f}%)")

if current_credit < threshold_credit:
    triggers.append("信用利差惡化 (高收益債相對強度弱於 20日均線 3%)")

trigger_count = len(triggers)

# ---------------------------------------------------------
# 5. Dashboard 燈號與核心指標展示
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("當前價格", f"${current_price:.2f}", f"ATH: ${ath_price:.2f}")
col2.metric("相對於 ATH 比例", f"{ath_ratio*100:.1f}%")
col3.metric("20日年化波動率", f"{current_vol*100:.1f}%", f"門檻: {vol_threshold*100:.1f}%", delta_color="inverse")
col4.metric("信用利差 Proxy", f"{current_credit:.2f}", f"門檻: {threshold_credit:.2f}", delta_color="normal")

st.divider()

# 狀態報告面板
if trigger_count == 0:
    st.success("🟢 **當前燈號：綠燈 (系統安全)**\n\n市場結構與流動性正常，可維持原配置或正常策略。")
elif trigger_count == 1:
    st.warning("🟡 **當前燈號：黃燈 (高度警戒)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 停止開槓桿，取消追高買單，設定移動止損。")
elif trigger_count == 2:
    st.error("🟠 **當前燈號：橘燈 (逃生區/減碼)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 進入微觀流動性風險區，建議現貨減碼 50%，清空槓桿部位。")
else:
    st.error("🔴 **當前燈號：紅燈 (極限離場)**\n\n**觸發項目：** " + "；".join(triggers) + "\n\n**建議動作：** 多重極限指標爆表，建議執行無條件清倉 (Market Sell) 並轉入現金避險。")

# ---------------------------------------------------------
# 6. 互動圖表區 (Plotly)
# ---------------------------------------------------------
st.subheader("📊 價格走勢與類高點 (B 浪) 警戒區間")

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df_target.index, y=df_target['Close'], name="收盤價", line=dict(color='skyblue', width=2)))
fig_price.add_hline(y=ath_price, line_dash="dash", line_color="gray", annotation_text="歷史最高點 (ATH)")
fig_price.add_hrect(y0=ath_price*b_wave_min, y1=ath_price*b_wave_max, fillcolor="orange", opacity=0.15, line_width=0, annotation_text="B 浪警戒區間")

fig_price.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig_price, use_container_width=True)

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("📈 20日歷史波動率 (VaR 監測)")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=df_target.index, y=df_target['Vol_20d']*100, name="波動率 (%)", line=dict(color='magenta')))
    fig_vol.add_hline(y=vol_threshold*100, line_dash="dash", line_color="red", annotation_text="VaR 警戒線")
    fig_vol.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_vol, use_container_width=True)

with col_chart2:
    st.subheader("💳 高收益債信用相對強度 (HYG/TLT)")
    fig_credit = go.Figure()
    fig_credit.add_trace(go.Scatter(x=credit_ratio.index, y=credit_ratio, name="HYG/TLT 比率", line=dict(color='yellow')))
    fig_credit.add_trace(go.Scatter(x=credit_mavg.index, y=credit_mavg*0.97, name="風控警戒線", line=dict(color='red', dash='dash')))
    fig_credit.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_credit, use_container_width=True)
