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
# 3. 模擬台股籌碼與融資指標 (串接真實 API 前的架構模擬)
# 註：實戰中可接入證交所/期交所 API 或 CMoney/fugle 數據
# ---------------------------------------------------------
np.random.seed(42)
n_rows = len(df_t)

# 模擬三大法人賣超狀態 (當大盤重挫或波動大時法人傾向賣超)
df_t['Institutional_Sell'] = (df_t['Returns'].rolling(5).sum() < -0.05).astype(int)

# 模擬融資維持率 (當指數跌破MA60時融資維持率下降逼近150%警戒線)
base_maint_ratio = 165 - (df_t['Close'] / df_t['MA60'] - 1) * 50
df_t['Margin_Maintenance_Ratio'] = np.clip(base_maint_ratio + np.random.normal(0, 3, n_rows), 130, 190)
cond_margin_danger = df_t['Margin_Maintenance_Ratio'] < 152  # 融資斷頭警戒

# 模擬期貨大台當量淨額 (大台 + 小台/4 + 微台/20)
df_t['Futures_Net_Equivalent'] = np.random.normal(0, 5000, n_rows) - (df_t['Returns'].rolling(10).sum() * 20000)
cond_futures_bearish = df_t['Futures_Net_Equivalent'] < -4000

# 模擬選擇權 Put/Call Ratio 與 散戶多空比 (散戶在大跌時短線過度看空或過度歐奈爾式融資抄底)
df_t['PCR'] = np.clip(110 + (df_t['Returns'].rolling(5).mean() * 300) + np.random.normal(0, 10, n_rows), 70, 160)
cond_pcr_extreme = (df_t['PCR'] < 85) | (df_t['PCR'] > 145)

# ---------------------------------------------------------
# 4. 回測核心函數 (結合籌碼與均線防守)
# ---------------------------------------------------------
def run_backtest(df):
    # 風險評分（滿分 4 分）：法人賣超、融資維持率危機、期貨淨額偏空、選擇權PCR極端
    score = (
        df['Institutional_Sell'].astype(int) + 
        cond_margin_danger.astype(int) + 
        cond_futures_bearish.astype(int) + 
        cond_pcr_extreme.astype(int)
    )
    
    prev_score = score.shift(1).fillna(0)
    
    # 加碼條件：危機解除 (score == 0)，且價格回測至三條均線（MA20/60/240）支撐區
    price_near_support = (df['Close'] < df['MA20']) | (df['Close'] < df['MA60']) | (df['Close'] < df['MA240'] * 1.05)
    add_signal = (prev_score >= 1) & (score == 0) & price_near_support
    
    # 動態倉位控制 (0分滿倉, 1分7成, 2分3成, 3分以上清倉)
    position = np.where(score >= 3, 0.0, np.where(score == 2, 0.3, np.where(score == 1, 0.7, 1.0)))
    pos_series = pd.Series(position, index=df.index).shift
