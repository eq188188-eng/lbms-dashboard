import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

st.title("00631L 全球情緒抄底交易策略回測")

# ==============================================================================
# 1. 數據下載與安全對齊（修正 yfinance 多重索引問題）
# ==============================================================================
st.write("正在從 Yahoo Finance 下載最新數據...")

# 下載台股與美股波動率指標
taiex = yf.download("^TWII", start="2018-01-01")
twn2x = yf.download("00631L.TW", start="2018-01-01")
vix = yf.download("^VIX", start="2018-01-01")   # S&P500 恐慌指數
vxn = yf.download("^VXN", start="2018-01-01")   # Nasdaq100 恐慌指數

# 建立整合 DataFrame
df = pd.DataFrame(index=taiex.index)

# 【關鍵修正】：使用 .loc[:, ('Close', TICKER)] 精確指定多重索引欄位，並轉換為單一 Series
df['TAIEX_Close'] = taiex.loc[:, ('Close', '^TWII')]
df['ETF_Close'] = twn2x.loc[:, ('Close', '00631L.TW')]

# 提取美股 VIX 與 VXN 的收盤價 Series
vix_series = vix.loc[:, ('Close', '^VIX')].rename('VIX')
vxn_series = vxn.loc[:, ('Close', '^VXN')].rename('VXN')

# 透過對齊日期，將美股數值塞入（處理台美交易日不一致問題）
df = df.join(vix_series, how='left')
df = df.join(vxn_series, how='left')

# 先用前面的數值填補假日的空值(NaN)，再雙向填補，確保 df 不會變空！
df = df.ffill().bfill() 

# 計算技術指標
df['MA240'] = df['TAIEX_Close'].rolling(window=240).mean()
df = df.dropna() # 這時只會切掉最前面無法計算 240 MA 的天數

# 防錯機制：檢查資料是否真的存在
if df.empty:
    st.error("錯誤：資料表經對齊後變為空值，請檢查網路或 yfinance 套件狀態。")
    st.stop()
