import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

st.title("00631L 全球情緒抄底交易策略回測")

# ==============================================================================
# 1. 數據下載與安全對齊（多重索引安全降維）
# ==============================================================================
st.write("正在從 Yahoo Finance 下載最新數據...")

# 下載台股與美股波動率指標
taiex = yf.download("^TWII", start="2018-01-01")
twn2x = yf.download("00631L.TW", start="2018-01-01")
vix = yf.download("^VIX", start="2018-01-01")   # S&P500 恐慌指數
vxn = yf.download("^VXN", start="2018-01-01")   # Nasdaq100 恐慌指數

# 建立整合 DataFrame
df = pd.DataFrame(index=taiex.index)

# 使用安全的 .loc 元組指定方式，並強制轉換為 float 序列，避免 str 形態誤判
df['TAIEX_Close'] = taiex.loc[:, ('Close', '^TWII')].astype(float)
df['ETF_Close'] = twn2x.loc[:, ('Close', '00631L.TW')].astype(float)

vix_series = vix.loc[:, ('Close', '^VIX')].astype(float).rename('VIX')
vxn_series = vxn.loc[:, ('Close', '^VXN')].astype(float).rename('VXN')

# 透過對齊日期合併
df = df.join(vix_series, how='left')
df = df.join(vxn_series, how='left')

# 雙向填補假日空值，確保資料連續性
df = df.ffill().bfill() 

# 計算技術指標
df['MA240'] = df['TAIEX_Close'].rolling(window=240).mean()
df = df.dropna() 

if df.empty:
    st.error("錯誤：資料表經對齊後變為空值，請檢查網路或 yfinance 套件狀態。")
    st.stop()

# ==============================================================================
# 2. 核心交易引擎（數值全數強制浮點數化）
# ==============================================================================
initial_capital = 1000000.0
cash = initial_capital
etf_shares = 0.0
portfolio_values = []
trade_log = []

fee_rate = 0.001425
tax_rate = 0.001
in_position = False

for i in range(len(df)):
    current_date = df.index[i]
    
    # 使用 float() 強制轉型，確保絕對不會發生 'str' object is not callable
    current_taiex = float(df['TAIEX_Close'].iloc[i])
    current_etf = float(df['ETF_Close'].iloc[i])
    current_ma240 = float(df['MA240'].iloc[i])
    
    c_vix = float(df['VIX'].iloc[i])
    c_vxn = float(df['VXN'].iloc[i])
    
    current_portfolio_value = (etf_shares * current_etf) + cash
    
    # 狀況 A：持有部位時（年線防守 或 貪婪出場）
    if in_position:
        if current_taiex < current_ma240:
            cash += etf_shares * current_etf * (1.0 - fee_rate - tax_rate)
            etf_shares = 0.0
            in_position = False
            trade_log.append(f"{current_date.strftime('%Y-%m-%d')} | 🚨 跌破年線，強制平倉避險！")
        else:
            if c_vix < 13.0 and c_vxn < 13.0:
                cash += etf_shares * current_etf * (1.0 - fee_rate - tax_rate)
                etf_shares = 0.0
                in_position = False
                trade_log.append(f"{current_date.strftime('%Y-%m-%d')} | 💰 市場極度貪婪 (VIX/VXN < 13)，全數獲利平倉！")
                
    # 狀況 B：空倉時（恐懼抄底）
    else:
        if current_taiex >= current_ma240:
            if c_vix > 28.0 or c_vxn > 32.0:
                buy_budget = current_portfolio_value * 0.50
                etf_shares = buy_budget / (current_etf * (1.0 + fee_rate))
                cash = current_portfolio_value - (etf_shares * current_etf)
                in_position = True
                trade_log.append(f"{current_date.strftime('%Y-%m-%d')} | 🎯 觸發極度恐懼抄底進場！")

    portfolio_values.append((etf_shares * current_etf) + cash)

# ==============================================================================
# 3. Streamlit 網頁圖表與報告輸出
# ==============================================================================
if len(portfolio_values) > 0:
    final_value = portfolio_values[-1]
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    
    st.subheader("📊 回測績效報告")
    st.metric(label="最終資產總值", value=f"${final_value:,.0f} 元")
    st.metric(label="策略累積總報酬率", value=f"{total_return:.2f} %")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, portfolio_values, label="Strategy Value", color='darkblue')
    ax.set_title("00631L Rebalance Strategy Curve")
    ax.grid(True, linestyle='--')
    st.pyplot(fig)
    
    st.subheader("📜 交易歷史日誌")
    for log in trade_log[::-1]:
        st.text(log)
else:
    st.warning("沒有產生任何資產數據。")
