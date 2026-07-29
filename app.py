import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")
st.title("📈 00631L 全球情緒抄底與雙均線防守策略大師")

# ==============================================================================
# 1. 互動式側邊欄控制面板 (Sidebar)
# ==============================================================================
st.sidebar.header("🛠️ 策略參數調整面板")

# 資金與成本設定
initial_capital = st.sidebar.number_input("初始投資本金 (TWD)", min_value=100000, max_value=10000000, value=1000000, step=100000)
fee_rate = st.sidebar.slider("券商手續費率 (%)", min_value=0.0, max_value=0.5, value=0.1425, step=0.01) / 100
tax_rate = st.sidebar.slider("槓桿 ETF 證交稅率 (%)", min_value=0.0, max_value=0.5, value=0.1, step=0.01) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 恐懼抄底閥值 (美股指標)")
vix_buy_trigger = st.sidebar.slider("S&P500 恐慌指數 (VIX) 突破值", min_value=20, max_value=45, value=28, step=1)
vxn_buy_trigger = st.sidebar.slider("Nasdaq100 恐慌指數 (VXN) 突破值", min_value=25, max_value=50, value=32, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 貪婪出場閥值")
vix_sell_trigger = st.sidebar.slider("VIX 跌破此數值 (市場過熱)", min_value=10, max_value=18, value=13, step=1)
vxn_sell_trigger = st.sidebar.slider("VXN 跌破此數值 (市場過熱)", min_value=10, max_value=18, value=13, step=1)

# ==============================================================================
# 2. 資料安全下載與「全降維」拍平處理 (防噴錯關鍵)
# ==============================================================================
st.write("🔄 正在從 Yahoo Finance 同步全球最新市場數據...")

@st.cache_data
def load_and_clean_data():
    taiex_raw = yf.download("^TWII", start="2018-01-01")
    twn2x_raw = yf.download("00631L.TW", start="2018-01-01")
    vix_raw = yf.download("^VIX", start="2018-01-01")
    vxn_raw = yf.download("^VXN", start="2018-01-01")
    
    # 徹底攤平 MultiIndex 矩陣，轉為純粹一維 Series
    s_taiex = pd.Series(taiex_raw['Close'].values.flatten(), index=taiex_raw.index, name='TAIEX_Close').astype(float)
    s_etf = pd.Series(twn2x_raw['Close'].values.flatten(), index=twn2x_raw.index, name='ETF_Close').astype(float)
    s_vix = pd.Series(vix_raw['Close'].values.flatten(), index=vix_raw.index, name='VIX').astype(float)
    s_vxn = pd.Series(vxn_raw['Close'].values.flatten(), index=vxn_raw.index, name='VXN').astype(float)
    
    # 合併並填補假日
    raw_df = pd.DataFrame(index=s_taiex.index)
    raw_df['TAIEX_Close'] = s_taiex
    raw_df['ETF_Close'] = s_etf
    raw_df = raw_df.join(s_vix, how='left').join(s_vxn, how='left')
    raw_df = raw_df.ffill().bfill()
    
    # 事先計算好雙均線
    raw_df['MA240'] = raw_df['TAIEX_Close'].rolling(window=240).mean()
    raw_df['MA120'] = raw_df['TAIEX_Close'].rolling(window=120).mean()
    return raw_df.dropna()

df = load_and_clean_data()

# ==============================================================================
# 3. 雙策略動態回測引擎
# ==============================================================================
def run_advanced_backtest(ma_column):
    cash = float(initial_capital)
    etf_shares = 0.0
    in_position = False
    portfolio_values = []
    logs = []
    
    # 轉換成 numpy 陣列加快速度且防錯
    dates = df.index
    taiex_vals = df['TAIEX_Close'].values
    etf_vals = df['ETF_Close'].values
    ma_vals = df[ma_column].values
    vix_vals = df['VIX'].values
    vxn_vals = df['VXN'].values
    
    for i in range(len(df)):
        c_date = dates[i]
        c_taiex = float(taiex_vals[i])
        c_etf = float(etf_vals[i])
        c_ma = float(ma_vals[i])
        c_vix = float(vix_vals[i])
        c_vxn = float(vxn_vals[i])
        
        current_portfolio_value = (etf_shares * c_etf) + cash
        
        # 狀況 A：持有部位時（均線防守 或 貪婪平倉）
        if in_position:
            if c_taiex < c_ma: # 跌破防守均線
                cash += etf_shares * c_etf * (1.0 - fee_rate - tax_rate)
                etf_shares = 0.0
                in_position = False
                logs.append(f"{c_date.strftime('%Y-%m-%d')} | 🚨 跌破 {ma_column}，全數強制清倉避險！")
            elif c_vix < vix_sell_trigger and c_vxn < vxn_sell_trigger: # 市場回到過熱貪婪區
                cash += etf_shares * c_etf * (1.0 - fee_rate - tax_rate)
                etf_shares = 0.0
                in_position = False
                logs.append(f"{c_date.strftime('%Y-%m-%d')} | 💰 市場極度貪婪 (VIX/VXN 過低)，全數獲利平倉！")
                
        # 狀況 B：空倉時（均線之上 + 恐懼抄底）
        else:
            if c_taiex >= c_ma:
                if c_vix > vix_buy_trigger or c_vxn > vxn_buy_trigger:
                    buy_budget = current_portfolio_value * 0.50 # 50% 折半配置
                    etf_shares = buy_budget / (c_etf * (1.0 + fee_rate))
                    cash = current_portfolio_value - (etf_shares * c_etf)
                    in_position = True
                    logs.append(f"{c_date.strftime('%Y-%m-%d')} | 🎯 觸發情緒極度恐懼，50% 資金抄底進場！")
                    
        portfolio_values.append((etf_shares * c_etf) + cash)
        
    return portfolio_values, logs

# 執行回測
values_240, logs_240 = run_advanced_backtest('MA240')
values_120, logs_120 = run_advanced_backtest('MA120')

# ==============================================================================
# 4. Streamlit 前端排版與視覺化儀表板
# ==============================================================================
ret_240 = ((values_240[-1] - initial_capital) / initial_capital) * 100
ret_120 = ((values_120[-1] - initial_capital) / initial_capital) * 100

col1, col2 = st.columns(2)
with col1:
    st.metric(label="🛡️ 240MA 年線策略最終資產", value=f"${values_240[-1]:,.0f} 元", delta=f"{ret_240:.2f} %")
with col2:
    st.metric(label="⚡ 120MA 半年線策略最終資產", value=f"${values_120[-1]:,.0f} 元", delta=f"{ret_120:.2f} %")

st.markdown("---")
st.subheader("📈 雙策略資產增長曲線對比圖")

# 繪製主圖表
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(df.index, values_240, label=f"240MA (Yearly) Strategy", color='darkblue', lw=2)
ax.plot(df.index, values_120, label=f"120MA (Half-Year) Strategy", color='crimson', lw=1.5)
ax.set_title("00631L Global Emotion Rebalance Performance Comparison", fontsize=12, fontweight='bold')
ax.set_xlabel("Timeline")
ax.set_ylabel("Portfolio Value (TWD)")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=10)
st.pyplot(fig)

st.markdown("---")
# 雙交易日誌分欄呈現
col_log1, col_log2 = st.columns(2)
with col_log1:
    st.subheader("📜 240MA 策略交易歷史")
    if logs_240:
        for log in logs_240[::-1]:
            st.text(log)
    else:
        st.caption("此參數配置下在回測期間未觸發任何交易。")

with col_log2:
    st.subheader("📜 120MA 策略交易歷史")
    if logs_120:
        for log in logs_120[::-1]:
            st.text(log)
    else:
        st.caption("此參數配置下在回測期間未觸發任何交易。")
