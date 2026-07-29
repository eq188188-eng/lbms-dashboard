import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os

st.set_page_config(layout="wide", page_title="00631L 策略儀表板")
st.title("📊 00631L 本地數據版 (NDFI + Put/Call Ratio) 雙策略儀表板")

# ==============================================================================
# 1. 側邊欄控制面板
# ==============================================================================
st.sidebar.header("🛠️ 策略參數調整面板")

initial_capital = st.sidebar.number_input("初始投資本金 (TWD)", min_value=100000, max_value=10000000, value=1000000, step=100000)
fee_rate = st.sidebar.slider("券商手續費率 (%)", min_value=0.0, max_value=0.5, value=0.1425, step=0.01) / 100
tax_rate = st.sidebar.slider("槓桿 ETF 證交稅率 (%)", min_value=0.0, max_value=0.5, value=0.1, step=0.01) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 恐懼抄底閥值")
ndfi_buy_trigger = st.sidebar.slider("NDFI 跌破此數值 (極度恐懼)", min_value=10, max_value=30, value=15, step=1)
pcr_buy_trigger = st.sidebar.slider("5日 P/C Ratio 突破此數值", min_value=0.9, max_value=1.3, value=1.05, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 貪婪出場閥值")
ndfi_sell_trigger = st.sidebar.slider("NDFI 突破此數值 (市場過熱)", min_value=70, max_value=90, value=80, step=1)
pcr_sell_trigger = st.sidebar.slider("5日 P/C Ratio 跌破此數值", min_value=0.5, max_value=0.7, value=0.6, step=0.05)

# ==============================================================================
# 2. 安全讀取本地 CSV 檔案（直接讀取，不使用 cache 避開型態衝突）
# ==============================================================================
csv_filename = "data.csv"

if not os.path.exists(csv_filename):
    st.error(f"❌ 找不到數據檔案！請確認專案根目錄下已放置 `{csv_filename}`。")
    st.info("💡 您的 CSV 欄位名稱應包含: `Date`, `TAIEX`, `00631L`, `NDFI`, `PCR_5MA`")
    st.stop()

try:
    df = pd.read_csv(csv_filename)
    df.columns = df.columns.str.strip()  # 清除欄位空白
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    
    for col in ['TAIEX', '00631L', 'NDFI', 'PCR_5MA']:
        df[col] = df[col].astype(float)
        
    df = df.ffill().bfill()
    df['MA240'] = df['TAIEX'].rolling(window=240).mean()
    df['MA120'] = df['TAIEX'].rolling(window=120).mean()
    df = df.dropna()
except Exception as e:
    st.error(f"❌ 讀取或處理 CSV 發生例外錯誤：{e}")
    st.stop()

# ==============================================================================
# 3. 雙策略回測引擎
# ==============================================================================
def run_backtest(ma_col_name):
    cash = float(initial_capital)
    shares = 0.0
    in_pos = False
    portfolio_vals = []
    hist_logs = []
    
    dates = df.index
    t_vals = df['TAIEX'].values
    e_vals = df['00631L'].values
    m_vals = df[ma_col_name].values
    n_vals = df['NDFI'].values
    p_vals = df['PCR_5MA'].values
    
    for i in range(len(df)):
        d = dates[i]
        t = float(t_vals[i])
        e = float(e_vals[i])
        m = float(m_vals[i])
        n = float(n_vals[i])
        p = float(p_vals[i])
        
        cur_val = (shares * e) + cash
        
        if in_pos:
            if t < m:
                cash += shares * e * (1.0 - fee_rate - tax_rate)
                shares = 0.0
                in_pos = False
                hist_logs.append(f"{d.strftime('%Y-%m-%d')} | 🚨 跌破 {ma_col_name}，全數清倉避險！")
            elif n > ndfi_sell_trigger or p < pcr_sell_trigger:
                cash += shares * e * (1.0 - fee_rate - tax_rate)
                shares = 0.0
                in_pos = False
                hist_logs.append(f"{d.strftime('%Y-%m-%d')} | 💰 市場轉為過熱，全數平倉！")
        else:
            if t >= m:
                if n < ndfi_buy_trigger and p > pcr_buy_trigger:
                    budget = cur_val * 0.50
                    shares = budget / (e * (1.0 + fee_rate))
                    cash = cur_val - (shares * e)
                    in_pos = True
                    hist_logs.append(f"{d.strftime('%Y-%m-%d')} | 🎯 觸發極度恐懼抄底，50% 進場！")
                    
        portfolio_vals.append((shares * e) + cash)
        
    return portfolio_vals, hist_logs

values_240, logs_240 = run_backtest('MA240')
values_120, logs_120 = run_backtest('MA120')

# ==============================================================================
# 4. 畫面渲染
# ==============================================================================
ret_240 = ((values_240[-1] - initial_capital) / initial_capital) * 100
ret_120 = ((values_120[-1] - initial_capital) / initial_capital) * 100

col1, col2 = st.columns(2)
with col1:
    st.metric("🛡️ 240MA 策略最終資產", f"${values_240[-1]:,.0f} 元", f"{ret_240:.2f} %")
with col2:
    st.metric("⚡ 120MA 策略最終資產", f"${values_120[-1]:,.0f} 元", f"{ret_120:.2f} %")

st.markdown("---")
st.subheader("📈 策略資產增長曲線對比")

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(df.index, values_240, label="240MA Strategy", color='darkblue', lw=2)
ax.plot(df.index, values_120, label="120MA Strategy", color='crimson', lw=1.5)
ax.set_title("Performance Comparison", fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
st.pyplot(fig)

st.markdown("---")
col_log1, col_log2 = st.columns(2)
with col_log1:
    st.subheader("📜 240MA 策略交易紀錄")
    if logs_240:
        for log_item in logs_240[::-1]:
            st.text(log_item)
    else:
        st.caption("此配置下無交易觸發。")

with col_log2:
    st.subheader("📜 120MA 策略交易紀錄")
    if logs_120:
        for log_item in logs_120[::-1]:
            st.text(log_item)
    else:
        st.caption("此配置下無交易觸發。")
