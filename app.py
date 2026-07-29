import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st  
import os

# 基本網頁排版設定
st.set_page_config(layout="wide")
st.title("📈 00631L 終極本地數據策略儀表板 (含大盤對比、MDD、CAGR)")

# ==============================================================================
# 1. 互動式側邊欄控制面板 (Sidebar)
# ==============================================================================
st.sidebar.header("🛠️ 策略參數調整面板")

initial_capital = st.sidebar.number_input("初始投資本金 (TWD)", min_value=100000, max_value=10000000, value=1000000, step=100000)
fee_rate = st.sidebar.slider("券商手續費率 (%)", min_value=0.0, max_value=0.5, value=0.1425, step=0.01) / 100
tax_rate = st.sidebar.slider("槓桿 ETF 證交稅率 (%)", min_value=0.0, max_value=0.5, value=0.1, step=0.01) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 恐懼抄底閥值")
ndfi_buy_trigger = st.sidebar.slider("NDFI 跌破此數值 (極度恐懼)", min_value=10, max_value=30, value=15, step=1)
pcr_buy_trigger = st.sidebar.slider("5日 P/C Ratio 突破此數值 (避險極致)", min_value=0.9, max_value=1.3, value=1.05, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 貪婪出場閥值")
ndfi_sell_trigger = st.sidebar.slider("NDFI 突破此數值 (市場過熱)", min_value=70, max_value=90, value=80, step=1)
pcr_sell_trigger = st.sidebar.slider("5日 P/C Ratio 跌破此數值 (散戶瘋狂)", min_value=0.5, max_value=0.7, value=0.6, step=0.05)

# ==============================================================================
# 2. 安全讀取與嚴格檢查本地 CSV 檔案
# ==============================================================================
csv_filename = "data.csv"

if not os.path.exists(csv_filename):
    st.error(f"❌ 找不到數據檔案！請確認您的專案根目錄下已放置 `{csv_filename}`。")
    st.stop()

@st.cache_data
def load_and_validate_data():
    try:
        raw_df = pd.read_csv(csv_filename)
    except Exception as e:
        st.error(f"❌ 讀取 CSV 發生錯誤: {e}")
        return None
        
    raw_df.columns = raw_df.columns.str.strip()
    
    required_cols = ['Date', 'TAIEX', '00631L', 'NDFI', 'PCR_5MA']
    for col in required_cols:
        if col not in raw_df.columns:
            st.error(f"❌ 您的 CSV 缺少必要欄位: `{col}`。現有欄位為: {list(raw_df.columns)}")
            return None

    raw_df['Date'] = pd.to_datetime(raw_df['Date'], format='mixed', errors='coerce')
    raw_df = raw_df.dropna(subset=['Date'])
    raw_df = raw_df.sort_values('Date').set_index('Date')
    
    for col in ['TAIEX', '00631L', 'NDFI', 'PCR_5MA']:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
        
    raw_df = raw_df.ffill().bfill()
    
    raw_df['MA240'] = raw_df['TAIEX'].rolling(window=240).mean()
    raw_df['MA120'] = raw_df['TAIEX'].rolling(window=120).mean()
    
    return raw_df.dropna(subset=['TAIEX', '00631L'])

df = load_and_validate_data()

if df is None or df.empty:
    st.error("❌ 資料集載入失敗或內容為空，請檢查 CSV 內容。")
    st.stop()

total_days = (df.index[-1] - df.index[0]).days
years = total_days / 365.25 if total_days > 0 else 1.0

# ==============================================================================
# 3. 雙策略動態回測引擎 (修復每日資產滾動連動)
# ==============================================================================
def run_local_backtest(ma_column):
    cash = float(initial_capital)
    etf_shares = 0.0
    in_position = False
    portfolio_values = []
    logs = []
    
    dates = df.index
    taiex_vals = df['TAIEX'].values
    etf_vals = df['00631L'].values
    ma_vals = df[ma_column].values
    ndfi_vals = df['NDFI'].values
    pcr_vals = df['PCR_5MA'].values
    
    for i in range(len(df)):
        c_date = dates[i]
        c_taiex = float(taiex_vals[i])
        c_etf = float(etf_vals[i])
        c_ma = float(ma_vals[i]) if not np.isnan(ma_vals[i]) else c_taiex
        c_ndfi = float(ndfi_vals[i])
        c_pcr = float(pcr_vals[i])
        
        # 1. 如果持有部位，先檢查是否符合「出場」條件
        if in_position:
            if c_taiex < c_ma: # 跌破均線防守
                cash = (etf_shares * c_etf) * (1.0 - fee_rate - tax_rate)
                etf_shares = 0.0
                in_position = False
                logs.append(f"{c_date.strftime('%Y-%m-%d')} | 🚨 跌破 {ma_column} ({c_ma:.1f})，全數強制清倉避險！")
            elif c_ndfi > ndfi_sell_trigger or c_pcr < pcr_sell_trigger: # 貪婪過熱出場
                cash = (etf_shares * c_etf) * (1.0 - fee_rate - tax_rate)
                etf_shares = 0.0
                in_position = False
                logs.append(f"{c_date.strftime('%Y-%m-%d')} | 💰 市場轉為貪婪 (NDFI:{c_ndfi})，全數獲利平倉！")
                
        # 2. 如果是空手狀態，檢查是否符合「進場」條件
        else:
            if c_taiex >= c_ma:
                if c_ndfi < ndfi_buy_trigger and c_pcr > pcr_buy_trigger:
                    current_total = cash  
                    buy_budget = current_total * 0.50 # 50% 資金進場
                    etf_shares = buy_budget / (c_etf * (1.0 + fee_rate))
                    cash = current_total - buy_budget
                    in_position = True
                    logs.append(f"{c_date.strftime('%Y-%m-%d')} | 🎯 觸發極度恐懼抄底 (NDFI:{c_ndfi})，50% 資金以價格 {c_etf} 進場！")
                    
        # 3. 每日結算總資產 = 現金餘額 + 當前持股市值
        current_portfolio_value = cash + (etf_shares * c_etf)
        portfolio_values.append(current_portfolio_value)
        
    return portfolio_values, logs

# ==============================================================================
# 4. 績效指標計算函數
# ==============================================================================
def calculate_metrics(portfolio_values):
    p_series = pd.Series(portfolio_values)
    total_ret = ((p_series.iloc[-1] - initial_capital) / initial_capital) * 100
    cagr = (((p_series.iloc[-1] / initial_capital) ** (1 / years)) - 1) * 100 if years > 0 else 0.0
    peak = p_series.cummax()
    drawdown = (p_series - peak) / peak
    mdd = drawdown.min() * 100
    return total_ret, cagr, mdd

values_240, logs_240 = run_local_backtest('MA240')
values_120, logs_120 = run_local_backtest('MA120')

ret_240, cagr_240, mdd_240 = calculate_metrics(values_240)
ret_120, cagr_120, mdd_120 = calculate_metrics(values_120)

taiex_start = df['TAIEX'].iloc[0]
values_taiex = (df['TAIEX'] / taiex_start) * initial_capital
ret_taiex, cagr_taiex, mdd_taiex = calculate_metrics(values_taiex.values)

# ==============================================================================
# 5. Streamlit 儀表板前端呈現
# ==============================================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("🛡️ 240MA 年線策略")
    st.metric(label="最終資產總值", value=f"${values_240[-1]:,.0f} 元")
    st.metric(label="累積總報酬 / 年化 CAGR", value=f"{ret_240:.2f} %", delta=f"CAGR: {cagr_240:.2f} %")
    st.metric(label="歷年最大回撤 (MDD)", value=f"{mdd_240:.2f} %")

with col2:
    st.subheader("⚡ 120MA 半年線策略")
    st.metric(label="最終資產總值", value=f"${values_120[-1]:,.0f} 元")
    st.metric(label="累積總報酬 / 年化 CAGR", value=f"{ret_120:.2f} %", delta=f"CAGR: {cagr_120:.2f} %")
    st.metric(label="歷年最大回撤 (MDD)", value=f"{mdd_120:.2f} %")

with col3:
    st.subheader("🏛️ 大盤同期表現 (Buy & Hold)")
    st.metric(label="對比資產總值", value=f"${values_taiex.iloc[-1]:,.0f} 元")
    st.metric(label="累積總報酬 / 年化 CAGR", value=f"{ret_taiex:.2f} %", delta=f"CAGR: {cagr_taiex:.2f} %")
    st.metric(label="歷年最大回撤 (MDD)", value=f"{mdd_taiex:.2f} %")

st.markdown("---")
st.subheader("📈 00631L 策略與加權指數資產增長曲線對比圖")

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(df.index, values_240, label=f"240MA Strategy (MDD: {mdd_240:.1f}%)", color='darkblue', lw=2)
ax.plot(df.index, values_120, label=f"120MA Strategy (MDD: {mdd_120:.1f}%)", color='crimson', lw=1.5)
ax.plot(df.index, values_taiex, label=f"Benchmark TAIEX (MDD: {mdd_taiex:.1f}%)", color='gray', linestyle='--', alpha=0.7, lw=1.2)
ax.set_title("Strategy Performance vs TAIEX Benchmark Comparison", fontsize=12, fontweight='bold')
ax.set_xlabel("Timeline")
ax.set_ylabel("Portfolio Value (TWD)")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=10)
st.pyplot(fig)

st.markdown("---")
col_log1, col_log2 = st.columns(2)
with col_log1:
    st.subheader("📜 240MA 策略交易歷史")
    if logs_240:
        for log in logs_240[::-1]: st.text(log)
    else: st.caption("此配置下未觸發交易。")

with col_log2:
    st.subheader("📜 120MA 策略交易歷史")
    if logs_120:
        for log in logs_120[::-1]: st.text(log)
    else: st.caption("此配置下未觸發交易。")
