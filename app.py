import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 數據下載與準備
# ==============================================================================
print("正在下載歷史數據並建構『恐懼進、貪婪出』策略...")
taiex = yf.download("^TWII", start="2018-01-01")
twn2x = yf.download("00631L.TW", start="2018-01-01")
vix = yf.download("^VIX", start="2018-01-01")

# 建立整合 DataFrame (實務上 NDFI 與 PCR 可透過其他免費金融 API 或數據商接入)
df = pd.DataFrame(index=taiex.index)
df['TAIEX_Close'] = taiex['Close']
df['ETF_Close'] = twn2x['Close']
df['VIX'] = vix['Close']
# 模擬歷史極值數據用於邏輯測試 (實務對接後直接讀取真實欄位)
df['NDFI'] = 50 
df['PCR_Daily'] = 0.7

# 計算核心指標
df['MA240'] = df['TAIEX_Close'].rolling(window=240).mean()
df['PCR_5MA'] = df['PCR_Daily'].rolling(window=5).mean()
df = df.dropna()

# ==============================================================================
# 2. 核心交易引擎（恐懼進場 ➔ 貪婪出場）
# ==============================================================================
initial_capital = 1000000.0  # 初始本金 100 萬台幣
cash = initial_capital
etf_shares = 0
portfolio_values = []
trade_log = []

# 手續費與證交稅
fee_rate = 0.001425
tax_rate = 0.001

in_position = False

for i in range(len(df)):
    current_date = df.index[i]
    current_taiex = df['TAIEX_Close'].iloc[i]
    current_etf = df['ETF_Close'].iloc[i]
    current_ma240 = df['MA240'].iloc[i]
    
    c_vix = df['VIX'].iloc[i]
    c_ndfi = df['NDFI'].iloc[i]
    c_pcr5m = df['PCR_5MA'].iloc[i]
    
    current_portfolio_value = (etf_shares * current_etf) + cash
    
    # --------------------------------------------------------------------------
    # 狀況 A：持有部位時的判斷（檢查是否觸發『絕對防守』或『貪婪出場』）
    # --------------------------------------------------------------------------
    if in_position:
        # 1. 【防禦機制第一優先】：大盤不幸跌破年線，無條件清倉避險
        if current_taiex < current_ma240:
            cash += etf_shares * current_etf * (1 - fee_rate - tax_rate)
            etf_shares = 0
            in_position = False
            trade_log.append(f"{current_date.strftime('%Y-%m-%d')} | 🚨 跌破年線，強制平倉避險！")
            
        # 2. 【正常獲利出場】：市場進入貪婪區間
        else:
            greed_signals = 0
            if c_vix < 13: greed_signals += 1
            if c_ndfi > 80: greed_signals += 1
            if c_pcr5m < 0.60: greed_signals += 1
            
            if greed_signals >= 2: # 滿足任兩個貪婪指標，獲利入袋
                cash += etf_shares * current_etf * (1 - fee_rate - tax_rate)
                etf_shares = 0
                in_position = False
                trade_log.append(f"{current_date.strftime('%Y-%m-%d')} | 💰 市場極度貪婪，全數獲利平倉！")
                
    # --------------------------------------------------------------------------
    # 狀況 B：空倉時的判斷（檢查是否觸發『恐懼抄底』）
    # --------------------------------------------------------------------------
    else:
        # 必須在大盤年線之上（非系統性崩盤），且市場極度恐懼
        if current_taiex >= current_ma240:
            fear_signals = 0
            if c_vix > 30: fear_signals += 1
            if c_ndfi < 15: fear_signals += 1
            if c_pcr5m > 1.05: fear_signals += 1
            
            if fear_signals >= 2: # 滿足任兩個恐懼指標，50%資金折半進場
                buy_budget = current_portfolio_value * 0.50
                etf_shares = buy_budget / (current_etf * (1 + fee_rate))
                cash = current_portfolio_value - (etf_shares * current_etf)
                in_position = True
                trade_log.append(f"{current_date.strftime('%Y-%m-%d')} | 🎯 觸發極度恐懼抄底進場！")

    # 紀錄每日資產價值
    portfolio_values.append((etf_shares * current_etf) + cash)

# ==============================================================================
# 3. 輸出回測報告
# ==============================================================================
final_value = portfolio_values[-1]
print("\n" + "="*40 + "\n【『恐懼進、貪婪出』策略日誌摘要】\n" + "="*40)
for log in trade_log[:10]: # 顯示前10筆交易紀錄
    print(log)
print("...")
print(f"最終資產總值：${final_value:,.0f} 元")
print(f"策略累積總報酬率：{((final_value - initial_capital)/initial_capital)*100:.2f} %")
print("="*40)
