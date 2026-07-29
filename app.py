import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 0. 基本網頁排版設定
# ==========================================
st.set_page_config(
    page_title="00631L 終極本地數據策略儀表板", layout="wide"
)
st.title("📊 00631L 終極本地數據策略儀表板 (含大盤對比、MDD、CAGR)")

# ==========================================
# 1. 互動式側邊欄控制面板 (Sidebar)
# ==========================================
st.sidebar.header("⚙️ 策略參數調整面板")

initial_capital = st.sidebar.number_input(
    "初始投資本金 (TWD)",
    min_value=100000,
    max_value=10000000,
    value=1000000,
    step=100000,
)
fee_rate = st.sidebar.slider(
    "券商手續費 (%)", min_value=0.0, max_value=0.5, value=0.1425, step=0.01
)
tax_rate = st.sidebar.slider(
    "證券交易稅 (%)", min_value=0.0, max_value=0.5, value=0.1, step=0.01
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 抄底與風控參數")
ndfi_buy_trigger = st.sidebar.slider(
    "NDFI 抄底門檻 (極度恐慌)", min_value=0, max_value=100, value=20, step=1
)
pcr_buy_trigger = st.sidebar.slider(
    "PCR_5MA 抄底門檻", min_value=0.5, max_value=2.0, value=0.75, step=0.05
)
ndfi_sell_trigger = st.sidebar.slider(
    "NDFI 貪婪出場門檻", min_value=0, max_value=100, value=80, step=1
)
pcr_sell_trigger = st.sidebar.slider(
    "PCR_5MA 貪婪出場門檻", min_value=0.5, max_value=2.0, value=1.30, step=0.05
)

csv_filename = st.sidebar.text_input(
    "本地數據檔名 (CSV)", value="data.csv"
)


# ==========================================
# 2. 資料載入與防呆機制
# ==========================================
@st.cache_data
def load_and_validate_data(filename):
  import os

  if not os.path.exists(filename):
    st.warning(f"找不到檔案 `{filename}`，請確認該檔案是否已上傳至 GitHub 倉庫。")
    return pd.DataFrame()

  try:
    df = pd.read_csv(filename)
    if "Date" in df.columns:
      df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
      df = df.dropna(subset=["Date"])
      df = df.sort_values("Date").reset_index(drop=True)
    return df
  except Exception as e:
    st.error(f"讀取 CSV 發生錯誤: {e}")
    return pd.DataFrame()


df = load_and_validate_data(csv_filename)

if df.empty:
  st.info(
      "目前沒有可用資料，請在左側確認 CSV 檔名或上傳 `data.csv` 檔案。"
  )
else:
  st.success(
      f"成功載入資料！共計 {len(df)} 筆交易日紀錄（從"
      f" {df['Date'].min().strftime('%Y-%m-%d')} 到"
      f" {df['Date'].max().strftime('%Y-%m-%d')}）。"
  )

  # 顯示原始資料預覽
  with st.expander("查看原始數據預覽"):
    st.dataframe(df.tail(10))

  # ==========================================
  # 3. 策略核心運算邏輯
  # ==========================================
  # 這裡預留策略計算區塊（包含雙均線防守、50%折半配置、左側抄底與右側出場）
  st.markdown("---")
  st.subheader("📈 策略回測與績效分析")
  st.info(
      "策略核心架構已就緒，可於此處擴充動態再平衡、MDD、CAGR 計算與對比圖表。"
  )
