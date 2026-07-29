@st.cache_data
def load_local_data():
    raw_df = pd.read_csv(csv_filename)
    
    # 💡 修正處：使用 format='mixed' 自動解析混合格式，errors='coerce' 將無法解析的轉為 NaT
    raw_df['Date'] = pd.to_datetime(raw_df['Date'], format='mixed', errors='coerce')
    
    # 移除日期轉換失敗的無效資料列
    raw_df = raw_df.dropna(subset=['Date'])
    
    raw_df = raw_df.sort_values('Date').set_index('Date')
    
    # 強制數字型態轉換
    raw_df['TAIEX'] = raw_df['TAIEX'].astype(float)
    raw_df['00631L'] = raw_df['00631L'].astype(float)
    raw_df['NDFI'] = raw_df['NDFI'].astype(float)
    raw_df['PCR_5MA'] = raw_df['PCR_5MA'].astype(float)
    
    raw_df = raw_df.ffill().bfill()
    
    # 計算防守用雙均線
    raw_df['MA240'] = raw_df['TAIEX'].rolling(window=240).mean()
    raw_df['MA120'] = raw_df['TAIEX'].rolling(window=120).mean()
    return raw_df.dropna()
