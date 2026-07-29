@st.cache_data
def load_local_data():
    if not os.path.exists(csv_filename):
        return None
        
    raw_df = pd.read_csv(csv_filename)
    
    # 使用 format='mixed' 與 errors='coerce' 避免日期格式解析錯誤
    raw_df['Date'] = pd.to_datetime(raw_df['Date'], format='mixed', errors='coerce')
    raw_df = raw_df.dropna(subset=['Date'])
    
    if raw_df.empty:
        return None
        
    raw_df = raw_df.sort_values('Date').set_index('Date')
    
    # 強制數字型態轉換
    for col in ['TAIEX', '00631L', 'NDFI', 'PCR_5MA']:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
            
    raw_df = raw_df.ffill().bfill()
    
    # 計算防守用雙均線
    raw_df['MA240'] = raw_df['TAIEX'].rolling(window=240).mean()
    raw_df['MA120'] = raw_df['TAIEX'].rolling(window=120).mean()
    
    # 回傳時改用 dropna 移除 MA 產生的 NaN，但若資料不夠 240 筆則退而求其次只 Drop 子欄位
    return raw_df.dropna(subset=['TAIEX', '00631L'])

df = load_local_data()

# 💡 防呆檢查：如果 df 為空，直接顯示提示並停止執行
if df is None or df.empty:
    st.error("❌ 載入的資料集為空或格式錯誤！請檢查您的 `data.csv` 檔案內容、欄位名稱是否正確，或確認資料筆數是否足夠計算均線（至少需 240 筆以上）。")
    st.stop()
