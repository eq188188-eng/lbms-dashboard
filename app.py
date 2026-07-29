def run_backtest(df):
    score = (df['Institutional_Sell'].astype(int) + 
             cond_margin_danger.astype(int) + 
             cond_futures_bearish.astype(int) + 
             cond_pcr_extreme.astype(int))
    
    prev_score = score.shift(1).fillna(0)
    
    # 原始均線支撐條件
    price_near_support = (df['Close'] < df['MA20']) | (df['Close'] < df['MA60']) | (df['Close'] < df['MA240'] * 1.05)
    
    # 【優化項目】新增：計算成交量條件 (確保 yfinance 資料包含 Volume)
    if 'Volume' in df.columns:
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        volume_ok = df['Volume'] >= df['Vol_MA20'] * 0.8  # 至少不能極度無量
    else:
        volume_ok = True

    # 【優化項目】綜合條件：籌碼壓力解除 + 接近均線 + 成交量配合
    add_signal = (prev_score >= 1) & (score == 0) & price_near_support & volume_ok
    
    position = np.where(score >= 3, 0.0, 
                        np.where(score == 2, 0.3, 
                                 np.where(score == 1, 0.7, 1.0)))
    
    pos_series = pd.Series(position, index=df.index).shift(1).fillna(1.0)
    strat_ret = df['Returns'] * pos_series
    cum_strat = (1 + strat_ret.fillna(0)).cumprod()
    mdd = ((cum_strat / cum_strat.cummax()) - 1).min()
    total_ret = cum_strat.iloc[-1] - 1
    
    return total_ret, mdd, score, add_signal
