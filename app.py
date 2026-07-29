def run_backtest(df):
    score = (df['Institutional_Sell'].astype(int) + 
             cond_margin_danger.astype(int) + 
             cond_futures_bearish.astype(int) + 
             cond_pcr_extreme.astype(int))
    
    prev_score = score.shift(1).fillna(0)
    
    # 原始均線支撐條件
    price_near_support = (df['Close'] < df['MA20']) | (df['Close'] < df['MA60']) | (df['Close'] < df['MA240'] * 1.05)
    
    # 原始買入信號（僅依據籌碼風險解除與均線位置）
    add_signal = (prev_score >= 1) & (score == 0) & price_near_support
    
    position = np.where(score >= 3, 0.0, 
                        np.where(score == 2, 0.3, 
                                 np.where(score == 1, 0.7, 1.0)))
    
    pos_series = pd.Series(position, index=df.index).shift(1).fillna(1.0)
    strat_ret = df['Returns'] * pos_series
    cum_strat = (1 + strat_ret.fillna(0)).cumprod()
    mdd = ((cum_strat / cum_strat.cummax()) - 1).min()
    total_ret = cum_strat.iloc[-1] - 1
    
    return total_ret, mdd, score, add_signal
