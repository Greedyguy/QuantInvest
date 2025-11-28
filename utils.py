import numpy as np, pandas as pd

def perf_stats(ec):
    # equity_curve가 Series인 경우 처리
    if isinstance(ec, pd.Series):
        equity_values = ec
    elif "equity" in ec.columns:
        equity_values = ec["equity"]
    else:
        equity_values = ec.iloc[:, 0] if len(ec.columns) > 0 else pd.Series()

    equity_values = equity_values.astype(float).dropna()
    if len(equity_values) == 0:
        return dict(CAGR=0, Vol=0, Sharpe=0, MDD=0, Days=0)

    ret = equity_values.pct_change().fillna(0)
    cagr = (equity_values.iloc[-1]/equity_values.iloc[0])**(252/len(equity_values)) - 1
    vol  = ret.std()*np.sqrt(252)
    sharpe = (ret.mean()/ret.std())*np.sqrt(252) if ret.std()>0 else 0
    dd = (equity_values/equity_values.cummax()-1).min()
    turn = np.nan  # 2차에서 추가
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, MDD=dd, Days=len(equity_values))
