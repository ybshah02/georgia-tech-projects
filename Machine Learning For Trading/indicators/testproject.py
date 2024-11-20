import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from TheoreticallyOptimalStrategy import testPolicy
import marketsimcode as msc
import indicators as ind
from util import get_data

def author():
    return 'yshah89'

def study_group():
    return 'yshah89'

def process_data():
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    symbol = 'JPM'

    date_range = pd.date_range(start_date, end_date)
    data_f = get_data([symbol], date_range)
    price_series = data_f[symbol]

    bench_trades = pd.DataFrame(0, index=price_series.index, columns=['Shares'])
    bench_trades.iloc[0] = 1000

    opt_trades = testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=100000)
    bench_pv = msc.compute_portvals(bench_trades, start_val=100000, commission=0, impact=0)
    opt_pv = msc.compute_portvals(opt_trades, start_val=100000, commission=0, impact=0)

    bench_pv_norm = bench_pv / bench_pv.iloc[0]
    opt_pv_norm = opt_pv / opt_pv.iloc[0]

    plot_results(bench_pv_norm, opt_pv_norm)
    stats = compute_statistics(bench_pv_norm, opt_pv_norm)
    ind.run_indicators(symbol, start_date, end_date)
    return stats

def plot_results(bench_pv_norm, opt_pv_norm):
    plt.figure(figsize=(14, 7))
    bench_pv_norm.plot(color='purple', label='Benchmark')
    opt_pv_norm.plot(color='red', label='Optimal Strategy')
    plt.title('Benchmark vs Optimal Strategy')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend()
    plt.savefig('images/bm_vs_opt.png')
    plt.close()

def compute_statistics(bench_pv_norm, opt_pv_norm):
    def stats(pv):
        daily_ret = pv.pct_change().dropna()
        cum_ret = (pv.iloc[-1] / pv.iloc[0]) - 1
        return cum_ret,  daily_ret.mean(), daily_ret.std()

    bm_cr, bm_adr, bm_sddr = stats(bench_pv_norm)
    opt_cr, opt_adr, opt_sddr = stats(opt_pv_norm)

    return (bm_cr, bm_adr, bm_sddr), (opt_cr, opt_adr, opt_sddr)


if __name__ == "__main__":
    stats = process_data()
    print(stats)