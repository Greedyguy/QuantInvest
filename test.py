from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

sec = pdr.get_data_yahoo('005930.KS', start='2018-05-04')
msft = pdr.get_data_yahoo('MSFT', start='2018-05-04')

import matplotlib.pyplot as plt
sec_dpc = (sec['Close']-sec['Close'].shift(1)) / sec['Close'].shift(1)  * 100
sec_dpc.iloc[0] = 0
plt.hist(sec_dpc,bins=18)
plt.grid(True)
plt.show()