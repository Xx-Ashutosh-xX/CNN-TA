import pandas as pd
import talib as ta # TA-Lib for calculation of indicators


def add_indicators(data):
    data["RSI"] = ta.RSI(data["Close"])
    data["EMA"] = ta.EMA(data["Close"])
    data["WMA"] = ta.WMA(data["Close"])
    data["ROC"] = ta.ROC(data["Close"])
    data["TEMA"] = ta.TEMA(data["Close"])
    data["CMO"] = ta.CMO(data["Close"])
    data["SAR"] = ta.SAR(data["High"],data["Low"])
    data["WILLR"] = ta.WILLR(data["High"],data["Low"],data["Close"],timeperiod = 15)
    data["CCI"] = ta.CCI(data["High"],data["Low"],data["Close"],timeperiod = 15)
    data["PPO"] = ta.PPO(data["Close"],fastperiod = 6,slowperiod = 15)
    data["MACD"] = ta.MACD(data["Close"],fastperiod = 6,slowperiod = 15)[0]
    a = ta.WMA(data["Close"],timeperiod = 15//2)
    b = data["WMA"]
    data["HMA"] = ta.WMA(2*a - b,timeperiod = int(15**(0.5)))
    data["ADX"] = ta.ADX(data["High"],data["Low"],data["Close"],timeperiod = 15)
    data.dropna(inplace = True)
    

data = pd.read_csv("Data/TTM.csv")

data.dropna(inplace = True) # Drop missing entries

data.set_index("Date",inplace = True) # Set Date as the index column 

data.drop(labels = ["Close"], axis = 1, inplace = True) # Use Adj Close instead of Close
data.rename(columns = {"Adj Close":"Close"}, inplace = True)

add_indicators(data) # Add indicators to the dataframe

data.drop(labels = ["Volume","Low","High"], axis = 1, inplace = True) # Drop Volume column

data.to_csv("data.csv") # Save Data as CSV