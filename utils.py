import os
import torch
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def get_data():
    stock_list = os.listdir("dataset")
    stock_data = pd.DataFrame()
    for stocks in stock_list:
        stock_nm = stocks.split(".")[0]
        df = pd.read_csv("dataset/"+stocks)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df = df[["Open"]]
        df.columns = [stock_nm]
        stock_data = pd.concat([stock_data, df], axis=1)
        stock_data.sort_index(inplace=True)
    return stock_data


def scaled_data(data, step):
    train_data = data.loc[:"2014-12-31"]
    val_data = data.loc[:"2015-12-31"]
    test_data = data.loc[:]

    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(data)

    scaled_train = scaler.transform(train_data)
    scaled_val = scaler.transform(val_data)
    scaled_test = scaler.transform(test_data)
    
    x_train, y_train = scaled_train[:-step].T, scaled_train[step:].T
    x_val, y_val = scaled_val[:-step].T, scaled_val[step:].T
    x_test, y_test = scaled_test[:-step].T, scaled_test[step:].T
    
    x_train = torch.from_numpy(x_train).float().unsqueeze(-1)
    y_train = torch.from_numpy(y_train).float().unsqueeze(-1)
    x_val = torch.from_numpy(x_val).float().unsqueeze(-1)
    y_val = torch.from_numpy(y_val).float().unsqueeze(-1)
    x_test = torch.from_numpy(x_test).float().unsqueeze(-1)
    y_test = torch.from_numpy(y_test).float().unsqueeze(-1)
    
    return x_train, y_train, x_val, y_val, x_test, y_test, scaler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
