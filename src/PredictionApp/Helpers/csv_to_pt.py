import json

from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import pandas as pd
import torch

if __name__ == "__main__":
    train = pd.read_csv("C:/GitHub/Microvest/src/PredictionApp/Data/CSV/train.csv")
    test = pd.read_csv("C:/GitHub/Microvest/src/PredictionApp/Data/CSV/test.csv")
    stock_dict = json.load(open("C:/GitHub/Microvest/src/PredictionApp/Data/stock_dict.json", "r"))
    for i in range(len(train.to_numpy().T[0])):
        train.at[train.index[i], "ticker"] = stock_dict[train.at[train.index[i], "ticker"]]
    for i in range(len(test.to_numpy().T[0])):
        test.at[test.index[i], "ticker"] = stock_dict[test.at[test.index[i], "ticker"]]
    y_train = torch.tensor(train.drop(columns=list(train.columns)[0:len(list(train.columns)) - 1]).values)
    y_test = torch.tensor(test.drop(columns=list(test.columns)[0:len(list(test.columns)) - 1]).values)
    x_train = torch.tensor(train.drop(columns=["Label", "Date"]).astype(float).values, dtype=torch.float32)
    x_test = torch.tensor(test.drop(columns=["Label", "Date"]).astype(float).values, dtype=torch.float32)
    #
    torch.save(x_train, "C:/GitHub/Microvest/src/PredictionApp/Data/CSV/x_train.pt")
    torch.save(y_train, "C:/GitHub/Microvest/src/PredictionApp/Data/CSV/y_train.pt")
    torch.save(x_test, "C:/GitHub/Microvest/src/PredictionApp/Data/CSV/x_test.pt")
    torch.save(y_test, "C:/GitHub/Microvest/src/PredictionApp/Data/CSV/y_test.pt")
