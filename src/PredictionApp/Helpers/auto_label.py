import pandas as pd

INPUT = "C:/GitHub/Microvest/src/PredictionApp/Data/CSV/test.csv"
OUTPUT = "C:/GitHub/Microvest/src/PredictionApp/Data/CSV/test.csv"

df = pd.read_csv(INPUT)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
percent = 0.07
# 1 = long (>=+5%), -1 = short (<=-5%), 0 = no trade, -2 = insufficient data
def label_ticker(group):
    group = group.copy()
    future_close = group["Close"].shift(-5)
    ret = (future_close - group["Close"]) / group["Close"]

    group["Label"] = 0
    group.loc[ret >= percent, "Label"] = 1
    group.loc[ret <= -percent, "Label"] = -1
    group.loc[group.index[-5:], "Label"] = -2  # insufficient future data
    return group

df = df.groupby("ticker", group_keys=False).apply(label_ticker)
df = df[df["Label"] != -2].copy()
df["Label"] = df["Label"].astype(int)

df.to_csv(OUTPUT, index=False)
print(f"Done. Saved to {OUTPUT}")
print(f"Label distribution:\n{df['Label'].value_counts()}")