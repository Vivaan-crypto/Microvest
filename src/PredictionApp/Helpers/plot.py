import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

df = pd.read_csv("C:/GitHub/Microvest/src/PredictionApp/Data/CSV/train.csv", parse_dates=["Date"])
df = df.sort_values(["ticker", "Date"])

color_map = {1: "green", -1: "red", 0: "yellow"}
tickers = df["ticker"].unique()

output_dir = "../label_charts"
os.makedirs(output_dir, exist_ok=True)

for ticker in tickers:
    data = df[df["ticker"] == ticker].reset_index(drop=True)
    colors = data["Label"].map(color_map)

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(data["Date"], data["Close"], color="#CCCCCC", linewidth=0.8, zorder=1)
    ax.scatter(data["Date"], data["Close"], c=colors, s=12, zorder=2, linewidths=0)

    ax.set_facecolor("#0A0A0F")
    fig.patch.set_facecolor("#0A0A0F")
    ax.tick_params(colors="#888888")
    ax.spines["bottom"].set_color("#222233")
    ax.spines["top"].set_color("#222233")
    ax.spines["left"].set_color("#222233")
    ax.spines["right"].set_color("#222233")
    ax.yaxis.label.set_color("#888888")
    ax.xaxis.label.set_color("#888888")
    ax.title.set_color("#FFFFFF")

    ax.set_title(f"{ticker}", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")

    patches = [
        mpatches.Patch(color="green", label="Long (+1)"),
        mpatches.Patch(color="red", label="Short (-1)"),
        mpatches.Patch(color="yellow", label="No Trade (0)"),
    ]
    ax.legend(handles=patches, facecolor="#0F0F1A", edgecolor="#222233",
              labelcolor="#CCCCCC", fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{ticker}.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {ticker}")

print(f"/nDone. {len(tickers)} charts saved to '{output_dir}/'")