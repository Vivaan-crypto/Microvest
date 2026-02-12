import json

stocks = [
    "NVDA", "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "AVGO", "META", "TSLA",
    "LLY", "JPM", "WMT", "ORCL", "V", "XOM", "MA", "JNJ", "NFLX", "PLTR",
    "ABBV", "COST", "AMD", "BAC", "HD", "PG", "GE", "CVX", "CSCO", "KO",
    "UNH", "IBM", "MU", "WFC", "MS", "CAT", "GS", "AXP", "PM", "TMUS",
    "RTX", "CRM", "MRK", "ABT", "MCD", "TMO", "PEP", "LIN", "ISRG", "UBER",
    "DIS", "APP", "QCOM", "LRCX", "INTU", "T", "AMGN", "AMAT", "C", "NOW",
    "NEE", "BX", "VZ", "BLK", "INTC", "SCHW", "ANET", "APH", "BKNG", "TJX",
    "GEV", "DHR", "GILD", "BSX", "ACN", "SPGI", "KLAC", "BA", "TXN", "PFE",
    "PANW", "ADBE", "SYK", "ETN", "CRWD", "COF", "WELL", "UNP", "PGR", "DE",
    "LOW", "HON", "MDT", "PLD", "CB", "ADI", "COP", "VRTX", "HOOD", "HCA",
    "LMT", "KKR", "CEG", "PH", "MCK", "CME", "ADP", "CMCSA", "SO", "CVS",
    "MO", "SBUX", "NEM", "DUK", "BMY", "NKE", "GD", "TT", "DELL", "MMC",
    "DASH", "MMM", "ICE", "AMT", "CDNS", "MCO", "WM", "ORLY", "SHW", "HWM",
    "UPS", "NOC", "JCI", "EQIX", "BK", "MAR", "COIN", "APO", "TDG", "AON",
    "CTAS", "WMB", "ABNB", "MDLZ", "ECL", "USB", "ELV", "SNPS", "PNC", "CI",
    "EMR", "REGN", "ITW", "GLW", "COR", "TEL", "MNST", "RCL", "SPG", "AJG",
    "GM", "CSX", "RSG", "DDOG", "AEP", "AZO", "TRV", "PWR", "CMI", "NSC",
    "ADSK", "MSI", "FDX", "CL", "HLT", "WDAY", "FTNT", "KMI", "MPC", "SRE",
    "AFL", "EOG", "VST", "PYPL", "APD", "FCX", "TFC", "PSX", "WBD", "STX",
    "ALL", "VLO", "BDX", "DLR", "SLB", "IDXX", "LHX", "WDC", "ZTS", "URI",
    "F", "O", "ROST", "MET", "D", "PCAR", "EA", "EW", "NDAQ", "NXPI",
    "CAH", "ROP", "PSA", "BKR", "XEL", "FAST", "EXC", "CARR", "CBRE", "CTVA",
    "AME", "OKE", "KR", "LVS", "MPWR", "GWW", "AXON", "TTWO", "ETR", "FANG",
    "AMP", "MSCI", "ROK", "OXY", "AIG", "DHI", "CMG", "A", "YUM", "PEG",
    "FICO", "TGT", "PAYX", "CCI", "CPRT", "EBAY", "DAL", "IQV", "PRU", "EQT",
    "GRMN", "HIG", "TRGP", "VMC", "VTR", "KDP", "XYZ", "ED", "HSY", "PCG",
    "WEC", "MLM", "TKO", "SYY", "RMD", "CTSH", "WAB", "XYL", "OTIS", "KMB",
    "CCL", "NUE", "ACGL", "GEHC", "FIS", "STT", "VICI", "EXPE", "KVUE", "EL",
    "NRG", "LYV", "RJF", "LEN", "WTW", "KEYS", "UAL", "HPE", "VRSK", "IR",
    "CHTR", "EXR", "KHC", "IBKR", "TSCO", "WRB", "K", "MCHP", "CSGP", "FOXA",
    "MTB", "MTD", "HUM", "DTE", "AEE", "ADM", "FITB", "ATO", "ROL", "EXE",
    "EME", "ODFL", "BRO", "ES", "FOX", "PPL", "FSLR", "CBOE", "IRM", "TER",
    "FE", "BR", "SYF", "CNP", "AWK", "CINF", "STE", "EFX", "GIS", "AVB",
    "DOV", "HBAN", "BIIB", "VLTO", "LDOS", "NTRS", "ULTA", "TDY", "TPL", "VRSN",
    "PODD", "EQR", "PHM", "HUBB", "HAL", "DG", "HPQ", "STLD", "DXCM", "EIX",
    "WAT", "CMS", "DVN", "STZ", "CFG", "TROW", "WSM", "LH", "RF", "NTAP",
    "PPG", "SMCI", "L", "JBL", "PTC", "DLTR", "SBAC", "DGX", "TPR", "NVR",
    "NI", "INCY", "TTD", "LULU", "DRI", "CHD", "TYL", "RL", "CTRA", "IP",
    "AMCR", "CPAY", "KEY", "TSN", "CDW", "ON", "WST", "BG", "PFG", "EXPD",
    "J", "TRMB", "CHRW", "CNC", "SW", "ZBH", "GPC", "PKG", "EVRG", "GPN",
    "MKC", "GDDY", "Q", "INVH", "LNT", "PSKY", "SNA", "PNR", "APTV", "LUV",
    "ESS", "IFF", "IT", "DD", "LII", "HOLX", "GEN", "FTV", "DOW", "WY",
    "BBY", "MAA", "JBHT", "NWS", "ERIE", "NWSA", "LYB", "COO", "TXT", "UHS",
    "OMC", "ALLE", "KIM", "DPZ", "EG", "ALB", "FFIV", "AVY", "CF",
    "SOLV", "REG", "NDSN", "BALL", "CLX", "MAS", "UDR", "AKAM", "BXP", "HRL",
    "WYNN", "VTRS", "HII", "IEX", "DOC", "HST", "ZBRA", "DECK", "JKHY", "SJM",
    "BEN", "AIZ", "BLDR", "CPT", "DAY", "HAS", "PNW", "RVTY", "GL", "IVZ",
    "FDS", "SWK", "SWKS", "EPAM", "AES", "ALGN", "MRNA", "BAX", "CPB", "TECH",
    "TAP", "PAYC", "ARE", "POOL", "AOS", "IPG", "MGM", "GNRC", "APA", "DVA",
    "FRT", "HSIC", "CAG", "NCLH", "MOS", "CRL", "LW", "LKQ", "MTCH", "MOH",
    "SOLS", "MHK"
]
file_path = "Data/stock_dict.json"

def main():
    dict = {}
    for i, ticker in enumerate(stocks):
        dict[ticker] = i
    with open(file_path, 'w') as f:
        json.dump(dict, f, indent=2)

if __name__ == "__main__":
    main()
