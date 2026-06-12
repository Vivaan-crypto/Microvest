"""
Configuration for stock prediction model training.
Centralized hyperparameters and settings.
"""

# ========== DATA CONFIGURATION ==========
DATA_CONFIG = {
    "tickers": [
        # Technology
        "AAPL", "MSFT", "GOOGL", "NVDA", "AVGO", "AMD", "INTC", "ASML", "CRM", "ADBE",
        "META", "NFLX", "DIS", "CSCO", "IBM", "ORCL", "QCOM", "BROADCOM", "PAYX", "ADP",
        "INTU", "SNPS", "CDNS", "SPLK", "DDOG", "CRWD", "OKTA", "ZM", "FTNT",
        # Consumer
        "AMZN", "TSLA", "HD", "TJX", "LOW", "CPRT", "BURL", "ULTA", "DECK", "LULU",
        "NKE", "VF", "GIL",
        # Healthcare
        "JNJ", "PFE", "MRK", "UNH", "ABBV", "AMGN", "GILD", "BIIB", "ALNY", "VRNA",
        "BNTX", "MRNA", "VRTX", "BMRN", "SGEN", "JAZZ", "EXAS",
        # Financials
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "CME", "ICE",
        "CBOE", "AME", "AMP", "BEN", "BRL", "CFG", "COF",
        # Energy
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "WMB", "OKE", "EPD",
        "LNG", "RRC", "MRO", "DVN",
        # Industrials
        "CAT", "BA", "GE", "HON", "MMM", "RTX", "LMT", "NOC", "GD", "TRI",
        "DXPE", "PH", "FLS", "IEX", "RHI", "ROL",
        # Materials
        "PG", "KO", "PEP", "COST", "WMT", "MDLZ", "MO", "PM",
        "NEM", "GOLD", "SLG", "DD", "FCX", "STLD", "X",
        # Utilities
        "NEE", "DUK", "SO", "EXC", "AWK", "LW", "CEG", "AEP", "XEL",
        # Real Estate
        "VICI", "EQIX", "DLR", "PSA", "CCI", "SBAC", "PEAK", "PLD", "DRE",
        # Communication
        "VZ", "T", "TMUS", "DISH", "LBRDK", "CHTR", "CMCSA",
    ],
    "start_date": "2021-01-01",
    "end_date": "2026-01-01",
    "train_end": "2024-12-31",
    "target_horizon": 5,  # Predict 5 days ahead
    "threshold_pct": 0.5,  # ±0.5% threshold for Up/Down
    "window_size": 20,  # 20-day lookback window
}

# ========== MODEL CONFIGURATION ==========
MODEL_CONFIG = {
    "model_type": "lstm",  # "lstm" or "transformer"
    "input_size": 70,  # Number of features (from indicators_extended)
    "num_classes": 3,  # Down=0, Flat=1, Up=2

    # LSTM specific
    "lstm": {
        "hidden_size": 256,
        "num_layers": 3,
        "bidirectional": True,
        "dropout": 0.3,
    },

    # Transformer specific
    "transformer": {
        "d_model": 256,
        "num_layers": 6,
        "nhead": 8,
        "dim_feedforward": 1024,
        "dropout": 0.3,
    },
}

# ========== TRAINING CONFIGURATION ==========
TRAINING_CONFIG = {
    "batch_size": 64,  # Increased from 32 for better gradient estimates
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    "max_epochs": 300,
    "early_stopping_patience": 30,
    "early_stopping_min_delta": 0.001,

    # Loss function
    "loss_fn": "focal",  # "focal" (recommended) or "ce"
    "focal_gamma": 2.0,  # Focusing parameter for focal loss
    "focal_alpha": "balanced",  # Use balanced class weights

    # Optimizer
    "optimizer": "adamw",
    "gradient_clip_val": 1.0,
    "gradient_clip_algorithm": "norm",

    # Scheduler
    "use_lr_scheduler": True,
    "scheduler_type": "cosine",  # "cosine", "plateau", or "step"
    "cosine_t_max": 100,
    "plateau_patience": 10,
    "plateau_factor": 0.5,

    # Device
    "accelerator": "cpu",  # "cpu" or "gpu"
    "num_workers": 0,  # Disable multiprocessing for debugging
}

# ========== VALIDATION & MONITORING ==========
VALIDATION_CONFIG = {
    "validation_split": 0.2,
    "test_split": 0.1,
    "monitor_metric": "val/f1",  # Metric to track for best model
    "monitor_mode": "max",  # "max" for F1, "min" for loss
    "save_top_k": 3,  # Save top 3 checkpoints
    "log_every_n_steps": 10,
    "enable_progress_bar": True,
}

# ========== POST-TRAINING ==========
INFERENCE_CONFIG = {
    "use_threshold_optimization": True,
    "threshold_tuning_metric": "f1",  # "f1", "accuracy", "balanced_accuracy"
    "confidence_threshold": 0.5,  # Minimum predicted confidence
    "return_probabilities": True,
}

# ========== AUGMENTATION (Future) ==========
AUGMENTATION_CONFIG = {
    "use_mixup": False,  # Mixup for sequence data
    "use_cutmix": False,
    "mixup_alpha": 0.2,
    "use_oversample_minority": False,
    "minority_oversample_ratio": 1.5,
}
