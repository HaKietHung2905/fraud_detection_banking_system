"""
Credit Card Fraud Detection — Transaction Feature Engineering
=============================================================
File  : src/models/card_detector.py
Covers: Velocity, geography, merchant risk, device presence,
        spending pattern deviation, and card-not-present signals.

Architecture
------------
  Raw transaction event
        │
        ▼
  CardFeatureEngineer     ← derives 16 velocity + pattern signals
        │
        ├─► apply_hard_rules()   ← deterministic blocks (impossible travel, known BIN fraud, etc.)
        │
        └─► XGBoost              ← primary classifier (handles imbalance via scale_pos_weight)
              + calibrated threshold tuning via Precision-Recall curve
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
    precision_recall_curve,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw Feature Schema
# ─────────────────────────────────────────────────────────────────────────────

# All 30 raw fields a card transaction event should carry.
# Missing fields are filled with safe defaults during inference.
RAW_SCHEMA: Dict[str, object] = {

    # ── Transaction basics ────────────────────────────────────────
    "amount":                   50.0,   # transaction amount in local currency
    "hour":                     12,     # local hour (0–23)
    "day_of_week":              2,      # 0=Mon … 6=Sun
    "is_weekend":               0,      # 1 = Saturday or Sunday
    "month":                    6,      # 1–12

    # ── Merchant & Terminal ───────────────────────────────────────
    "merchant_risk_score":      0.1,    # merchant category fraud rate [0–1]
    "is_online":                0,      # 1 = card-not-present (e-commerce)
    "is_new_merchant":          0,      # 1 = never seen this MCC/merchant before
    "merchant_country_match":   1,      # 1 = merchant country == card home country
    "mcc_code":                 5411,   # Merchant Category Code (grocery = 5411)

    # ── Card presence & auth ─────────────────────────────────────
    "card_present":             1,      # 1 = physical card swiped/dipped
    "chip_used":                1,      # 1 = EMV chip used (harder to clone)
    "pin_used":                 1,      # 1 = PIN entered
    "contactless":              0,      # 1 = tap-to-pay

    # ── Velocity — short window ───────────────────────────────────
    "tx_count_1h":              1,      # transactions in last 1 hour
    "tx_count_6h":              2,      # transactions in last 6 hours
    "tx_count_24h":             4,      # transactions in last 24 hours
    "tx_count_7d":              18,     # transactions in last 7 days
    "amount_sum_1h":            50.0,   # total spend in last 1 hour
    "amount_sum_24h":           150.0,  # total spend in last 24 hours
    "declined_count_24h":       0,      # declined attempts in last 24 hours
    "distinct_merchants_24h":   2,      # unique merchants in last 24 hours
    "distinct_countries_24h":   1,      # unique countries in last 24 hours

    # ── Geography ─────────────────────────────────────────────────
    "distance_from_home_km":    5.0,    # distance from cardholder home address
    "distance_from_last_tx_km": 2.0,    # distance from last transaction location
    "time_since_last_tx_min":   240.0,  # minutes since last transaction
    "country_mismatch":         0,      # IP/merchant country ≠ card home country

    # ── Historical pattern ────────────────────────────────────────
    "avg_amount_30d":           55.0,   # cardholder's 30-day average transaction
    "std_amount_30d":           20.0,   # standard deviation of amounts
    "max_amount_30d":           200.0,  # largest transaction in 30 days
    "avg_tx_per_day_30d":       3.0,    # average daily transaction count
}

# Features passed to XGBoost after engineering
MODEL_FEATURES: List[str] = [
    # raw pass-through
    "merchant_risk_score", "is_online", "is_new_merchant",
    "merchant_country_match", "card_present", "chip_used", "pin_used",
    "contactless", "is_weekend", "country_mismatch",
    "tx_count_1h", "tx_count_6h", "tx_count_24h", "tx_count_7d",
    "declined_count_24h", "distinct_merchants_24h", "distinct_countries_24h",
    # engineered
    "log_amount",
    "amount_vs_avg",
    "amount_vs_max",
    "amount_zscore",
    "log_distance_home",
    "log_distance_last_tx",
    "speed_km_per_min",
    "impossible_speed",
    "velocity_1h_ratio",
    "velocity_24h_ratio",
    "spend_ratio_1h",
    "spend_ratio_24h",
    "is_night",
    "is_high_risk_hour",
    "cnp_risk",
    "velocity_spike",
]

# High-risk MCC codes (cash advance, gambling, crypto, wire transfer)
HIGH_RISK_MCC = {
    6010, 6011, 6012,   # ATM / financial institutions
    7995,               # gambling
    6051,               # crypto / money orders
    4829,               # wire transfer
    6540,               # stored value load
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering Transformer
# ─────────────────────────────────────────────────────────────────────────────

class CardFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Derives 16 high-signal features from raw card transaction fields.

    Engineered features
    -------------------
    log_amount          : log1p(amount) — reduce right-skew
    amount_vs_avg       : amount / (avg_amount_30d + 1)    — deviation from normal spend
    amount_vs_max       : amount / (max_amount_30d + 1)    — relative to historical max
    amount_zscore       : (amount - avg) / (std + 1)       — standard-score anomaly
    log_distance_home   : log1p(distance_from_home_km)
    log_distance_last   : log1p(distance_from_last_tx_km)
    speed_km_per_min    : distance_from_last_tx_km / (time_since_last_tx_min + 0.1)
    impossible_speed    : 1 if speed > 3 km/min (180 km/h — impossible on foot)
    velocity_1h_ratio   : tx_count_1h / (avg_tx_per_day_30d / 24 + 0.01)
    velocity_24h_ratio  : tx_count_24h / (avg_tx_per_day_30d + 0.01)
    spend_ratio_1h      : amount_sum_1h / (avg_amount_30d * avg_tx_per_day_30d / 24 + 1)
    spend_ratio_24h     : amount_sum_24h / (avg_amount_30d * avg_tx_per_day_30d + 1)
    is_night            : 1 if hour in 00:00–05:59
    is_high_risk_hour   : 1 if hour in 00:00–04:59 or 23:00
    cnp_risk            : card-not-present + new merchant + high risk MCC composite
    velocity_spike      : sharp increase in tx count relative to 7-day baseline
    """

    CAPS: Dict[str, float] = {
        "amount":                    100_000.0,
        "distance_from_home_km":     20_000.0,
        "distance_from_last_tx_km":  20_000.0,
        "time_since_last_tx_min":    10_080.0,  # 7 days
        "tx_count_1h":               100.0,
        "tx_count_24h":              500.0,
        "amount_sum_1h":             500_000.0,
        "amount_sum_24h":            1_000_000.0,
        "declined_count_24h":        50.0,
        "distinct_merchants_24h":    100.0,
    }

    def fit(self, X: pd.DataFrame, y=None) -> "CardFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Fill missing columns with schema defaults
        for col, default in RAW_SCHEMA.items():
            if col not in df.columns:
                df[col] = default

        # Cap extreme outliers
        for col, cap in self.CAPS.items():
            if col in df.columns:
                df[col] = df[col].clip(upper=cap)

        # ── Amount features ───────────────────────────────────────
        df["log_amount"]    = np.log1p(df["amount"])
        df["amount_vs_avg"] = df["amount"] / (df["avg_amount_30d"] + 1)
        df["amount_vs_max"] = df["amount"] / (df["max_amount_30d"] + 1)
        df["amount_zscore"] = (
            (df["amount"] - df["avg_amount_30d"])
            / (df["std_amount_30d"] + 1)
        ).clip(-10, 10)

        # ── Geography ─────────────────────────────────────────────
        df["log_distance_home"]    = np.log1p(df["distance_from_home_km"])
        df["log_distance_last_tx"] = np.log1p(df["distance_from_last_tx_km"])

        # Speed between consecutive transactions
        df["speed_km_per_min"] = (
            df["distance_from_last_tx_km"]
            / (df["time_since_last_tx_min"] + 0.1)
        ).clip(upper=10)

        # Impossible physical speed (>3 km/min ≈ 180 km/h while shopping)
        df["impossible_speed"] = (df["speed_km_per_min"] > 3).astype(int)

        # ── Velocity ratios ───────────────────────────────────────
        # Compare current velocity to personal historical baseline
        expected_per_hour = df["avg_tx_per_day_30d"] / 24 + 0.01
        df["velocity_1h_ratio"]  = df["tx_count_1h"]  / expected_per_hour
        df["velocity_24h_ratio"] = df["tx_count_24h"] / (df["avg_tx_per_day_30d"] + 0.01)

        expected_spend_1h  = df["avg_amount_30d"] * df["avg_tx_per_day_30d"] / 24 + 1
        expected_spend_24h = df["avg_amount_30d"] * df["avg_tx_per_day_30d"] + 1
        df["spend_ratio_1h"]  = df["amount_sum_1h"]  / expected_spend_1h
        df["spend_ratio_24h"] = df["amount_sum_24h"] / expected_spend_24h

        # ── Timing ────────────────────────────────────────────────
        df["is_night"]         = df["hour"].between(0, 5).astype(int)
        df["is_high_risk_hour"] = (
            df["hour"].between(0, 4) | (df["hour"] == 23)
        ).astype(int)

        # ── Card-not-present risk composite ───────────────────────
        # Online + new merchant + no chip + high-risk MCC → elevated CNP risk
        high_risk_mcc = df["mcc_code"].isin(HIGH_RISK_MCC).astype(int)
        df["cnp_risk"] = (
            df["is_online"]        * 0.35
            + df["is_new_merchant"] * 0.25
            + (1 - df["chip_used"]) * 0.20
            + high_risk_mcc         * 0.20
        )

        # ── Velocity spike ────────────────────────────────────────
        # Sudden burst: much higher 1h rate than 7d daily average
        daily_baseline = df["tx_count_7d"] / 7 + 0.01
        hourly_rate    = df["tx_count_1h"] / 1.0
        df["velocity_spike"] = (hourly_rate / (daily_baseline / 24)).clip(upper=100)

        return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Synthetic Data Generator
# ─────────────────────────────────────────────────────────────────────────────

class CardDataGenerator:
    """
    Generates 4 fraud archetypes:
      - Card cloner     : physical card cloned, used far from home
      - CNP fraudster   : stolen card details used online
      - Account drainer : rapid high-value burst before card is blocked
      - Friendly fraud  : normal-looking transaction, slightly anomalous amount
    """

    @staticmethod
    def generate(
        n_samples: int = 20_000,
        fraud_rate: float = 0.02,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n_fraud = int(n_samples * fraud_rate)
        n_legit = n_samples - n_fraud

        # ── Legitimate transactions ───────────────────────────────
        avg_30d = rng.lognormal(3.8, 0.6, n_legit)
        amount_legit = np.clip(
            rng.normal(avg_30d, avg_30d * 0.3), 1, None
        )
        legit = pd.DataFrame({
            "amount":                   amount_legit,
            "hour":                     rng.integers(7, 22,    n_legit),
            "day_of_week":              rng.integers(0, 7,     n_legit),
            "is_weekend":               rng.binomial(1, 0.29,  n_legit),
            "month":                    rng.integers(1, 13,    n_legit),
            "merchant_risk_score":      rng.uniform(0.0, 0.15, n_legit),
            "is_online":                rng.binomial(1, 0.30,  n_legit),
            "is_new_merchant":          rng.binomial(1, 0.08,  n_legit),
            "merchant_country_match":   rng.binomial(1, 0.97,  n_legit),
            "mcc_code":                 rng.choice([5411,5812,5941,7011,5912], n_legit),
            "card_present":             rng.binomial(1, 0.80,  n_legit),
            "chip_used":                rng.binomial(1, 0.78,  n_legit),
            "pin_used":                 rng.binomial(1, 0.70,  n_legit),
            "contactless":              rng.binomial(1, 0.35,  n_legit),
            "tx_count_1h":              rng.integers(0, 3,     n_legit),
            "tx_count_6h":              rng.integers(0, 5,     n_legit),
            "tx_count_24h":             rng.integers(1, 8,     n_legit),
            "tx_count_7d":              rng.integers(5, 30,    n_legit),
            "amount_sum_1h":            amount_legit * rng.uniform(1, 2, n_legit),
            "amount_sum_24h":           amount_legit * rng.uniform(2, 5, n_legit),
            "declined_count_24h":       rng.integers(0, 1,     n_legit),
            "distinct_merchants_24h":   rng.integers(1, 5,     n_legit),
            "distinct_countries_24h":   np.ones(n_legit, int),
            "distance_from_home_km":    rng.exponential(8,     n_legit),
            "distance_from_last_tx_km": rng.exponential(3,     n_legit),
            "time_since_last_tx_min":   rng.exponential(300,   n_legit),
            "country_mismatch":         rng.binomial(1, 0.01,  n_legit),
            "avg_amount_30d":           avg_30d,
            "std_amount_30d":           avg_30d * rng.uniform(0.2, 0.5, n_legit),
            "max_amount_30d":           avg_30d * rng.uniform(2, 5,     n_legit),
            "avg_tx_per_day_30d":       rng.uniform(1, 8,      n_legit),
            "label":                    np.zeros(n_legit, int),
        })

        # ── Fraud archetypes ──────────────────────────────────────
        n_each = n_fraud // 4
        n_rem  = n_fraud - n_each * 4

        avg_legit = rng.lognormal(3.8, 0.6, n_each)

        # Archetype 1: Card cloner — physical card cloned, used far from home
        cloner = pd.DataFrame({
            "amount":                   rng.lognormal(5.5, 1.0, n_each),
            "hour":                     rng.choice([0,1,2,3,4,22,23], n_each),
            "day_of_week":              rng.integers(0, 7,     n_each),
            "is_weekend":               rng.binomial(1, 0.50,  n_each),
            "month":                    rng.integers(1, 13,    n_each),
            "merchant_risk_score":      rng.uniform(0.4, 0.9,  n_each),
            "is_online":                np.zeros(n_each, int),
            "is_new_merchant":          rng.binomial(1, 0.80,  n_each),
            "merchant_country_match":   rng.binomial(1, 0.10,  n_each),
            "mcc_code":                 rng.choice([6010,6011,7995,4829], n_each),
            "card_present":             np.ones(n_each, int),
            "chip_used":                np.zeros(n_each, int),   # magstripe clone
            "pin_used":                 np.zeros(n_each, int),
            "contactless":              np.zeros(n_each, int),
            "tx_count_1h":              rng.integers(2, 8,     n_each),
            "tx_count_6h":              rng.integers(4, 15,    n_each),
            "tx_count_24h":             rng.integers(5, 20,    n_each),
            "tx_count_7d":              rng.integers(5, 25,    n_each),
            "amount_sum_1h":            rng.lognormal(6.5, 1.0, n_each),
            "amount_sum_24h":           rng.lognormal(7.0, 1.0, n_each),
            "declined_count_24h":       rng.integers(1, 5,     n_each),
            "distinct_merchants_24h":   rng.integers(3, 10,    n_each),
            "distinct_countries_24h":   rng.integers(2, 4,     n_each),
            "distance_from_home_km":    rng.exponential(1500,  n_each),
            "distance_from_last_tx_km": rng.exponential(200,   n_each),
            "time_since_last_tx_min":   rng.exponential(30,    n_each),
            "country_mismatch":         rng.binomial(1, 0.85,  n_each),
            "avg_amount_30d":           avg_legit,
            "std_amount_30d":           avg_legit * 0.3,
            "max_amount_30d":           avg_legit * 3,
            "avg_tx_per_day_30d":       rng.uniform(1, 5,      n_each),
            "label":                    np.ones(n_each, int),
        })

        # Archetype 2: CNP fraudster — stolen card details, online purchase
        cnp = pd.DataFrame({
            "amount":                   rng.lognormal(5.0, 1.2, n_each),
            "hour":                     rng.choice([0,1,2,3,4,5,23], n_each),
            "day_of_week":              rng.integers(0, 7,     n_each),
            "is_weekend":               rng.binomial(1, 0.50,  n_each),
            "month":                    rng.integers(1, 13,    n_each),
            "merchant_risk_score":      rng.uniform(0.3, 0.8,  n_each),
            "is_online":                np.ones(n_each, int),
            "is_new_merchant":          rng.binomial(1, 0.90,  n_each),
            "merchant_country_match":   rng.binomial(1, 0.15,  n_each),
            "mcc_code":                 rng.choice([6051,5999,7995,4829], n_each),
            "card_present":             np.zeros(n_each, int),
            "chip_used":                np.zeros(n_each, int),
            "pin_used":                 np.zeros(n_each, int),
            "contactless":              np.zeros(n_each, int),
            "tx_count_1h":              rng.integers(3, 12,    n_each),
            "tx_count_6h":              rng.integers(5, 20,    n_each),
            "tx_count_24h":             rng.integers(5, 25,    n_each),
            "tx_count_7d":              rng.integers(5, 20,    n_each),
            "amount_sum_1h":            rng.lognormal(6.0, 1.0, n_each),
            "amount_sum_24h":           rng.lognormal(6.5, 1.0, n_each),
            "declined_count_24h":       rng.integers(2, 8,     n_each),
            "distinct_merchants_24h":   rng.integers(4, 12,    n_each),
            "distinct_countries_24h":   rng.integers(1, 3,     n_each),
            "distance_from_home_km":    rng.exponential(800,   n_each),
            "distance_from_last_tx_km": rng.exponential(500,   n_each),
            "time_since_last_tx_min":   rng.exponential(15,    n_each),
            "country_mismatch":         rng.binomial(1, 0.70,  n_each),
            "avg_amount_30d":           avg_legit,
            "std_amount_30d":           avg_legit * 0.3,
            "max_amount_30d":           avg_legit * 3,
            "avg_tx_per_day_30d":       rng.uniform(1, 5,      n_each),
            "label":                    np.ones(n_each, int),
        })

        # Archetype 3: Account drainer — rapid burst before card blocked
        n_drainer = n_each
        avg_d = rng.lognormal(3.8, 0.6, n_drainer)
        drainer = pd.DataFrame({
            "amount":                   rng.lognormal(6.5, 0.8, n_drainer),
            "hour":                     rng.integers(0, 24,    n_drainer),
            "day_of_week":              rng.integers(0, 7,     n_drainer),
            "is_weekend":               rng.binomial(1, 0.30,  n_drainer),
            "month":                    rng.integers(1, 13,    n_drainer),
            "merchant_risk_score":      rng.uniform(0.5, 1.0,  n_drainer),
            "is_online":                rng.binomial(1, 0.50,  n_drainer),
            "is_new_merchant":          rng.binomial(1, 0.70,  n_drainer),
            "merchant_country_match":   rng.binomial(1, 0.20,  n_drainer),
            "mcc_code":                 rng.choice([6010,6011,7995,6051], n_drainer),
            "card_present":             rng.binomial(1, 0.50,  n_drainer),
            "chip_used":                rng.binomial(1, 0.30,  n_drainer),
            "pin_used":                 rng.binomial(1, 0.20,  n_drainer),
            "contactless":              rng.binomial(1, 0.20,  n_drainer),
            "tx_count_1h":              rng.integers(5, 20,    n_drainer),
            "tx_count_6h":              rng.integers(8, 30,    n_drainer),
            "tx_count_24h":             rng.integers(10, 40,   n_drainer),
            "tx_count_7d":              rng.integers(10, 35,   n_drainer),
            "amount_sum_1h":            rng.lognormal(7.5, 0.8, n_drainer),
            "amount_sum_24h":           rng.lognormal(8.0, 0.8, n_drainer),
            "declined_count_24h":       rng.integers(3, 10,    n_drainer),
            "distinct_merchants_24h":   rng.integers(5, 15,    n_drainer),
            "distinct_countries_24h":   rng.integers(1, 4,     n_drainer),
            "distance_from_home_km":    rng.exponential(500,   n_drainer),
            "distance_from_last_tx_km": rng.exponential(100,   n_drainer),
            "time_since_last_tx_min":   rng.exponential(10,    n_drainer),
            "country_mismatch":         rng.binomial(1, 0.50,  n_drainer),
            "avg_amount_30d":           avg_d,
            "std_amount_30d":           avg_d * 0.3,
            "max_amount_30d":           avg_d * 3,
            "avg_tx_per_day_30d":       rng.uniform(1, 6,      n_drainer),
            "label":                    np.ones(n_drainer, int),
        })

        # Archetype 4: Friendly fraud — normal-looking, slightly high amount
        n_friendly = n_each + n_rem
        avg_f = rng.lognormal(3.8, 0.5, n_friendly)
        friendly = pd.DataFrame({
            "amount":                   avg_f * rng.uniform(2.5, 6, n_friendly),
            "hour":                     rng.integers(8, 22,    n_friendly),
            "day_of_week":              rng.integers(0, 7,     n_friendly),
            "is_weekend":               rng.binomial(1, 0.30,  n_friendly),
            "month":                    rng.integers(1, 13,    n_friendly),
            "merchant_risk_score":      rng.uniform(0.1, 0.4,  n_friendly),
            "is_online":                rng.binomial(1, 0.70,  n_friendly),
            "is_new_merchant":          rng.binomial(1, 0.50,  n_friendly),
            "merchant_country_match":   rng.binomial(1, 0.80,  n_friendly),
            "mcc_code":                 rng.choice([5411,5812,5945,5999], n_friendly),
            "card_present":             rng.binomial(1, 0.40,  n_friendly),
            "chip_used":                rng.binomial(1, 0.40,  n_friendly),
            "pin_used":                 rng.binomial(1, 0.35,  n_friendly),
            "contactless":              rng.binomial(1, 0.30,  n_friendly),
            "tx_count_1h":              rng.integers(0, 2,     n_friendly),
            "tx_count_6h":              rng.integers(0, 4,     n_friendly),
            "tx_count_24h":             rng.integers(1, 6,     n_friendly),
            "tx_count_7d":              rng.integers(5, 25,    n_friendly),
            "amount_sum_1h":            avg_f * rng.uniform(2, 5,   n_friendly),
            "amount_sum_24h":           avg_f * rng.uniform(3, 8,   n_friendly),
            "declined_count_24h":       rng.integers(0, 2,     n_friendly),
            "distinct_merchants_24h":   rng.integers(1, 4,     n_friendly),
            "distinct_countries_24h":   np.ones(n_friendly, int),
            "distance_from_home_km":    rng.exponential(20,    n_friendly),
            "distance_from_last_tx_km": rng.exponential(10,    n_friendly),
            "time_since_last_tx_min":   rng.exponential(180,   n_friendly),
            "country_mismatch":         rng.binomial(1, 0.05,  n_friendly),
            "avg_amount_30d":           avg_f,
            "std_amount_30d":           avg_f * rng.uniform(0.2, 0.4, n_friendly),
            "max_amount_30d":           avg_f * rng.uniform(2, 4,     n_friendly),
            "avg_tx_per_day_30d":       rng.uniform(1, 6,      n_friendly),
            "label":                    np.ones(n_friendly, int),
        })

        return (
            pd.concat([legit, cloner, cnp, drainer, friendly])
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rule-Based Pre-Filter
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    triggered:   bool
    rules_fired: List[str] = field(default_factory=list)
    score_boost: float = 0.0


def apply_hard_rules(record: Dict) -> RuleResult:
    """
    Hard rules → immediate BLOCK regardless of ML score.
    Soft rules → add score boost to ML output.
    """
    fired: List[str] = []
    boost = 0.0

    # ── Hard rules ────────────────────────────────────────────────
    hard = [
        (record.get("impossible_speed", 0) == 1
         and record.get("distance_from_last_tx_km", 0) > 100,
         "HARD:IMPOSSIBLE_TRAVEL"),

        (record.get("distinct_countries_24h", 1) >= 3,
         "HARD:MULTI_COUNTRY_24H"),

        (record.get("declined_count_24h", 0) >= 5,
         "HARD:EXCESSIVE_DECLINES"),

        (record.get("tx_count_1h", 0) >= 10,
         "HARD:VELOCITY_BURST_1H"),

        (record.get("mcc_code", 0) in HIGH_RISK_MCC
         and record.get("amount", 0) > 5_000,
         "HARD:HIGH_RISK_MCC_LARGE_AMOUNT"),
    ]
    for condition, name in hard:
        if condition:
            fired.append(name)

    # ── Soft rules ────────────────────────────────────────────────
    soft = [
        (record.get("country_mismatch", 0)
         and record.get("is_online", 0),
         "SOFT:COUNTRY_MISMATCH_CNP", 0.20),

        (not record.get("chip_used", 1)
         and not record.get("pin_used", 1)
         and record.get("card_present", 0),
         "SOFT:MAGSTRIPE_NO_PIN", 0.15),

        (record.get("is_new_merchant", 0)
         and record.get("merchant_risk_score", 0) > 0.5,
         "SOFT:NEW_HIGH_RISK_MERCHANT", 0.15),

        (record.get("amount", 0) > record.get("max_amount_30d", 9e9) * 1.5,
         "SOFT:AMOUNT_EXCEEDS_HISTORICAL_MAX", 0.20),

        (record.get("declined_count_24h", 0) >= 2
         and record.get("tx_count_1h", 0) >= 3,
         "SOFT:DECLINES_PLUS_VELOCITY", 0.15),

        (record.get("is_night", 0)
         and record.get("country_mismatch", 0),
         "SOFT:NIGHT_PLUS_COUNTRY_MISMATCH", 0.10),
    ]
    for condition, name, b in soft:
        if condition:
            fired.append(name)
            boost += b

    return RuleResult(
        triggered=any("HARD:" in f for f in fired),
        rules_fired=fired,
        score_boost=min(boost, 0.40),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Card Fraud Detector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CardResult:
    score:        float
    label:        int
    risk_level:   str
    action:       str
    archetype:    str
    reason_codes: List[str]
    rule_result:  RuleResult
    threshold:    float

    def __str__(self) -> str:
        return (
            f"score={self.score:.4f} | risk={self.risk_level} | "
            f"action={self.action} | archetype={self.archetype} | "
            f"reasons={self.reason_codes}"
        )


class CardFraudDetector:
    """
    XGBoost-based credit card fraud detector.

    Why XGBoost over Random Forest here?
    - Handles class imbalance natively via scale_pos_weight.
    - Better on tabular transaction data with many correlated features.
    - Faster inference — critical for real-time card auth (<200ms SLA).
    - Built-in regularisation (L1/L2) prevents overfitting on rare fraud.
    """

    THRESHOLD     = 0.50    # tuned for high recall; lower = more fraud caught
    FRAUD_RATE    = 0.02    # expected fraud prevalence for scale_pos_weight

    def __init__(self):
        self.fe     = CardFeatureEngineer()
        self.scaler = StandardScaler()
        self.model  = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            scale_pos_weight=int((1 - self.FRAUD_RATE) / self.FRAUD_RATE),
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="aucpr",
            
            random_state=42,
            n_jobs=-1,
        )
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "CardFraudDetector":
        X = self.fe.fit_transform(df)
        y = X.pop("label")

        for f in MODEL_FEATURES:
            if f not in X.columns:
                X[f] = 0.0

        X_feat = X[MODEL_FEATURES]
        X_sc   = self.scaler.fit_transform(X_feat)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sc, y, stratify=y, test_size=0.20, random_state=42
        )
        self.model.fit(
            X_tr, y_tr,
            
        )
        self._trained = True

        if verbose:
            val_proba = self.model.predict_proba(X_val)[:, 1]
            val_pred  = (val_proba >= self.THRESHOLD).astype(int)
            print("── Validation (hold-out 20%) ──")
            print(f"  ROC-AUC        : {roc_auc_score(y_val, val_proba):.4f}")
            print(f"  Avg Precision  : {average_precision_score(y_val, val_proba):.4f}")
            print(classification_report(y_val, val_pred, digits=4))

        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _prepare_single(self, record: Dict) -> np.ndarray:
        row = {**RAW_SCHEMA, **record}
        df  = self.fe.transform(pd.DataFrame([row]))
        for f in MODEL_FEATURES:
            if f not in df.columns:
                df[f] = 0.0
        return self.scaler.transform(df[MODEL_FEATURES])

    # ── Archetype ─────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_archetype(record: Dict, score: float) -> str:
        if score < 0.35:
            return "benign"
        online      = record.get("is_online", 0)
        card_pres   = record.get("card_present", 1)
        distance    = record.get("distance_from_home_km", 0)
        velocity    = record.get("tx_count_1h", 0)
        chip        = record.get("chip_used", 1)

        if velocity >= 5 and record.get("amount", 0) > record.get("avg_amount_30d", 1) * 5:
            return "account_drainer"
        if online and not card_pres:
            return "cnp_fraudster"
        if distance > 200 and not chip:
            return "card_cloner"
        return "friendly_fraud"

    # ── Reason codes ──────────────────────────────────────────────────────────

    @staticmethod
    def _reason_codes(record: Dict) -> List[str]:
        codes = []
        if record.get("country_mismatch"):                          codes.append("COUNTRY_MISMATCH")
        if record.get("distance_from_home_km", 0) > 100:           codes.append("FAR_FROM_HOME")
        if record.get("impossible_speed", 0):                       codes.append("IMPOSSIBLE_TRAVEL_SPEED")
        if record.get("tx_count_1h", 0) > 4:                       codes.append("HIGH_VELOCITY_1H")
        if record.get("declined_count_24h", 0) > 1:                codes.append("PRIOR_DECLINES")
        if record.get("merchant_risk_score", 0) > 0.5:             codes.append("HIGH_RISK_MERCHANT")
        if record.get("is_new_merchant", 0):                        codes.append("NEW_MERCHANT")
        if not record.get("chip_used", 1) and record.get("card_present", 0):
                                                                    codes.append("MAGSTRIPE_ONLY")
        if record.get("is_online", 0) and not record.get("card_present", 1):
                                                                    codes.append("CARD_NOT_PRESENT")
        if record.get("distinct_countries_24h", 1) > 1:            codes.append("MULTI_COUNTRY")
        amount = record.get("amount", 0)
        avg    = record.get("avg_amount_30d", 1)
        if amount > avg * 5:                                        codes.append("AMOUNT_5X_AVERAGE")
        elif amount > avg * 3:                                      codes.append("AMOUNT_3X_AVERAGE")
        if record.get("mcc_code", 0) in HIGH_RISK_MCC:             codes.append("HIGH_RISK_MCC")
        if record.get("is_night", 0):                               codes.append("NIGHT_TRANSACTION")
        if record.get("spend_ratio_1h", 0) > 5:                    codes.append("SPEND_SPIKE_1H")
        return codes

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self, record: Dict) -> CardResult:
        # Enrich with engineered fields needed by rules
        enriched = self._enrich_for_rules(record)

        rule_result = apply_hard_rules(enriched)

        X_sc     = self._prepare_single(enriched)
        ml_score = float(self.model.predict_proba(X_sc)[0, 1])

        final_score = min(ml_score + rule_result.score_boost, 1.0)
        if rule_result.triggered:
            final_score = 1.0

        label     = int(final_score >= self.THRESHOLD)
        risk      = self._risk_level(final_score)
        action    = {"HIGH": "BLOCK", "MEDIUM": "REVIEW", "LOW": "ALLOW"}[risk]
        archetype = self._classify_archetype(enriched, final_score)
        reasons   = self._reason_codes(enriched) + rule_result.rules_fired

        return CardResult(
            score=round(final_score, 4),
            label=label,
            risk_level=risk,
            action=action,
            archetype=archetype,
            reason_codes=reasons,
            rule_result=rule_result,
            threshold=self.THRESHOLD,
        )

    def _enrich_for_rules(self, record: Dict) -> Dict:
        """Pre-compute engineered flags needed by the rule engine."""
        r = dict(record)
        dist = r.get("distance_from_last_tx_km", 0)
        mins = r.get("time_since_last_tx_min", 1) + 0.1
        r["impossible_speed"] = int(dist / mins > 3)
        r["is_night"]         = int(r.get("hour", 12) < 6)
        r["spend_ratio_1h"]   = (
            r.get("amount_sum_1h", 0)
            / (r.get("avg_amount_30d", 1) * r.get("avg_tx_per_day_30d", 1) / 24 + 1)
        )
        return r

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.70:   return "HIGH"
        elif score >= 0.35: return "MEDIUM"
        else:               return "LOW"

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> Dict:
        X = self.fe.transform(df)
        y_true = X.pop("label")
        for f in MODEL_FEATURES:
            if f not in X.columns:
                X[f] = 0.0
        X_sc   = self.scaler.transform(X[MODEL_FEATURES])
        proba  = self.model.predict_proba(X_sc)[:, 1]
        y_pred = (proba >= self.THRESHOLD).astype(int)
        return {
            "roc_auc":       roc_auc_score(y_true, proba),
            "avg_precision": average_precision_score(y_true, proba),
            "confusion":     confusion_matrix(y_true, y_pred).tolist(),
            "report":        classification_report(y_true, y_pred, digits=4),
        }

    def feature_importance(self) -> pd.DataFrame:
        return (
            pd.DataFrame({
                "feature":    MODEL_FEATURES,
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def tune_threshold(self, df: pd.DataFrame) -> pd.DataFrame:
        """Precision-Recall tradeoff across thresholds."""
        X = self.fe.transform(df)
        y_true = X.pop("label")
        for f in MODEL_FEATURES:
            if f not in X.columns:
                X[f] = 0.0
        proba = self.model.predict_proba(
            self.scaler.transform(X[MODEL_FEATURES])
        )[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        return pd.DataFrame({
            "threshold": np.append(thresholds, 1.0),
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
        })

    def cross_validate(self, df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
        """Stratified k-fold CV for robust performance estimate."""
        X = self.fe.transform(df)
        y = X.pop("label")
        for f in MODEL_FEATURES:
            if f not in X.columns:
                X[f] = 0.0

        skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        records = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            sc = StandardScaler()
            Xtr_sc = sc.fit_transform(X_tr[MODEL_FEATURES])
            Xvl_sc = sc.transform(X_val[MODEL_FEATURES])

            m = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                scale_pos_weight=49, subsample=0.8, colsample_bytree=0.8,
                eval_metric="aucpr", random_state=42, n_jobs=-1,
            )
            m.fit(Xtr_sc, y_tr, eval_set=[(Xvl_sc, y_val)], verbose=False)

            proba  = m.predict_proba(Xvl_sc)[:, 1]
            y_pred = (proba >= self.THRESHOLD).astype(int)
            records.append({
                "fold":          fold,
                "roc_auc":       roc_auc_score(y_val, proba),
                "avg_precision": average_precision_score(y_val, proba),
                "precision_1":   confusion_matrix(y_val, y_pred)[1, 1]
                                 / (y_pred.sum() + 1e-9),
                "recall_1":      confusion_matrix(y_val, y_pred)[1, 1]
                                 / (y_val.sum() + 1e-9),
            })

        cv_df = pd.DataFrame(records)
        cv_df.loc["mean"] = cv_df.mean()
        return cv_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main — Train + Evaluate + Demo
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str):
    print(f"\n{'═' * 64}\n  {title}\n{'═' * 64}")


def main():
    _section("Generating Synthetic Card Transaction Data (4 archetypes)")
    df = CardDataGenerator.generate(n_samples=30_000, fraud_rate=0.02)
    n_fraud = df["label"].sum()
    print(f"  Total    : {len(df):,} transactions")
    print(f"  Fraud    : {n_fraud:,} ({n_fraud/len(df):.1%})")
    print(f"  Features : {len(RAW_SCHEMA)} raw → {len(MODEL_FEATURES)} model features")

    _section("Training CardFraudDetector (XGBoost)")
    detector = CardFraudDetector()
    detector.fit(df)

    _section("Full Dataset Evaluation")
    metrics = detector.evaluate(df)
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision  : {metrics['avg_precision']:.4f}")
    print(f"  Confusion      : {metrics['confusion']}")
    print(metrics["report"])

    _section("Feature Importance (top 12)")
    print(detector.feature_importance().head(12).to_string(index=False))

    _section("Threshold Tuning — Precision / Recall tradeoff")
    tdf = detector.tune_threshold(df)
    best = tdf[tdf["f1"] > 0.60].sort_values("f1", ascending=False).head(6)
    print(best.to_string(index=False))

    _section("Cross-Validation (5-fold stratified)")
    cv = detector.cross_validate(df, n_splits=5)
    print(cv.to_string())

    _section("Live Inference Demo")

    cases = [
        # ── Card cloner ───────────────────────────────────────────
        ({
            "amount": 3_500, "hour": 2, "mcc_code": 6011,
            "merchant_risk_score": 0.85, "is_online": 0, "is_new_merchant": 1,
            "merchant_country_match": 0, "card_present": 1,
            "chip_used": 0, "pin_used": 0, "contactless": 0,
            "tx_count_1h": 5, "tx_count_24h": 12, "tx_count_7d": 14,
            "amount_sum_1h": 8_000, "amount_sum_24h": 15_000,
            "declined_count_24h": 3, "distinct_merchants_24h": 6,
            "distinct_countries_24h": 2,
            "distance_from_home_km": 2_200, "distance_from_last_tx_km": 800,
            "time_since_last_tx_min": 25, "country_mismatch": 1,
            "avg_amount_30d": 60, "std_amount_30d": 20,
            "max_amount_30d": 250, "avg_tx_per_day_30d": 3,
        }, "Card cloner — ATM abroad, no chip, far from home"),

        # ── CNP fraudster ─────────────────────────────────────────
        ({
            "amount": 1_200, "hour": 3, "mcc_code": 6051,
            "merchant_risk_score": 0.75, "is_online": 1, "is_new_merchant": 1,
            "merchant_country_match": 0, "card_present": 0,
            "chip_used": 0, "pin_used": 0, "contactless": 0,
            "tx_count_1h": 7, "tx_count_24h": 18, "tx_count_7d": 20,
            "amount_sum_1h": 5_000, "amount_sum_24h": 9_000,
            "declined_count_24h": 4, "distinct_merchants_24h": 9,
            "distinct_countries_24h": 2,
            "distance_from_home_km": 900, "distance_from_last_tx_km": 400,
            "time_since_last_tx_min": 8, "country_mismatch": 1,
            "avg_amount_30d": 70, "std_amount_30d": 25,
            "max_amount_30d": 300, "avg_tx_per_day_30d": 4,
        }, "CNP fraudster — stolen card, crypto merchant online"),

        # ── Account drainer ───────────────────────────────────────
        ({
            "amount": 9_800, "hour": 1, "mcc_code": 6010,
            "merchant_risk_score": 0.90, "is_online": 0, "is_new_merchant": 1,
            "merchant_country_match": 0, "card_present": 1,
            "chip_used": 0, "pin_used": 0, "contactless": 0,
            "tx_count_1h": 12, "tx_count_24h": 30, "tx_count_7d": 32,
            "amount_sum_1h": 40_000, "amount_sum_24h": 80_000,
            "declined_count_24h": 6, "distinct_merchants_24h": 12,
            "distinct_countries_24h": 3,
            "distance_from_home_km": 1_500, "distance_from_last_tx_km": 300,
            "time_since_last_tx_min": 5, "country_mismatch": 1,
            "avg_amount_30d": 80, "std_amount_30d": 30,
            "max_amount_30d": 400, "avg_tx_per_day_30d": 3,
        }, "Account drainer — rapid burst, ATM, massive spend spike"),

        # ── Borderline (should REVIEW) ────────────────────────────
        ({
            "amount": 450, "hour": 20, "mcc_code": 5812,
            "merchant_risk_score": 0.20, "is_online": 1, "is_new_merchant": 1,
            "merchant_country_match": 1, "card_present": 0,
            "chip_used": 0, "pin_used": 0, "contactless": 0,
            "tx_count_1h": 2, "tx_count_24h": 4, "tx_count_7d": 15,
            "amount_sum_1h": 500, "amount_sum_24h": 700,
            "declined_count_24h": 0, "distinct_merchants_24h": 3,
            "distinct_countries_24h": 1,
            "distance_from_home_km": 30, "distance_from_last_tx_km": 15,
            "time_since_last_tx_min": 90, "country_mismatch": 0,
            "avg_amount_30d": 65, "std_amount_30d": 25,
            "max_amount_30d": 280, "avg_tx_per_day_30d": 3,
        }, "Borderline — online, new merchant, 7x average (expected: REVIEW)"),

        # ── Legitimate ────────────────────────────────────────────
        ({
            "amount": 72, "hour": 13, "mcc_code": 5411,
            "merchant_risk_score": 0.05, "is_online": 0, "is_new_merchant": 0,
            "merchant_country_match": 1, "card_present": 1,
            "chip_used": 1, "pin_used": 1, "contactless": 0,
            "tx_count_1h": 1, "tx_count_24h": 2, "tx_count_7d": 14,
            "amount_sum_1h": 72, "amount_sum_24h": 130,
            "declined_count_24h": 0, "distinct_merchants_24h": 2,
            "distinct_countries_24h": 1,
            "distance_from_home_km": 2, "distance_from_last_tx_km": 0.5,
            "time_since_last_tx_min": 300, "country_mismatch": 0,
            "avg_amount_30d": 68, "std_amount_30d": 22,
            "max_amount_30d": 250, "avg_tx_per_day_30d": 3,
        }, "Legitimate — grocery store, chip+PIN, home neighbourhood"),
    ]

    for record, description in cases:
        result = detector.predict(record)
        print(f"\n  Case      : {description}")
        print(f"  Result    : {result}")


if __name__ == "__main__":
    main()