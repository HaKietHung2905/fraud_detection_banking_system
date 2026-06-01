"""
ATM & Card Skimming Detection
==============================
File  : src/models/atm_detector.py
Covers: Physical card skimming, ATM jackpotting, cash-out attacks,
        PIN capture, and coordinated ATM fraud rings.

Attack patterns
---------------
  Skimming     : card reader overlay captures magnetic stripe + camera captures PIN
  Jackpotting  : malware makes ATM dispense cash on command
  Cash-out ring: compromised card data used at multiple ATMs simultaneously
  Shimming     : chip-card skimming via thin device inside card slot
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve,
)
from sklearn.base import BaseEstimator, TransformerMixin

RAW_SCHEMA: Dict[str, object] = {
    # ATM & transaction signals
    "amount":                        200.0,
    "is_max_withdrawal":             0,      # 1 = hit daily withdrawal limit
    "pin_entry_speed_ms":            2000,   # milliseconds to enter PIN (fast = suspicious)
    "card_read_method":              1,      # 1=chip, 0=magstripe
    "contactless":                   0,
    "atm_country_risk":              0.1,

    # ATM device signals
    "atm_tamper_alert":              0,      # 1 = ATM reported tampering
    "atm_fraud_reports_24h":         0,      # other fraud reports at same ATM in 24h
    "atm_location_type":             1,      # 1=bank branch, 0=standalone/offsite
    "atm_network_anomaly":           0,      # 1 = unusual network traffic to ATM

    # Velocity & pattern
    "tx_count_same_atm_1h":          1,      # transactions at this exact ATM in 1h
    "cards_used_same_atm_1h":        1,      # distinct cards at same ATM in 1h
    "atm_hop_count_1h":              1,      # number of distinct ATMs used in 1h
    "failed_pin_attempts":           0,
    "withdrew_yesterday":            0,      # 1 = also withdrew at ATM yesterday
    "time_since_last_atm_tx_min":    1440,   # minutes since last ATM tx

    # Geographic
    "distance_from_home_atm_km":     5.0,    # distance from usual ATM location
    "country_mismatch":              0,
    "cross_border_atm":              0,

    # Card signals
    "card_present_without_chip":     0,      # magstripe fallback on chip card
    "international_card_domestic_atm": 0,
    "card_reported_compromised":     0,      # card on known compromise list

    # Behavioral
    "atm_usage_pattern_deviation":   0.05,   # deviation from normal ATM usage [0-1]
    "time_of_day_risk":              0.1,    # risk score based on hour [0-1]
}

MODEL_FEATURES: List[str] = [
    "is_max_withdrawal", "card_read_method", "atm_tamper_alert",
    "atm_fraud_reports_24h", "atm_location_type", "atm_network_anomaly",
    "tx_count_same_atm_1h", "cards_used_same_atm_1h", "atm_hop_count_1h",
    "failed_pin_attempts", "withdrew_yesterday",
    "country_mismatch", "cross_border_atm", "card_present_without_chip",
    "international_card_domestic_atm", "card_reported_compromised",
    "atm_usage_pattern_deviation", "time_of_day_risk",
    # engineered
    "log_amount", "pin_speed_risk", "atm_cluster_risk",
    "geographic_risk", "cash_out_pattern",
]


class ATMFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, val in RAW_SCHEMA.items():
            if col not in df.columns: df[col] = val

        df["log_amount"]  = np.log1p(df["amount"])

        # Fast PIN entry (< 800ms) suggests pre-captured PIN
        df["pin_speed_risk"] = (df["pin_entry_speed_ms"] < 800).astype(int)

        # Multiple cards at same ATM = skimming device present
        df["atm_cluster_risk"] = (
            np.log1p(df["cards_used_same_atm_1h"]) * 0.50
            + df["atm_tamper_alert"] * 0.30
            + df["atm_fraud_reports_24h"].clip(upper=10) / 10 * 0.20
        )

        df["geographic_risk"] = (
            df["country_mismatch"] * 0.40
            + df["cross_border_atm"] * 0.30
            + np.log1p(df["distance_from_home_atm_km"]) / 10 * 0.30
        ).clip(0, 1)

        # Rapid ATM hopping + max withdrawals = coordinated cash-out
        df["cash_out_pattern"] = (
            df["is_max_withdrawal"] * 0.40
            + np.log1p(df["atm_hop_count_1h"]) * 0.30
            + df["atm_usage_pattern_deviation"] * 0.30
        ).clip(0, 1)

        return df


class ATMDataGenerator:
    @staticmethod
    def generate(n=15_000, fraud_rate=0.025, seed=49):
        rng = np.random.default_rng(seed)
        nf = int(n * fraud_rate); nl = n - nf

        legit = pd.DataFrame({
            "amount": rng.choice([20,40,60,80,100,200,300,400,500], nl),
            "is_max_withdrawal": rng.binomial(1, 0.05, nl),
            "pin_entry_speed_ms": rng.integers(1500, 8000, nl),
            "card_read_method": rng.binomial(1, 0.90, nl),
            "contactless": rng.binomial(1, 0.20, nl),
            "atm_country_risk": rng.uniform(0.0, 0.1, nl),
            "atm_tamper_alert": rng.binomial(1, 0.01, nl),
            "atm_fraud_reports_24h": rng.integers(0, 1, nl),
            "atm_location_type": rng.binomial(1, 0.70, nl),
            "atm_network_anomaly": rng.binomial(1, 0.01, nl),
            "tx_count_same_atm_1h": rng.integers(1, 3, nl),
            "cards_used_same_atm_1h": rng.integers(1, 5, nl),
            "atm_hop_count_1h": rng.integers(1, 2, nl),
            "failed_pin_attempts": rng.integers(0, 1, nl),
            "withdrew_yesterday": rng.binomial(1, 0.20, nl),
            "time_since_last_atm_tx_min": rng.exponential(2000, nl),
            "distance_from_home_atm_km": rng.exponential(3, nl),
            "country_mismatch": rng.binomial(1, 0.02, nl),
            "cross_border_atm": rng.binomial(1, 0.03, nl),
            "card_present_without_chip": rng.binomial(1, 0.02, nl),
            "international_card_domestic_atm": rng.binomial(1, 0.05, nl),
            "card_reported_compromised": np.zeros(nl, int),
            "atm_usage_pattern_deviation": rng.uniform(0.0, 0.15, nl),
            "time_of_day_risk": rng.uniform(0.0, 0.2, nl),
            "label": np.zeros(nl, int),
        })

        fraud = pd.DataFrame({
            "amount": rng.choice([200, 300, 400, 500, 1000], nf),
            "is_max_withdrawal": rng.binomial(1, 0.85, nf),
            "pin_entry_speed_ms": rng.integers(200, 700, nf),   # very fast
            "card_read_method": rng.binomial(1, 0.10, nf),      # magstripe fallback
            "contactless": rng.binomial(1, 0.05, nf),
            "atm_country_risk": rng.uniform(0.5, 1.0, nf),
            "atm_tamper_alert": rng.binomial(1, 0.60, nf),
            "atm_fraud_reports_24h": rng.integers(3, 20, nf),
            "atm_location_type": rng.binomial(1, 0.10, nf),     # standalone ATM
            "atm_network_anomaly": rng.binomial(1, 0.70, nf),
            "tx_count_same_atm_1h": rng.integers(5, 30, nf),
            "cards_used_same_atm_1h": rng.integers(10, 50, nf),
            "atm_hop_count_1h": rng.integers(3, 10, nf),
            "failed_pin_attempts": rng.integers(0, 2, nf),
            "withdrew_yesterday": rng.binomial(1, 0.80, nf),
            "time_since_last_atm_tx_min": rng.exponential(10, nf),
            "distance_from_home_atm_km": rng.exponential(500, nf),
            "country_mismatch": rng.binomial(1, 0.70, nf),
            "cross_border_atm": rng.binomial(1, 0.60, nf),
            "card_present_without_chip": rng.binomial(1, 0.80, nf),
            "international_card_domestic_atm": rng.binomial(1, 0.50, nf),
            "card_reported_compromised": rng.binomial(1, 0.40, nf),
            "atm_usage_pattern_deviation": rng.uniform(0.7, 1.0, nf),
            "time_of_day_risk": rng.uniform(0.6, 1.0, nf),
            "label": np.ones(nf, int),
        })

        return pd.concat([legit, fraud]).sample(frac=1, random_state=seed).reset_index(drop=True)


@dataclass
class ATMResult:
    score: float; label: int; risk_level: str
    action: str; attack_type: str
    reason_codes: List[str]; threshold: float
    def __str__(self):
        return (f"score={self.score:.4f} | risk={self.risk_level} | "
                f"action={self.action} | attack={self.attack_type} | "
                f"reasons={self.reason_codes}")


class ATMSkimmingDetector:
    THRESHOLD = 0.40
    ISO_WEIGHT = 0.35; RF_WEIGHT = 0.65

    def __init__(self):
        self.fe = ATMFeatureEngineer()
        self.scaler = RobustScaler()
        self.isoforest = IsolationForest(n_estimators=250, contamination=0.025,
                                          random_state=42, n_jobs=-1)
        self.rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                          class_weight="balanced", random_state=42, n_jobs=-1)

    def fit(self, df, verbose=True):
        X = self.fe.fit_transform(df); y = X.pop("label")
        for f in MODEL_FEATURES:
            if f not in X.columns: X[f] = 0.0
        Xs = self.scaler.fit_transform(X[MODEL_FEATURES])
        self.isoforest.fit(Xs)
        Xtr, Xv, ytr, yv = train_test_split(Xs, y, stratify=y, test_size=0.2, random_state=42)
        self.rf.fit(Xtr, ytr)
        if verbose:
            p = self._score(Xv)
            print(f"  ROC-AUC: {roc_auc_score(yv,p):.4f} | AvgP: {average_precision_score(yv,p):.4f}")
        return self

    def _score(self, Xs):
        iso = np.clip((-self.isoforest.score_samples(Xs) - 0.05) / 0.5, 0, 1)
        rf  = self.rf.predict_proba(Xs)[:,1]
        return self.ISO_WEIGHT * iso + self.RF_WEIGHT * rf

    def _prepare(self, record):
        row = {**RAW_SCHEMA, **record}
        df = self.fe.transform(pd.DataFrame([row]))
        for f in MODEL_FEATURES:
            if f not in df.columns: df[f] = 0.0
        return self.scaler.transform(df[MODEL_FEATURES])

    def _attack_type(self, record, score):
        if score < self.THRESHOLD: return "legitimate"
        if record.get("atm_tamper_alert") or record.get("cards_used_same_atm_1h", 1) > 8:
            return "skimming_device"
        if record.get("atm_network_anomaly"): return "jackpotting"
        if record.get("atm_hop_count_1h", 1) > 3: return "cash_out_ring"
        return "card_cloning"

    def _reason_codes(self, record):
        codes = []
        if record.get("atm_tamper_alert"): codes.append("ATM_TAMPER_ALERT")
        if record.get("cards_used_same_atm_1h", 1) > 8: codes.append("CLUSTER_ATM_USAGE")
        if record.get("card_reported_compromised"): codes.append("COMPROMISED_CARD")
        if record.get("card_present_without_chip"): codes.append("MAGSTRIPE_FALLBACK")
        if record.get("is_max_withdrawal"): codes.append("MAX_WITHDRAWAL")
        if record.get("atm_hop_count_1h", 1) > 3: codes.append("RAPID_ATM_HOPPING")
        if record.get("atm_network_anomaly"): codes.append("ATM_NETWORK_ANOMALY")
        if record.get("country_mismatch"): codes.append("COUNTRY_MISMATCH")
        return codes

    def predict(self, record: Dict) -> ATMResult:
        Xs = self._prepare(record)
        ml = float(self._score(Xs)[0])
        # Hard rule: compromised card = instant block
        if record.get("card_reported_compromised"): ml = 1.0
        if record.get("atm_tamper_alert") and record.get("is_max_withdrawal"): ml = max(ml, 0.90)
        risk = "HIGH" if ml >= 0.75 else "MEDIUM" if ml >= self.THRESHOLD else "LOW"
        action = {"HIGH": "BLOCK_CARD", "MEDIUM": "STEP_UP_AUTH", "LOW": "ALLOW"}[risk]
        return ATMResult(score=round(ml, 4), label=int(ml >= self.THRESHOLD),
                         risk_level=risk, action=action,
                         attack_type=self._attack_type(record, ml),
                         reason_codes=self._reason_codes(record),
                         threshold=self.THRESHOLD)

    def evaluate(self, df):
        X = self.fe.transform(df); y = X.pop("label")
        for f in MODEL_FEATURES:
            if f not in X.columns: X[f] = 0.0
        p = self._score(self.scaler.transform(X[MODEL_FEATURES]))
        pred = (p >= self.THRESHOLD).astype(int)
        return {"roc_auc": roc_auc_score(y,p), "avg_precision": average_precision_score(y,p),
                "report": classification_report(y, pred, digits=4)}


def _s(t): print(f"\n{'═'*60}\n  {t}\n{'═'*60}")

def main():
    _s("ATM & Card Skimming Detection")
    df = ATMDataGenerator.generate(n=18_000, fraud_rate=0.025)
    print(f"  Total: {len(df):,} | Fraud: {df['label'].sum():,}")
    det = ATMSkimmingDetector(); det.fit(df)
    m = det.evaluate(df)
    print(f"\n  ROC-AUC: {m['roc_auc']:.4f} | AvgP: {m['avg_precision']:.4f}")
    print(m["report"])

    _s("Live Inference")
    cases = [
        ({"amount": 500, "is_max_withdrawal": 1, "pin_entry_speed_ms": 350,
          "card_read_method": 0, "atm_tamper_alert": 1, "cards_used_same_atm_1h": 25,
          "atm_hop_count_1h": 5, "atm_fraud_reports_24h": 12,
          "card_reported_compromised": 1, "country_mismatch": 1},
         "Card skimming cash-out ring (expected: BLOCK_CARD)"),
        ({"amount": 100, "is_max_withdrawal": 0, "pin_entry_speed_ms": 3500,
          "card_read_method": 1, "atm_tamper_alert": 0, "cards_used_same_atm_1h": 2,
          "atm_hop_count_1h": 1, "card_reported_compromised": 0, "country_mismatch": 0},
         "Routine ATM withdrawal (expected: ALLOW)"),
    ]
    for rec, desc in cases:
        r = det.predict(rec)
        print(f"\n  {desc}\n  {r}")

if __name__ == "__main__": main()