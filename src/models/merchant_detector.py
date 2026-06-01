"""
Merchant Fraud Detection
=========================
File  : src/models/merchant_detector.py
Covers: Fake merchant accounts, card-testing attacks, chargeback fraud,
        refund abuse, triangulation fraud, and merchant collusion.

Merchant fraud typologies
--------------------------
  Card Testing    : using stolen cards in small amounts to verify validity
  Chargeback Fraud: merchant disputes legitimate charges with bank
  Refund Abuse    : issuing fraudulent refunds to own cards
  Triangulation   : legitimate customer order fulfilled with stolen card data
  Collusion       : merchant + customer collude to commit fraud
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score,
)
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb

RAW_SCHEMA: Dict[str, object] = {
    # Merchant profile
    "merchant_age_days":           180,    # how long merchant account exists
    "mcc_risk_score":              0.1,    # risk score of merchant category [0-1]
    "business_type_verified":      1,      # 1 = business legitimacy verified
    "kyb_score":                   0.8,    # Know Your Business score [0-1]

    # Chargeback & dispute signals
    "chargeback_rate_30d":         0.01,   # chargeback rate last 30 days
    "chargeback_count_7d":         0,
    "dispute_rate_90d":            0.02,
    "first_chargeback":            0,      # 1 = first chargeback ever

    # Transaction patterns
    "avg_ticket_size":             50.0,
    "ticket_size_std":             20.0,
    "tx_count_1h":                 10,
    "unique_cards_1h":             5,      # distinct cards in 1 hour
    "decline_rate_24h":            0.05,   # auth decline rate
    "refund_rate_30d":             0.02,
    "refund_count_7d":             0,

    # Card-testing signals
    "micro_tx_count_1h":           0,      # transactions under $1 in last hour
    "sequential_card_attempt":     0,      # 1 = sequential card BINs attempted
    "auth_without_purchase":       0,      # 1 = authorization not followed by capture
    "velocity_spike":              0,      # 1 = transaction spike vs historical

    # Geographic & channel
    "ip_diversity_score":          0.1,    # how many distinct IPs transactions come from
    "country_count_24h":           1,      # distinct countries in 24h
    "is_online_merchant":          1,
    "shipping_billing_mismatch":   0.05,   # % of orders with shipping ≠ billing

    # Financial signals
    "revenue_vs_expected":         1.0,    # ratio vs industry benchmark
    "refund_to_sales_ratio":       0.02,
    "avg_days_to_refund":          7.0,
}

MODEL_FEATURES: List[str] = [
    "mcc_risk_score", "business_type_verified", "kyb_score",
    "chargeback_rate_30d", "chargeback_count_7d", "dispute_rate_90d",
    "decline_rate_24h", "refund_rate_30d", "refund_count_7d",
    "micro_tx_count_1h", "sequential_card_attempt",
    "auth_without_purchase", "velocity_spike",
    "ip_diversity_score", "country_count_24h",
    "is_online_merchant", "shipping_billing_mismatch",
    "refund_to_sales_ratio",
    # engineered
    "log_merchant_age", "card_testing_score",
    "chargeback_risk", "refund_abuse_score",
    "log_unique_cards", "ticket_anomaly",
]


class MerchantFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, val in RAW_SCHEMA.items():
            if col not in df.columns: df[col] = val

        df["log_merchant_age"] = np.log1p(df["merchant_age_days"])

        # Card testing: many small transactions + high decline rate + sequential BINs
        df["card_testing_score"] = (
            np.log1p(df["micro_tx_count_1h"]) * 0.40
            + df["sequential_card_attempt"] * 0.30
            + df["auth_without_purchase"] * 0.20
            + df["decline_rate_24h"].clip(upper=1) * 0.10
        )

        df["chargeback_risk"] = (
            df["chargeback_rate_30d"].clip(upper=0.3) / 0.3 * 0.60
            + df["dispute_rate_90d"].clip(upper=0.2) / 0.2 * 0.40
        )

        df["refund_abuse_score"] = (
            df["refund_rate_30d"].clip(upper=0.5) / 0.5 * 0.50
            + df["refund_to_sales_ratio"].clip(upper=0.5) / 0.5 * 0.30
            + np.log1p(df["refund_count_7d"]) / 5 * 0.20
        )

        df["log_unique_cards"] = np.log1p(df["unique_cards_1h"])

        df["ticket_anomaly"] = (
            np.abs(df["avg_ticket_size"] - 50) / (df["ticket_size_std"] + 1)
        ).clip(upper=10) / 10

        return df


class MerchantDataGenerator:
    @staticmethod
    def generate(n=10_000, fraud_rate=0.05, seed=51):
        rng = np.random.default_rng(seed)
        nf = int(n * fraud_rate); nl = n - nf

        legit = pd.DataFrame({
            "merchant_age_days": rng.integers(90, 3650, nl),
            "mcc_risk_score": rng.uniform(0.0, 0.15, nl),
            "business_type_verified": rng.binomial(1, 0.95, nl),
            "kyb_score": rng.uniform(0.7, 1.0, nl),
            "chargeback_rate_30d": rng.uniform(0.0, 0.01, nl),
            "chargeback_count_7d": rng.integers(0, 2, nl),
            "dispute_rate_90d": rng.uniform(0.0, 0.02, nl),
            "first_chargeback": rng.binomial(1, 0.10, nl),
            "avg_ticket_size": rng.lognormal(3.5, 0.8, nl),
            "ticket_size_std": rng.lognormal(2.5, 0.5, nl),
            "tx_count_1h": rng.integers(1, 50, nl),
            "unique_cards_1h": rng.integers(1, 20, nl),
            "decline_rate_24h": rng.uniform(0.0, 0.05, nl),
            "refund_rate_30d": rng.uniform(0.0, 0.03, nl),
            "refund_count_7d": rng.integers(0, 3, nl),
            "micro_tx_count_1h": rng.integers(0, 1, nl),
            "sequential_card_attempt": rng.binomial(1, 0.01, nl),
            "auth_without_purchase": rng.binomial(1, 0.02, nl),
            "velocity_spike": rng.binomial(1, 0.05, nl),
            "ip_diversity_score": rng.uniform(0.05, 0.30, nl),
            "country_count_24h": rng.integers(1, 3, nl),
            "is_online_merchant": rng.binomial(1, 0.60, nl),
            "shipping_billing_mismatch": rng.uniform(0.0, 0.05, nl),
            "revenue_vs_expected": rng.uniform(0.8, 1.2, nl),
            "refund_to_sales_ratio": rng.uniform(0.0, 0.03, nl),
            "avg_days_to_refund": rng.integers(3, 30, nl),
            "label": np.zeros(nl, int),
        })

        fraud = pd.DataFrame({
            "merchant_age_days": rng.integers(1, 30, nf),         # new account
            "mcc_risk_score": rng.uniform(0.5, 1.0, nf),
            "business_type_verified": rng.binomial(1, 0.10, nf),
            "kyb_score": rng.uniform(0.0, 0.3, nf),
            "chargeback_rate_30d": rng.uniform(0.05, 0.50, nf),
            "chargeback_count_7d": rng.integers(5, 50, nf),
            "dispute_rate_90d": rng.uniform(0.05, 0.40, nf),
            "first_chargeback": rng.binomial(1, 0.70, nf),
            "avg_ticket_size": rng.choice([0.1, 0.5, 1.0, 1.5], nf),  # card testing
            "ticket_size_std": rng.uniform(0.0, 0.5, nf),
            "tx_count_1h": rng.integers(100, 1000, nf),
            "unique_cards_1h": rng.integers(50, 500, nf),
            "decline_rate_24h": rng.uniform(0.20, 0.80, nf),
            "refund_rate_30d": rng.uniform(0.15, 0.80, nf),
            "refund_count_7d": rng.integers(20, 200, nf),
            "micro_tx_count_1h": rng.integers(50, 500, nf),
            "sequential_card_attempt": rng.binomial(1, 0.80, nf),
            "auth_without_purchase": rng.binomial(1, 0.70, nf),
            "velocity_spike": rng.binomial(1, 0.90, nf),
            "ip_diversity_score": rng.uniform(0.60, 1.0, nf),
            "country_count_24h": rng.integers(5, 30, nf),
            "is_online_merchant": rng.binomial(1, 0.95, nf),
            "shipping_billing_mismatch": rng.uniform(0.30, 0.90, nf),
            "revenue_vs_expected": rng.uniform(5.0, 20.0, nf),
            "refund_to_sales_ratio": rng.uniform(0.30, 1.0, nf),
            "avg_days_to_refund": rng.integers(0, 2, nf),
            "label": np.ones(nf, int),
        })

        return pd.concat([legit, fraud]).sample(frac=1, random_state=seed).reset_index(drop=True)


@dataclass
class MerchantResult:
    score: float; label: int; risk_level: str
    action: str; fraud_typology: str
    reason_codes: List[str]; threshold: float
    def __str__(self):
        return (f"score={self.score:.4f} | risk={self.risk_level} | "
                f"action={self.action} | typology={self.fraud_typology} | "
                f"reasons={self.reason_codes}")


class MerchantFraudDetector:
    THRESHOLD = 0.40

    def __init__(self):
        self.fe = MerchantFeatureEngineer()
        self.scaler = StandardScaler()
        self.model = xgb.XGBClassifier(
            n_estimators=350, max_depth=7, learning_rate=0.04,
            scale_pos_weight=19, subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr", random_state=42, n_jobs=-1,
        )

    def fit(self, df, verbose=True):
        X = self.fe.fit_transform(df); y = X.pop("label")
        for f in MODEL_FEATURES:
            if f not in X.columns: X[f] = 0.0
        Xs = self.scaler.fit_transform(X[MODEL_FEATURES])
        Xtr, Xv, ytr, yv = train_test_split(Xs, y, stratify=y, test_size=0.2, random_state=42)
        self.model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        if verbose:
            p = self.model.predict_proba(Xv)[:,1]
            print(f"  ROC-AUC: {roc_auc_score(yv,p):.4f} | AvgP: {average_precision_score(yv,p):.4f}")
        return self

    def _prepare(self, record):
        row = {**RAW_SCHEMA, **record}
        df = self.fe.transform(pd.DataFrame([row]))
        for f in MODEL_FEATURES:
            if f not in df.columns: df[f] = 0.0
        return self.scaler.transform(df[MODEL_FEATURES])

    def _typology(self, record, score):
        if score < self.THRESHOLD: return "legitimate"
        if record.get("micro_tx_count_1h", 0) > 20 or record.get("sequential_card_attempt", 0):
            return "card_testing"
        if record.get("refund_rate_30d", 0) > 0.15: return "refund_abuse"
        if record.get("chargeback_rate_30d", 0) > 0.05: return "chargeback_fraud"
        if record.get("shipping_billing_mismatch", 0) > 0.3: return "triangulation_fraud"
        return "merchant_collusion"

    def _reason_codes(self, record):
        codes = []
        if record.get("micro_tx_count_1h", 0) > 10: codes.append("CARD_TESTING_PATTERN")
        if record.get("sequential_card_attempt"): codes.append("SEQUENTIAL_BIN_ATTACK")
        if record.get("chargeback_rate_30d", 0) > 0.03: codes.append("HIGH_CHARGEBACK_RATE")
        if record.get("refund_rate_30d", 0) > 0.10: codes.append("HIGH_REFUND_RATE")
        if record.get("decline_rate_24h", 0) > 0.20: codes.append("HIGH_DECLINE_RATE")
        if record.get("merchant_age_days", 999) < 30: codes.append("NEW_MERCHANT")
        if not record.get("business_type_verified", 1): codes.append("UNVERIFIED_BUSINESS")
        if record.get("velocity_spike"): codes.append("TRANSACTION_SPIKE")
        if record.get("country_count_24h", 1) > 5: codes.append("MULTI_COUNTRY")
        return codes

    def predict(self, record: Dict) -> MerchantResult:
        ml = float(self.model.predict_proba(self._prepare(record))[0, 1])
        # Hard rules
        if record.get("chargeback_rate_30d", 0) > 0.10: ml = max(ml, 0.85)
        if record.get("sequential_card_attempt") and record.get("micro_tx_count_1h", 0) > 50:
            ml = 1.0
        risk = "HIGH" if ml >= 0.75 else "MEDIUM" if ml >= self.THRESHOLD else "LOW"
        action = {"HIGH": "TERMINATE_MERCHANT", "MEDIUM": "ENHANCED_MONITORING",
                  "LOW": "ALLOW"}[risk]
        return MerchantResult(score=round(ml, 4), label=int(ml >= self.THRESHOLD),
                               risk_level=risk, action=action,
                               fraud_typology=self._typology(record, ml),
                               reason_codes=self._reason_codes(record),
                               threshold=self.THRESHOLD)

    def evaluate(self, df):
        X = self.fe.transform(df); y = X.pop("label")
        for f in MODEL_FEATURES:
            if f not in X.columns: X[f] = 0.0
        p = self.model.predict_proba(self.scaler.transform(X[MODEL_FEATURES]))[:,1]
        pred = (p >= self.THRESHOLD).astype(int)
        return {"roc_auc": roc_auc_score(y,p), "avg_precision": average_precision_score(y,p),
                "report": classification_report(y, pred, digits=4)}


def _s(t): print(f"\n{'═'*60}\n  {t}\n{'═'*60}")

def main():
    _s("Merchant Fraud Detection")
    df = MerchantDataGenerator.generate(n=12_000, fraud_rate=0.05)
    print(f"  Total: {len(df):,} | Fraud: {df['label'].sum():,}")
    det = MerchantFraudDetector(); det.fit(df)
    m = det.evaluate(df)
    print(f"\n  ROC-AUC: {m['roc_auc']:.4f} | AvgP: {m['avg_precision']:.4f}")
    print(m["report"])

    _s("Live Inference")
    cases = [
        ({"merchant_age_days": 5, "micro_tx_count_1h": 300, "sequential_card_attempt": 1,
          "decline_rate_24h": 0.70, "unique_cards_1h": 200, "velocity_spike": 1,
          "business_type_verified": 0, "kyb_score": 0.1, "chargeback_rate_30d": 0.15},
         "Card-testing attack (expected: TERMINATE)"),
        ({"merchant_age_days": 730, "chargeback_rate_30d": 0.005,
          "refund_rate_30d": 0.02, "micro_tx_count_1h": 0,
          "business_type_verified": 1, "kyb_score": 0.9},
         "Legitimate merchant (expected: ALLOW)"),
    ]
    for rec, desc in cases:
        r = det.predict(rec)
        print(f"\n  {desc}\n  {r}")

if __name__ == "__main__": main()