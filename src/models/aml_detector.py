"""
Anti-Money Laundering (AML) Detection
======================================
File  : src/models/aml_detector.py
Covers: Structuring/smurfing, layering, placement, integration,
        round-amount patterns, shell company indicators.

Key AML typologies detected
----------------------------
  Structuring  : multiple transactions just below reporting threshold ($10k)
  Smurfing     : many small deposits across accounts to avoid detection
  Layering     : rapid fund movement through multiple accounts
  Integration  : converting laundered funds into legitimate assets
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve,
)
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb

# ── Reporting threshold (USD) — transactions above trigger CTR ────────────
CTR_THRESHOLD = 10_000

RAW_SCHEMA: Dict[str, object] = {
    # Transaction basics
    "amount":                    500.0,
    "is_cash":                   0,       # 1 = cash transaction
    "is_wire":                   0,       # 1 = wire transfer
    "is_crypto":                 0,       # 1 = crypto-related

    # Structuring signals
    "amount_below_ctr_margin":   0.0,     # CTR_THRESHOLD - amount (negative = above)
    "tx_count_48h":              2,       # transactions in last 48 hours
    "tx_sum_48h":                500.0,   # cumulative amount in last 48 hours
    "structured_count_30d":      0,       # prior structuring flags in 30 days
    "round_amount":              0,       # 1 = amount is round number ($5000, $9000)

    # Velocity & layering
    "account_hop_count":         1,       # number of accounts money passed through
    "unique_beneficiaries_7d":   1,       # distinct payees in last 7 days
    "unique_senders_7d":         1,       # distinct payers in last 7 days
    "funds_in_out_ratio":        1.0,     # ratio of inflows to outflows
    "avg_hold_time_hrs":         48.0,    # average hours funds held before moving

    # Geographic risk
    "sender_country_risk":       0.1,     # FATF country risk score [0-1]
    "receiver_country_risk":     0.1,
    "cross_border":              0,       # 1 = international transfer
    "high_risk_jurisdiction":    0,       # 1 = FATF grey/black list country

    # Entity risk
    "is_shell_indicator":        0,       # 1 = entity shows shell company signals
    "pep_involved":              0,       # 1 = politically exposed person
    "adverse_media_score":       0.0,     # negative news score [0-1]
    "kyc_risk_score":            0.1,     # KYC risk rating [0-1]
    "account_age_days":          365,     # account age in days

    # Behavioral baseline
    "deviation_from_profile":    0.05,    # how different from customer's normal [0-1]
    "unusual_business_activity": 0,       # 1 = activity inconsistent with business type
}

MODEL_FEATURES: List[str] = [
    "is_cash", "is_wire", "is_crypto",
    "round_amount", "structured_count_30d",
    "tx_count_48h", "account_hop_count",
    "unique_beneficiaries_7d", "unique_senders_7d",
    "sender_country_risk", "receiver_country_risk",
    "cross_border", "high_risk_jurisdiction",
    "is_shell_indicator", "pep_involved",
    "adverse_media_score", "kyc_risk_score",
    "deviation_from_profile", "unusual_business_activity",
    # engineered
    "log_amount",
    "near_ctr_threshold",
    "layering_score",
    "geographic_risk",
    "entity_risk",
    "velocity_spike",
]


class AMLFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Engineered features
    -------------------
    log_amount        : log1p(amount)
    near_ctr_threshold: 1 if amount in [8,000–9,999] (structuring zone)
    layering_score    : composite of account hops + fast fund movement
    geographic_risk   : combined sender/receiver country risk × cross_border
    entity_risk       : PEP + shell + adverse media composite
    velocity_spike    : tx_count_48h relative to expected baseline
    """
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, val in RAW_SCHEMA.items():
            if col not in df.columns: df[col] = val

        df["log_amount"]         = np.log1p(df["amount"])
        df["near_ctr_threshold"] = df["amount"].between(8_000, 9_999).astype(int)

        df["layering_score"] = (
            np.log1p(df["account_hop_count"]) * 0.5
            + (1 / (df["avg_hold_time_hrs"] + 1)).clip(upper=1) * 0.3
            + np.log1p(df["unique_beneficiaries_7d"]) / 5 * 0.2
        ).clip(0, 1)

        df["geographic_risk"] = (
            (df["sender_country_risk"] + df["receiver_country_risk"]) / 2
            * (1 + df["cross_border"] * 0.5)
            + df["high_risk_jurisdiction"] * 0.3
        ).clip(0, 1)

        df["entity_risk"] = (
            df["pep_involved"] * 0.35
            + df["is_shell_indicator"] * 0.35
            + df["adverse_media_score"] * 0.20
            + df["kyc_risk_score"] * 0.10
        )

        expected_tx = 2.0
        df["velocity_spike"] = (df["tx_count_48h"] / expected_tx).clip(upper=20)

        return df


class AMLDataGenerator:
    @staticmethod
    def generate(n=12_000, fraud_rate=0.03, seed=46):
        rng = np.random.default_rng(seed)
        nf = int(n * fraud_rate); nl = n - nf

        legit = pd.DataFrame({
            "amount": rng.lognormal(6.5, 1.2, nl),
            "is_cash": rng.binomial(1, 0.15, nl),
            "is_wire": rng.binomial(1, 0.20, nl),
            "is_crypto": rng.binomial(1, 0.02, nl),
            "amount_below_ctr_margin": rng.uniform(5000, 50000, nl),
            "tx_count_48h": rng.integers(1, 5, nl),
            "tx_sum_48h": rng.lognormal(6.0, 1.0, nl),
            "structured_count_30d": rng.integers(0, 1, nl),
            "round_amount": rng.binomial(1, 0.15, nl),
            "account_hop_count": rng.integers(1, 3, nl),
            "unique_beneficiaries_7d": rng.integers(1, 5, nl),
            "unique_senders_7d": rng.integers(1, 4, nl),
            "funds_in_out_ratio": rng.uniform(0.8, 1.2, nl),
            "avg_hold_time_hrs": rng.exponential(72, nl),
            "sender_country_risk": rng.uniform(0.0, 0.2, nl),
            "receiver_country_risk": rng.uniform(0.0, 0.2, nl),
            "cross_border": rng.binomial(1, 0.10, nl),
            "high_risk_jurisdiction": rng.binomial(1, 0.02, nl),
            "is_shell_indicator": rng.binomial(1, 0.02, nl),
            "pep_involved": rng.binomial(1, 0.01, nl),
            "adverse_media_score": rng.uniform(0.0, 0.1, nl),
            "kyc_risk_score": rng.uniform(0.0, 0.2, nl),
            "account_age_days": rng.integers(180, 3650, nl),
            "deviation_from_profile": rng.uniform(0.0, 0.15, nl),
            "unusual_business_activity": rng.binomial(1, 0.03, nl),
            "label": np.zeros(nl, int),
        })

        fraud = pd.DataFrame({
            "amount": rng.uniform(8_000, 9_999, nf),   # structuring zone
            "is_cash": rng.binomial(1, 0.70, nf),
            "is_wire": rng.binomial(1, 0.50, nf),
            "is_crypto": rng.binomial(1, 0.30, nf),
            "amount_below_ctr_margin": rng.uniform(1, 2000, nf),
            "tx_count_48h": rng.integers(5, 20, nf),
            "tx_sum_48h": rng.uniform(40_000, 200_000, nf),
            "structured_count_30d": rng.integers(2, 15, nf),
            "round_amount": rng.binomial(1, 0.60, nf),
            "account_hop_count": rng.integers(3, 12, nf),
            "unique_beneficiaries_7d": rng.integers(5, 30, nf),
            "unique_senders_7d": rng.integers(5, 25, nf),
            "funds_in_out_ratio": rng.uniform(0.95, 1.05, nf),
            "avg_hold_time_hrs": rng.exponential(2, nf),
            "sender_country_risk": rng.uniform(0.5, 1.0, nf),
            "receiver_country_risk": rng.uniform(0.4, 1.0, nf),
            "cross_border": rng.binomial(1, 0.80, nf),
            "high_risk_jurisdiction": rng.binomial(1, 0.60, nf),
            "is_shell_indicator": rng.binomial(1, 0.70, nf),
            "pep_involved": rng.binomial(1, 0.30, nf),
            "adverse_media_score": rng.uniform(0.4, 1.0, nf),
            "kyc_risk_score": rng.uniform(0.5, 1.0, nf),
            "account_age_days": rng.integers(7, 90, nf),
            "deviation_from_profile": rng.uniform(0.6, 1.0, nf),
            "unusual_business_activity": rng.binomial(1, 0.80, nf),
            "label": np.ones(nf, int),
        })

        return pd.concat([legit, fraud]).sample(frac=1, random_state=seed).reset_index(drop=True)


@dataclass
class RuleResult:
    triggered: bool
    rules_fired: List[str] = field(default_factory=list)
    score_boost: float = 0.0

def apply_aml_rules(record: Dict) -> RuleResult:
    fired, boost = [], 0.0
    hard = [
        (record.get("amount", 0) >= 8_000 and record.get("amount", 0) < 10_000
         and record.get("structured_count_30d", 0) >= 3,
         "HARD:STRUCTURING_PATTERN"),
        (record.get("pep_involved", 0) and record.get("high_risk_jurisdiction", 0),
         "HARD:PEP_HIGH_RISK_JURISDICTION"),
        (record.get("account_hop_count", 1) >= 8,
         "HARD:EXCESSIVE_LAYERING"),
    ]
    for cond, name in hard:
        if cond: fired.append(name)

    soft = [
        (record.get("near_ctr_threshold", 0), "SOFT:NEAR_CTR_THRESHOLD", 0.25),
        (record.get("is_shell_indicator", 0), "SOFT:SHELL_COMPANY", 0.20),
        (record.get("high_risk_jurisdiction", 0), "SOFT:HIGH_RISK_COUNTRY", 0.15),
        (record.get("round_amount", 0) and record.get("is_cash", 0),
         "SOFT:ROUND_CASH_AMOUNT", 0.15),
        (record.get("unusual_business_activity", 0), "SOFT:UNUSUAL_BUSINESS", 0.10),
    ]
    for cond, name, b in soft:
        if cond: fired.append(name); boost += b

    return RuleResult(triggered=any("HARD:" in f for f in fired),
                      rules_fired=fired, score_boost=min(boost, 0.40))


@dataclass
class AMLResult:
    score: float; label: int; risk_level: str
    action: str; typology: str
    reason_codes: List[str]; rule_result: RuleResult; threshold: float
    def __str__(self):
        return (f"score={self.score:.4f} | risk={self.risk_level} | "
                f"action={self.action} | typology={self.typology} | "
                f"reasons={self.reason_codes}")


class AMLDetector:
    """XGBoost AML detector with typology classification."""
    THRESHOLD = 0.40

    def __init__(self):
        self.fe = AMLFeatureEngineer()
        self.scaler = StandardScaler()
        self.model = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.04,
            scale_pos_weight=32, subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr", random_state=42, n_jobs=-1,
        )

    def fit(self, df: pd.DataFrame, verbose=True):
        X = self.fe.fit_transform(df); y = X.pop("label")
        for f in MODEL_FEATURES:
            if f not in X.columns: X[f] = 0.0
        Xs = self.scaler.fit_transform(X[MODEL_FEATURES])
        X_tr, X_val, y_tr, y_val = train_test_split(Xs, y, stratify=y, test_size=0.2, random_state=42)
        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        if verbose:
            p = self.model.predict_proba(X_val)[:,1]
            print(f"  ROC-AUC: {roc_auc_score(y_val,p):.4f} | AvgP: {average_precision_score(y_val,p):.4f}")
        return self

    def _prepare(self, record):
        row = {**RAW_SCHEMA, **record}
        df = self.fe.transform(pd.DataFrame([row]))
        for f in MODEL_FEATURES:
            if f not in df.columns: df[f] = 0.0
        return self.scaler.transform(df[MODEL_FEATURES])

    def _typology(self, record, score):
        if score < self.THRESHOLD: return "clean"
        if record.get("near_ctr_threshold", 0) or record.get("structured_count_30d", 0) > 2:
            return "structuring"
        if record.get("account_hop_count", 1) >= 5: return "layering"
        if record.get("is_shell_indicator", 0) or record.get("pep_involved", 0): return "integration"
        return "smurfing"

    def _reason_codes(self, record):
        codes = []
        if record.get("near_ctr_threshold"): codes.append("NEAR_CTR_THRESHOLD")
        if record.get("structured_count_30d", 0) > 1: codes.append("REPEAT_STRUCTURING")
        if record.get("account_hop_count", 1) >= 4: codes.append("LAYERING_DETECTED")
        if record.get("high_risk_jurisdiction"): codes.append("HIGH_RISK_COUNTRY")
        if record.get("is_shell_indicator"): codes.append("SHELL_COMPANY")
        if record.get("pep_involved"): codes.append("PEP_INVOLVED")
        if record.get("deviation_from_profile", 0) > 0.5: codes.append("PROFILE_DEVIATION")
        if record.get("unusual_business_activity"): codes.append("UNUSUAL_ACTIVITY")
        return codes

    def predict(self, record: Dict) -> AMLResult:
        enriched = {**record, "near_ctr_threshold": int(8000 <= record.get("amount", 0) < 10000)}
        rule_result = apply_aml_rules(enriched)
        ml_score = float(self.model.predict_proba(self._prepare(enriched))[0, 1])
        final = min(ml_score + rule_result.score_boost, 1.0)
        if rule_result.triggered: final = 1.0
        risk = "HIGH" if final >= 0.75 else "MEDIUM" if final >= self.THRESHOLD else "LOW"
        action = {"HIGH": "SAR_FILING", "MEDIUM": "ENHANCED_DUE_DILIGENCE", "LOW": "ALLOW"}[risk]
        return AMLResult(score=round(final,4), label=int(final>=self.THRESHOLD),
                         risk_level=risk, action=action,
                         typology=self._typology(enriched, final),
                         reason_codes=self._reason_codes(enriched)+rule_result.rules_fired,
                         rule_result=rule_result, threshold=self.THRESHOLD)

    def evaluate(self, df):
        X = self.fe.transform(df); y = X.pop("label")
        for f in MODEL_FEATURES:
            if f not in X.columns: X[f] = 0.0
        p = self.model.predict_proba(self.scaler.transform(X[MODEL_FEATURES]))[:,1]
        pred = (p >= self.THRESHOLD).astype(int)
        return {"roc_auc": roc_auc_score(y,p), "avg_precision": average_precision_score(y,p),
                "report": classification_report(y, pred, digits=4)}

    def feature_importance(self):
        return (pd.DataFrame({"feature": MODEL_FEATURES, "importance": self.model.feature_importances_})
                .sort_values("importance", ascending=False).reset_index(drop=True))


def _section(t): print(f"\n{'═'*60}\n  {t}\n{'═'*60}")

def main():
    _section("AML Detection — Generating Data")
    df = AMLDataGenerator.generate(n=15_000, fraud_rate=0.03)
    print(f"  Total: {len(df):,} | Fraud: {df['label'].sum():,} ({df['label'].mean():.1%})")

    _section("Training AMLDetector")
    det = AMLDetector(); det.fit(df)

    _section("Evaluation")
    m = det.evaluate(df)
    print(f"  ROC-AUC: {m['roc_auc']:.4f} | AvgP: {m['avg_precision']:.4f}")
    print(m["report"])

    _section("Feature Importance")
    print(det.feature_importance().head(8).to_string(index=False))

    _section("Live Inference")
    cases = [
        ({"amount": 9_500, "is_cash": 1, "structured_count_30d": 5,
          "account_hop_count": 8, "high_risk_jurisdiction": 1, "is_shell_indicator": 1,
          "pep_involved": 1, "deviation_from_profile": 0.90, "unusual_business_activity": 1,
          "cross_border": 1, "sender_country_risk": 0.85, "receiver_country_risk": 0.80},
         "Structuring + shell company + PEP (expected: SAR_FILING)"),
        ({"amount": 50_000, "is_wire": 1, "account_hop_count": 9,
          "unique_beneficiaries_7d": 15, "avg_hold_time_hrs": 1.5,
          "high_risk_jurisdiction": 1, "cross_border": 1, "deviation_from_profile": 0.85},
         "Layering — rapid wire hops (expected: SAR_FILING)"),
        ({"amount": 5_000, "is_cash": 0, "structured_count_30d": 0,
          "account_hop_count": 1, "cross_border": 0, "deviation_from_profile": 0.05},
         "Legitimate wire (expected: ALLOW)"),
    ]
    for rec, desc in cases:
        r = det.predict(rec)
        print(f"\n  {desc}")
        print(f"  {r}")

if __name__ == "__main__": main()