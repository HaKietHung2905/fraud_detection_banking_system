"""
Wire Transfer / Business Email Compromise (BEC) Fraud Detection
===============================================================
File  : src/models/wire_transfer_detector.py
Covers: CEO fraud, invoice fraud, new beneficiary scams,
        last-minute bank change requests, urgent payment fraud.

BEC typologies
--------------
  CEO Fraud      : attacker impersonates executive demanding urgent wire
  Invoice Fraud  : fake invoice with attacker's bank account
  Vendor Fraud   : attacker intercepts vendor payment, changes account
  Payroll Divert : employee tricks HR into changing direct deposit
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
    average_precision_score, precision_recall_curve,
)
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb

RAW_SCHEMA: Dict[str, object] = {
    # Wire details
    "amount":                      10_000.0,
    "is_international":            0,
    "is_new_beneficiary":          0,       # 1 = first wire to this account
    "beneficiary_account_age_days":365,     # how long this account has existed
    "beneficiary_country_risk":    0.1,

    # Request channel & urgency
    "requested_via_email":         0,       # 1 = payment requested by email
    "urgency_flag":                0,       # 1 = marked urgent / same-day
    "out_of_policy":               0,       # 1 = bypasses normal approval flow
    "after_hours_request":         0,       # 1 = requested outside business hours
    "requestor_email_domain_age":  365,     # domain age in days (typosquat detection)

    # Behavioral signals
    "amount_vs_avg_wire":          1.0,     # ratio vs sender's average wire amount
    "days_since_last_wire":        14,      # days since previous outgoing wire
    "wire_count_30d":              2,       # wires sent in last 30 days
    "beneficiary_change_last_7d":  0,       # 1 = beneficiary bank changed recently
    "invoice_amount_match":        1,       # 1 = amount matches open invoice
    "dual_approval_bypassed":      0,       # 1 = second approver not involved

    # Communication signals
    "email_header_anomaly":        0,       # 1 = reply-to ≠ from address
    "domain_lookalike":            0,       # 1 = slight misspelling in domain
    "thread_hijack":               0,       # 1 = reply to legitimate thread
    "phone_verification_done":     1,       # 1 = called back on known number

    # Historical
    "prior_bec_attempts":          0,       # prior BEC attempts on this entity
    "vendor_relationship_months":  24,      # how long vendor relationship exists
}

MODEL_FEATURES: List[str] = [
    "is_international", "is_new_beneficiary", "beneficiary_country_risk",
    "requested_via_email", "urgency_flag", "out_of_policy",
    "after_hours_request", "beneficiary_change_last_7d",
    "dual_approval_bypassed", "email_header_anomaly", "domain_lookalike",
    "thread_hijack", "prior_bec_attempts",
    # engineered
    "log_amount", "amount_spike", "log_beneficiary_age",
    "domain_age_risk", "communication_risk", "process_risk",
]


class WireFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, val in RAW_SCHEMA.items():
            if col not in df.columns: df[col] = val

        df["log_amount"]         = np.log1p(df["amount"])
        df["amount_spike"]       = (df["amount_vs_avg_wire"] > 3).astype(int)
        df["log_beneficiary_age"]= np.log1p(df["beneficiary_account_age_days"])
        df["domain_age_risk"]    = (df["requestor_email_domain_age"] < 30).astype(int)

        df["communication_risk"] = (
            df["email_header_anomaly"] * 0.35
            + df["domain_lookalike"] * 0.35
            + df["thread_hijack"] * 0.20
            + (1 - df["phone_verification_done"]) * 0.10
        )

        df["process_risk"] = (
            df["urgency_flag"] * 0.30
            + df["out_of_policy"] * 0.25
            + df["dual_approval_bypassed"] * 0.25
            + df["after_hours_request"] * 0.10
            + df["beneficiary_change_last_7d"] * 0.10
        )

        return df


class WireDataGenerator:
    @staticmethod
    def generate(n=10_000, fraud_rate=0.04, seed=47):
        rng = np.random.default_rng(seed)
        nf = int(n * fraud_rate); nl = n - nf

        legit = pd.DataFrame({
            "amount": rng.lognormal(10.5, 1.2, nl),
            "is_international": rng.binomial(1, 0.25, nl),
            "is_new_beneficiary": rng.binomial(1, 0.08, nl),
            "beneficiary_account_age_days": rng.integers(180, 3650, nl),
            "beneficiary_country_risk": rng.uniform(0.0, 0.15, nl),
            "requested_via_email": rng.binomial(1, 0.30, nl),
            "urgency_flag": rng.binomial(1, 0.05, nl),
            "out_of_policy": rng.binomial(1, 0.02, nl),
            "after_hours_request": rng.binomial(1, 0.05, nl),
            "requestor_email_domain_age": rng.integers(180, 3650, nl),
            "amount_vs_avg_wire": rng.uniform(0.5, 2.0, nl),
            "days_since_last_wire": rng.integers(1, 60, nl),
            "wire_count_30d": rng.integers(1, 10, nl),
            "beneficiary_change_last_7d": rng.binomial(1, 0.02, nl),
            "invoice_amount_match": rng.binomial(1, 0.90, nl),
            "dual_approval_bypassed": rng.binomial(1, 0.02, nl),
            "email_header_anomaly": rng.binomial(1, 0.02, nl),
            "domain_lookalike": rng.binomial(1, 0.01, nl),
            "thread_hijack": rng.binomial(1, 0.01, nl),
            "phone_verification_done": rng.binomial(1, 0.85, nl),
            "prior_bec_attempts": rng.integers(0, 1, nl),
            "vendor_relationship_months": rng.integers(12, 120, nl),
            "label": np.zeros(nl, int),
        })

        fraud = pd.DataFrame({
            "amount": rng.lognormal(12.0, 1.0, nf),
            "is_international": rng.binomial(1, 0.80, nf),
            "is_new_beneficiary": rng.binomial(1, 0.90, nf),
            "beneficiary_account_age_days": rng.integers(1, 30, nf),
            "beneficiary_country_risk": rng.uniform(0.5, 1.0, nf),
            "requested_via_email": rng.binomial(1, 0.95, nf),
            "urgency_flag": rng.binomial(1, 0.90, nf),
            "out_of_policy": rng.binomial(1, 0.80, nf),
            "after_hours_request": rng.binomial(1, 0.60, nf),
            "requestor_email_domain_age": rng.integers(1, 14, nf),
            "amount_vs_avg_wire": rng.uniform(3, 20, nf),
            "days_since_last_wire": rng.integers(1, 5, nf),
            "wire_count_30d": rng.integers(1, 3, nf),
            "beneficiary_change_last_7d": rng.binomial(1, 0.80, nf),
            "invoice_amount_match": rng.binomial(1, 0.20, nf),
            "dual_approval_bypassed": rng.binomial(1, 0.70, nf),
            "email_header_anomaly": rng.binomial(1, 0.85, nf),
            "domain_lookalike": rng.binomial(1, 0.75, nf),
            "thread_hijack": rng.binomial(1, 0.60, nf),
            "phone_verification_done": rng.binomial(1, 0.05, nf),
            "prior_bec_attempts": rng.integers(1, 5, nf),
            "vendor_relationship_months": rng.integers(0, 2, nf),
            "label": np.ones(nf, int),
        })

        return pd.concat([legit, fraud]).sample(frac=1, random_state=seed).reset_index(drop=True)


@dataclass
class RuleResult:
    triggered: bool
    rules_fired: List[str] = field(default_factory=list)
    score_boost: float = 0.0

def apply_wire_rules(record: Dict) -> RuleResult:
    fired, boost = [], 0.0
    hard = [
        (record.get("domain_lookalike", 0) and record.get("urgency_flag", 0)
         and record.get("is_new_beneficiary", 0), "HARD:BEC_SIGNATURE"),
        (record.get("dual_approval_bypassed", 0) and record.get("amount", 0) > 50_000,
         "HARD:LARGE_WIRE_NO_DUAL_APPROVAL"),
        (record.get("beneficiary_account_age_days", 999) < 7,
         "HARD:BRAND_NEW_BENEFICIARY"),
    ]
    for cond, name in hard:
        if cond: fired.append(name)

    soft = [
        (record.get("email_header_anomaly", 0), "SOFT:EMAIL_ANOMALY", 0.20),
        (record.get("urgency_flag", 0) and record.get("out_of_policy", 0),
         "SOFT:URGENT_OUT_OF_POLICY", 0.20),
        (record.get("beneficiary_change_last_7d", 0), "SOFT:RECENT_ACCOUNT_CHANGE", 0.15),
        (not record.get("phone_verification_done", 1) and record.get("amount", 0) > 10_000,
         "SOFT:NO_PHONE_VERIFICATION", 0.15),
    ]
    for cond, name, b in soft:
        if cond: fired.append(name); boost += b

    return RuleResult(triggered=any("HARD:" in f for f in fired),
                      rules_fired=fired, score_boost=min(boost, 0.40))


@dataclass
class WireResult:
    score: float; label: int; risk_level: str
    action: str; bec_type: str
    reason_codes: List[str]; rule_result: RuleResult; threshold: float
    def __str__(self):
        return (f"score={self.score:.4f} | risk={self.risk_level} | "
                f"action={self.action} | bec_type={self.bec_type} | "
                f"reasons={self.reason_codes}")


class WireTransferDetector:
    THRESHOLD = 0.40

    def __init__(self):
        self.fe = WireFeatureEngineer()
        self.scaler = StandardScaler()
        self.model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=24, subsample=0.8, colsample_bytree=0.8,
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

    def _bec_type(self, record, score):
        if score < self.THRESHOLD: return "legitimate"
        if record.get("domain_lookalike", 0) or record.get("email_header_anomaly", 0):
            return "ceo_fraud" if record.get("urgency_flag", 0) else "invoice_fraud"
        if record.get("beneficiary_change_last_7d", 0): return "vendor_fraud"
        return "payroll_divert"

    def _reason_codes(self, record):
        codes = []
        if record.get("is_new_beneficiary"): codes.append("NEW_BENEFICIARY")
        if record.get("domain_lookalike"): codes.append("DOMAIN_LOOKALIKE")
        if record.get("email_header_anomaly"): codes.append("EMAIL_HEADER_ANOMALY")
        if record.get("urgency_flag"): codes.append("URGENCY_FLAG")
        if record.get("dual_approval_bypassed"): codes.append("NO_DUAL_APPROVAL")
        if record.get("beneficiary_change_last_7d"): codes.append("RECENT_ACCOUNT_CHANGE")
        if not record.get("phone_verification_done", 1): codes.append("NO_CALLBACK_VERIFICATION")
        return codes

    def predict(self, record: Dict) -> WireResult:
        rule_result = apply_wire_rules(record)
        ml = float(self.model.predict_proba(self._prepare(record))[0, 1])
        final = min(ml + rule_result.score_boost, 1.0)
        if rule_result.triggered: final = 1.0
        risk = "HIGH" if final >= 0.75 else "MEDIUM" if final >= self.THRESHOLD else "LOW"
        action = {"HIGH": "BLOCK_AND_ALERT", "MEDIUM": "HOLD_FOR_REVIEW", "LOW": "ALLOW"}[risk]
        return WireResult(score=round(final, 4), label=int(final >= self.THRESHOLD),
                          risk_level=risk, action=action,
                          bec_type=self._bec_type(record, final),
                          reason_codes=self._reason_codes(record) + rule_result.rules_fired,
                          rule_result=rule_result, threshold=self.THRESHOLD)

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
    _s("Wire Transfer / BEC Fraud Detection")
    df = WireDataGenerator.generate(n=12_000, fraud_rate=0.04)
    print(f"  Total: {len(df):,} | Fraud: {df['label'].sum():,}")
    det = WireTransferDetector(); det.fit(df)
    m = det.evaluate(df)
    print(f"\n  ROC-AUC: {m['roc_auc']:.4f} | AvgP: {m['avg_precision']:.4f}")
    print(m["report"])

    _s("Live Inference")
    cases = [
        ({"amount": 250_000, "is_new_beneficiary": 1, "beneficiary_account_age_days": 3,
          "urgency_flag": 1, "domain_lookalike": 1, "email_header_anomaly": 1,
          "dual_approval_bypassed": 1, "phone_verification_done": 0,
          "requested_via_email": 1, "out_of_policy": 1, "is_international": 1},
         "CEO fraud — urgent wire (expected: BLOCK)"),
        ({"amount": 15_000, "is_new_beneficiary": 0, "urgency_flag": 0,
          "dual_approval_bypassed": 0, "phone_verification_done": 1,
          "is_international": 0, "email_header_anomaly": 0},
         "Legitimate wire (expected: ALLOW)"),
    ]
    for rec, desc in cases:
        r = det.predict(rec)
        print(f"\n  {desc}\n  {r}")

if __name__ == "__main__": main()