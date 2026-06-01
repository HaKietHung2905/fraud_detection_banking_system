"""
Synthetic Identity Fraud Detection
====================================
File  : src/models/synthetic_identity_detector.py
Covers: Fabricated identities (real SSN + fake info), credit piggybacking,
        bust-out fraud, thin-file manipulation.

Synthetic identity is the fastest-growing bank fraud type.
Fraudsters combine a real SSN (often a child's or deceased person's)
with fabricated name/DOB to build credit over months before cashing out.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve,
)
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb

RAW_SCHEMA: Dict[str, object] = {
    # Identity consistency
    "age_vs_credit_history_gap":   0,      # stated age - credit file age (years)
    "ssn_state_mismatch":          0,      # 1 = SSN issued state ≠ address state
    "ssn_issue_year_anomaly":      0,      # 1 = SSN issued after stated birth year
    "dob_format_inconsistency":    0,      # 1 = DOB varies across bureaus
    "name_ssn_mismatch_count":     0,      # number of names associated with SSN

    # Credit file signals
    "credit_file_age_months":      60,     # how old the credit file is
    "credit_score":                680,
    "derogatory_marks":            0,
    "thin_file":                   0,      # 1 = fewer than 3 tradelines
    "authorized_user_count":       0,      # number of AU accounts (piggybacking)
    "bureau_inquiry_spike":        0,      # 1 = sudden surge in hard inquiries
    "inquiries_last_90d":          1,
    "new_accounts_6m":             0,

    # Address & contact consistency
    "address_history_months":      24,     # months at current address
    "address_changes_2y":          1,      # address changes in 2 years
    "po_box_only":                 0,      # 1 = only PO Box on file
    "phone_is_voip":               0,      # 1 = VoIP number (harder to verify)
    "email_age_days":              180,    # email account age in days

    # Application behavior
    "applications_last_30d":       1,
    "lender_count_6m":             1,      # distinct lenders applied to
    "income_to_credit_request":    0.3,    # ratio of requested credit to stated income
    "employer_unverifiable":       0,      # 1 = employer cannot be verified
    "income_verified":             1,

    # Bureau cross-check
    "bureau_discrepancy_count":    0,      # fields that differ across bureaus
    "no_retail_tradelines":        0,      # 1 = no retail credit history
    "credit_building_pattern":     0,      # 1 = methodical credit-building detected
}

MODEL_FEATURES: List[str] = [
    "age_vs_credit_history_gap", "ssn_state_mismatch", "ssn_issue_year_anomaly",
    "dob_format_inconsistency", "name_ssn_mismatch_count",
    "thin_file", "authorized_user_count", "bureau_inquiry_spike",
    "inquiries_last_90d", "new_accounts_6m", "po_box_only",
    "phone_is_voip", "employer_unverifiable", "bureau_discrepancy_count",
    "no_retail_tradelines", "credit_building_pattern",
    # engineered
    "identity_inconsistency_score",
    "credit_behavior_anomaly",
    "application_velocity",
    "log_credit_file_age",
    "piggybacking_risk",
    "address_instability",
]


class SyntheticIDFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, val in RAW_SCHEMA.items():
            if col not in df.columns: df[col] = val

        df["identity_inconsistency_score"] = (
            df["ssn_state_mismatch"] * 0.25
            + df["ssn_issue_year_anomaly"] * 0.25
            + df["dob_format_inconsistency"] * 0.20
            + df["name_ssn_mismatch_count"].clip(upper=5) / 5 * 0.15
            + df["bureau_discrepancy_count"].clip(upper=5) / 5 * 0.15
        )

        df["credit_behavior_anomaly"] = (
            df["credit_building_pattern"] * 0.35
            + df["no_retail_tradelines"] * 0.25
            + df["thin_file"] * 0.20
            + df["bureau_inquiry_spike"] * 0.20
        )

        df["application_velocity"] = (
            df["applications_last_30d"].clip(upper=20) / 20 * 0.5
            + df["lender_count_6m"].clip(upper=10) / 10 * 0.5
        )

        df["log_credit_file_age"] = np.log1p(df["credit_file_age_months"])

        df["piggybacking_risk"] = (
            df["authorized_user_count"].clip(upper=10) / 10 * 0.7
            + df["thin_file"] * 0.3
        )

        df["address_instability"] = (
            df["address_changes_2y"].clip(upper=6) / 6 * 0.6
            + df["po_box_only"] * 0.4
        )

        return df


class SyntheticIDDataGenerator:
    @staticmethod
    def generate(n=8_000, fraud_rate=0.06, seed=48):
        rng = np.random.default_rng(seed)
        nf = int(n * fraud_rate); nl = n - nf

        legit = pd.DataFrame({
            "age_vs_credit_history_gap": rng.integers(0, 5, nl),
            "ssn_state_mismatch": rng.binomial(1, 0.05, nl),
            "ssn_issue_year_anomaly": rng.binomial(1, 0.02, nl),
            "dob_format_inconsistency": rng.binomial(1, 0.03, nl),
            "name_ssn_mismatch_count": rng.integers(0, 1, nl),
            "credit_file_age_months": rng.integers(24, 360, nl),
            "credit_score": rng.integers(580, 820, nl),
            "derogatory_marks": rng.integers(0, 2, nl),
            "thin_file": rng.binomial(1, 0.10, nl),
            "authorized_user_count": rng.integers(0, 2, nl),
            "bureau_inquiry_spike": rng.binomial(1, 0.05, nl),
            "inquiries_last_90d": rng.integers(0, 3, nl),
            "new_accounts_6m": rng.integers(0, 2, nl),
            "address_history_months": rng.integers(12, 120, nl),
            "address_changes_2y": rng.integers(0, 2, nl),
            "po_box_only": rng.binomial(1, 0.03, nl),
            "phone_is_voip": rng.binomial(1, 0.10, nl),
            "email_age_days": rng.integers(90, 3650, nl),
            "applications_last_30d": rng.integers(0, 2, nl),
            "lender_count_6m": rng.integers(1, 3, nl),
            "income_to_credit_request": rng.uniform(0.1, 0.4, nl),
            "employer_unverifiable": rng.binomial(1, 0.05, nl),
            "income_verified": rng.binomial(1, 0.85, nl),
            "bureau_discrepancy_count": rng.integers(0, 1, nl),
            "no_retail_tradelines": rng.binomial(1, 0.08, nl),
            "credit_building_pattern": rng.binomial(1, 0.05, nl),
            "label": np.zeros(nl, int),
        })

        fraud = pd.DataFrame({
            "age_vs_credit_history_gap": rng.integers(15, 50, nf),  # SSN of minor/deceased
            "ssn_state_mismatch": rng.binomial(1, 0.75, nf),
            "ssn_issue_year_anomaly": rng.binomial(1, 0.80, nf),
            "dob_format_inconsistency": rng.binomial(1, 0.70, nf),
            "name_ssn_mismatch_count": rng.integers(2, 8, nf),
            "credit_file_age_months": rng.integers(6, 36, nf),      # recently built
            "credit_score": rng.integers(620, 720, nf),
            "derogatory_marks": rng.integers(0, 1, nf),
            "thin_file": rng.binomial(1, 0.80, nf),
            "authorized_user_count": rng.integers(3, 10, nf),       # piggybacking
            "bureau_inquiry_spike": rng.binomial(1, 0.80, nf),
            "inquiries_last_90d": rng.integers(8, 25, nf),
            "new_accounts_6m": rng.integers(4, 12, nf),
            "address_history_months": rng.integers(1, 6, nf),
            "address_changes_2y": rng.integers(3, 8, nf),
            "po_box_only": rng.binomial(1, 0.50, nf),
            "phone_is_voip": rng.binomial(1, 0.70, nf),
            "email_age_days": rng.integers(1, 30, nf),
            "applications_last_30d": rng.integers(5, 20, nf),
            "lender_count_6m": rng.integers(5, 15, nf),
            "income_to_credit_request": rng.uniform(2.0, 10.0, nf),
            "employer_unverifiable": rng.binomial(1, 0.85, nf),
            "income_verified": rng.binomial(1, 0.10, nf),
            "bureau_discrepancy_count": rng.integers(3, 8, nf),
            "no_retail_tradelines": rng.binomial(1, 0.85, nf),
            "credit_building_pattern": rng.binomial(1, 0.90, nf),
            "label": np.ones(nf, int),
        })

        return pd.concat([legit, fraud]).sample(frac=1, random_state=seed).reset_index(drop=True)


@dataclass
class SyntheticIDResult:
    score: float; label: int; risk_level: str
    action: str; fraud_pattern: str
    reason_codes: List[str]; threshold: float
    def __str__(self):
        return (f"score={self.score:.4f} | risk={self.risk_level} | "
                f"action={self.action} | pattern={self.fraud_pattern} | "
                f"reasons={self.reason_codes}")


class SyntheticIdentityDetector:
    THRESHOLD = 0.45

    def __init__(self):
        self.fe = SyntheticIDFeatureEngineer()
        self.scaler = RobustScaler()
        self.model = xgb.XGBClassifier(
            n_estimators=350, max_depth=7, learning_rate=0.04,
            scale_pos_weight=15, subsample=0.8, colsample_bytree=0.8,
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

    def _pattern(self, record, score):
        if score < self.THRESHOLD: return "genuine"
        if record.get("age_vs_credit_history_gap", 0) > 10: return "minor_ssn_fraud"
        if record.get("authorized_user_count", 0) > 3: return "credit_piggybacking"
        if record.get("credit_building_pattern", 0): return "bust_out_setup"
        return "fabricated_identity"

    def _reason_codes(self, record):
        codes = []
        if record.get("ssn_issue_year_anomaly"): codes.append("SSN_ISSUE_ANOMALY")
        if record.get("age_vs_credit_history_gap", 0) > 10: codes.append("CREDIT_FILE_TOO_YOUNG")
        if record.get("bureau_inquiry_spike"): codes.append("INQUIRY_SPIKE")
        if record.get("authorized_user_count", 0) > 3: codes.append("EXCESSIVE_PIGGYBACKING")
        if record.get("thin_file") and record.get("applications_last_30d", 0) > 5:
            codes.append("THIN_FILE_HIGH_VELOCITY")
        if record.get("employer_unverifiable"): codes.append("EMPLOYER_UNVERIFIABLE")
        if record.get("bureau_discrepancy_count", 0) > 2: codes.append("BUREAU_DISCREPANCIES")
        if record.get("credit_building_pattern"): codes.append("SYSTEMATIC_CREDIT_BUILDING")
        return codes

    def predict(self, record: Dict) -> SyntheticIDResult:
        ml = float(self.model.predict_proba(self._prepare(record))[0, 1])
        # Hard rule: SSN anomaly + thin file + inquiry spike = definite flag
        if (record.get("ssn_issue_year_anomaly") and record.get("thin_file")
                and record.get("bureau_inquiry_spike")):
            ml = max(ml, 0.95)
        risk = "HIGH" if ml >= 0.75 else "MEDIUM" if ml >= self.THRESHOLD else "LOW"
        action = {"HIGH": "REJECT_AND_REFER", "MEDIUM": "MANUAL_REVIEW", "LOW": "APPROVE"}[risk]
        return SyntheticIDResult(score=round(ml, 4), label=int(ml >= self.THRESHOLD),
                                 risk_level=risk, action=action,
                                 fraud_pattern=self._pattern(record, ml),
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
    _s("Synthetic Identity Fraud Detection")
    df = SyntheticIDDataGenerator.generate(n=10_000, fraud_rate=0.06)
    print(f"  Total: {len(df):,} | Fraud: {df['label'].sum():,}")
    det = SyntheticIdentityDetector(); det.fit(df)
    m = det.evaluate(df)
    print(f"\n  ROC-AUC: {m['roc_auc']:.4f} | AvgP: {m['avg_precision']:.4f}")
    print(m["report"])

    _s("Live Inference")
    cases = [
        ({"age_vs_credit_history_gap": 35, "ssn_issue_year_anomaly": 1,
          "thin_file": 1, "bureau_inquiry_spike": 1, "authorized_user_count": 7,
          "credit_building_pattern": 1, "applications_last_30d": 15,
          "employer_unverifiable": 1, "bureau_discrepancy_count": 5},
         "Synthetic identity — child SSN (expected: REJECT)"),
        ({"age_vs_credit_history_gap": 2, "ssn_issue_year_anomaly": 0,
          "thin_file": 0, "bureau_inquiry_spike": 0, "authorized_user_count": 1,
          "credit_building_pattern": 0, "applications_last_30d": 1,
          "employer_unverifiable": 0, "income_verified": 1},
         "Genuine applicant (expected: APPROVE)"),
    ]
    for rec, desc in cases:
        r = det.predict(rec)
        print(f"\n  {desc}\n  {r}")

if __name__ == "__main__": main()