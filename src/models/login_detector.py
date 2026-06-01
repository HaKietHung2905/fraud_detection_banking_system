"""
Account Takeover Detection — Login & Session Feature Engineering
================================================================
File  : src/models/login_detector.py
Covers: Device, Network, Geography, Timing, Auth-failure,
        Post-login behaviour, and derived velocity features.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
)
from sklearn.base import BaseEstimator, TransformerMixin


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw Feature Schema
# ─────────────────────────────────────────────────────────────────────────────

# Every login/session event should carry these columns.
# Missing columns are filled with safe defaults during inference.
RAW_SCHEMA: Dict[str, object] = {
    # ── Device & Identity ────────────────────────────────────────
    "is_new_device":             0,     # 1 = device fingerprint never seen before
    "device_fingerprint_score":  1.0,   # similarity to known devices [0–1], 1 = identical
    "is_new_browser":            0,     # browser type changed on a known device
    "os_changed":                0,     # OS mismatch vs. user history

    # ── Network & IP ─────────────────────────────────────────────
    "ip_risk_score":             0.0,   # threat-intel score [0–1]
    "vpn_used":                  0,     # VPN/Tor/proxy detected
    "asn_changed":               0,     # ISP/ASN different from usual
    "ip_country_mismatch":       0,     # IP country ≠ account home country

    # ── Geographic & Velocity ────────────────────────────────────
    "location_change_km":        0.0,   # distance from last login location
    "time_since_last_login_h":   24.0,  # hours since previous successful login
    "new_city":                  0,     # city not seen in last 10 logins
    "country_mismatch":          0,     # registration country ≠ current country

    # ── Timing ───────────────────────────────────────────────────
    "hour":                      12,    # local hour of login (0–23)
    "day_of_week":               2,     # 0=Mon … 6=Sun
    "login_duration_sec":        180.0, # session length so far

    # ── Authentication Failures ──────────────────────────────────
    "failed_attempts":           0,     # failures before this success
    "failed_attempts_last_24h":  0,     # cumulative failures past 24 h
    "lockout_events_7d":         0,     # how many lockouts in past week
    "captcha_failed":            0,     # CAPTCHA challenge failed during login
    "mfa_bypassed":              0,     # MFA skipped or backup code used

    # ── Post-Login Behaviour ─────────────────────────────────────
    "sensitive_action":          0,     # pw/email change, MFA disable, transfer within 5 min
    "password_changed_recent":   0,     # password reset within last 24 h
    "api_call_velocity":         0.0,   # API calls per minute in current session
    "session_anomaly_score":     0.0,   # ML score for abnormal click/nav pattern [0–1]
}

# Features sent to the ML model after engineering
MODEL_FEATURES: List[str] = [
    # raw pass-through
    "is_new_device", "device_fingerprint_score", "is_new_browser", "os_changed",
    "ip_risk_score", "vpn_used", "asn_changed", "ip_country_mismatch",
    "location_change_km", "new_city", "country_mismatch",
    "failed_attempts", "failed_attempts_last_24h", "lockout_events_7d",
    "captcha_failed", "mfa_bypassed",
    "sensitive_action", "password_changed_recent",
    "api_call_velocity", "session_anomaly_score",
    # engineered
    "impossible_travel",
    "is_off_hours",
    "stuffing_score",
    "log_location_change",
    "log_time_since_login",
    "velocity_risk",
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering Transformer
# ─────────────────────────────────────────────────────────────────────────────

class LoginFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Derives high-signal features from raw login/session columns.

    Engineered features
    -------------------
    impossible_travel   : km/h between logins > 900 (faster than plane)
    is_off_hours        : local hour between 00:00–05:59
    stuffing_score      : weighted credential-stuffing risk composite
    log_location_change : log1p(location_change_km) — reduces skew
    log_time_since_login: log1p(time_since_last_login_h)
    velocity_risk       : failed_attempts / (time_since_last_login_h + 0.1)
    """

    # Cap extreme outliers before scaling
    CAPS: Dict[str, float] = {
        "location_change_km":     15_000.0,
        "time_since_last_login_h": 720.0,   # 30 days
        "failed_attempts":         50.0,
        "failed_attempts_last_24h": 100.0,
        "api_call_velocity":       500.0,
        "login_duration_sec":      7_200.0,
    }

    def fit(self, X: pd.DataFrame, y=None) -> "LoginFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # ── Fill any missing raw columns with safe defaults ──────
        for col, default in RAW_SCHEMA.items():
            if col not in df.columns:
                df[col] = default

        # ── Cap outliers ─────────────────────────────────────────
        for col, cap in self.CAPS.items():
            if col in df.columns:
                df[col] = df[col].clip(upper=cap)

        # ── Derived: Impossible travel ────────────────────────────
        # speed km/h between consecutive logins
        speed = df["location_change_km"] / (df["time_since_last_login_h"] + 0.01)
        df["impossible_travel"] = (speed > 900).astype(int)

        # ── Derived: Off-hours flag ───────────────────────────────
        df["is_off_hours"] = df["hour"].between(0, 5).astype(int)

        # ── Derived: Credential-stuffing composite ────────────────
        # High failures + new device + risky IP → likely stuffing
        df["stuffing_score"] = (
            df["failed_attempts_last_24h"].clip(upper=20) / 20 * 0.40
            + df["is_new_device"] * 0.30
            + df["ip_risk_score"] * 0.30
        )

        # ── Derived: Log-transforms for skewed numerics ───────────
        df["log_location_change"]  = np.log1p(df["location_change_km"])
        df["log_time_since_login"] = np.log1p(df["time_since_last_login_h"])

        # ── Derived: Velocity risk ────────────────────────────────
        # Many failures in a short window = brute-force pressure
        df["velocity_risk"] = (
            df["failed_attempts"] / (df["time_since_last_login_h"] + 0.1)
        ).clip(upper=50)

        return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Synthetic Data Generator
# ─────────────────────────────────────────────────────────────────────────────

class LoginDataGenerator:
    """
    Generates realistic synthetic login events.
    Replace with your real auth-log loader for production.
    """

    @staticmethod
    def generate(n_samples: int = 15_000, fraud_rate: float = 0.03,
                 seed: int = 43) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n_fraud = int(n_samples * fraud_rate)
        n_legit = n_samples - n_fraud

        legit = pd.DataFrame({
            "is_new_device":             rng.binomial(1, 0.05,  n_legit),
            "device_fingerprint_score":  rng.uniform(0.85, 1.0, n_legit),
            "is_new_browser":            rng.binomial(1, 0.04,  n_legit),
            "os_changed":                rng.binomial(1, 0.02,  n_legit),
            "ip_risk_score":             rng.uniform(0.0,  0.2, n_legit),
            "vpn_used":                  rng.binomial(1, 0.05,  n_legit),
            "asn_changed":               rng.binomial(1, 0.03,  n_legit),
            "ip_country_mismatch":       rng.binomial(1, 0.02,  n_legit),
            "location_change_km":        rng.exponential(5,     n_legit),
            "time_since_last_login_h":   rng.exponential(24,    n_legit),
            "new_city":                  rng.binomial(1, 0.04,  n_legit),
            "country_mismatch":          rng.binomial(1, 0.02,  n_legit),
            "hour":                      rng.integers(7, 22,    n_legit),
            "day_of_week":               rng.integers(0, 7,     n_legit),
            "login_duration_sec":        rng.exponential(300,   n_legit),
            "failed_attempts":           rng.integers(0, 2,     n_legit),
            "failed_attempts_last_24h":  rng.integers(0, 3,     n_legit),
            "lockout_events_7d":         rng.integers(0, 1,     n_legit),
            "captcha_failed":            rng.binomial(1, 0.03,  n_legit),
            "mfa_bypassed":              rng.binomial(1, 0.02,  n_legit),
            "sensitive_action":          rng.binomial(1, 0.10,  n_legit),
            "password_changed_recent":   rng.binomial(1, 0.03,  n_legit),
            "api_call_velocity":         rng.exponential(5,     n_legit),
            "session_anomaly_score":     rng.uniform(0.0, 0.2,  n_legit),
            "label":                     np.zeros(n_legit, int),
        })

        fraud = pd.DataFrame({
            "is_new_device":             rng.binomial(1, 0.90,  n_fraud),
            "device_fingerprint_score":  rng.uniform(0.0,  0.4, n_fraud),
            "is_new_browser":            rng.binomial(1, 0.70,  n_fraud),
            "os_changed":                rng.binomial(1, 0.50,  n_fraud),
            "ip_risk_score":             rng.uniform(0.6,  1.0, n_fraud),
            "vpn_used":                  rng.binomial(1, 0.70,  n_fraud),
            "asn_changed":               rng.binomial(1, 0.65,  n_fraud),
            "ip_country_mismatch":       rng.binomial(1, 0.75,  n_fraud),
            "location_change_km":        rng.exponential(2000,  n_fraud),
            "time_since_last_login_h":   rng.exponential(0.5,   n_fraud),
            "new_city":                  rng.binomial(1, 0.85,  n_fraud),
            "country_mismatch":          rng.binomial(1, 0.70,  n_fraud),
            "hour":                      rng.choice([0,1,2,3,4,5], n_fraud),
            "day_of_week":               rng.integers(0, 7,     n_fraud),
            "login_duration_sec":        rng.exponential(30,    n_fraud),
            "failed_attempts":           rng.integers(3, 20,    n_fraud),
            "failed_attempts_last_24h":  rng.integers(5, 80,    n_fraud),
            "lockout_events_7d":         rng.integers(1, 5,     n_fraud),
            "captcha_failed":            rng.binomial(1, 0.60,  n_fraud),
            "mfa_bypassed":              rng.binomial(1, 0.55,  n_fraud),
            "sensitive_action":          rng.binomial(1, 0.80,  n_fraud),
            "password_changed_recent":   rng.binomial(1, 0.50,  n_fraud),
            "api_call_velocity":         rng.exponential(80,    n_fraud),
            "session_anomaly_score":     rng.uniform(0.6,  1.0, n_fraud),
            "label":                     np.ones(n_fraud, int),
        })

        return (
            pd.concat([legit, fraud])
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rule-Based Pre-Filter
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    triggered: bool
    rules_fired: List[str] = field(default_factory=list)
    score_boost: float = 0.0   # added directly to final score


def apply_hard_rules(record: Dict) -> RuleResult:
    """
    Deterministic rules that fire BEFORE the ML model.
    A triggered hard rule immediately returns BLOCK regardless of ML score.
    Score boost is added to ML output for soft rules.
    """
    fired: List[str] = []
    boost = 0.0

    # ── Hard rules (deterministic block) ─────────────────────────
    hard_rules = [
        (record.get("impossible_travel", 0) == 1,         "IMPOSSIBLE_TRAVEL"),
        (record.get("ip_risk_score", 0) > 0.95,           "KNOWN_MALICIOUS_IP"),
        (record.get("failed_attempts_last_24h", 0) > 50,  "CREDENTIAL_STUFFING"),
        (record.get("lockout_events_7d", 0) >= 3,         "REPEATED_LOCKOUTS"),
    ]
    for condition, name in hard_rules:
        if condition:
            fired.append(f"HARD:{name}")

    # ── Soft rules (score boost) ──────────────────────────────────
    soft_rules = [
        (record.get("is_new_device", 0) and record.get("sensitive_action", 0),
         "NEW_DEVICE+SENSITIVE_ACTION", 0.25),
        (record.get("vpn_used", 0) and record.get("ip_country_mismatch", 0),
         "VPN+COUNTRY_MISMATCH", 0.15),
        (record.get("mfa_bypassed", 0) and record.get("new_city", 0),
         "MFA_BYPASS+NEW_CITY", 0.20),
        (record.get("captcha_failed", 0) and record.get("failed_attempts", 0) > 3,
         "CAPTCHA_FAIL+BRUTE_FORCE", 0.15),
    ]
    for condition, name, b in soft_rules:
        if condition:
            fired.append(f"SOFT:{name}")
            boost += b

    return RuleResult(
        triggered=any("HARD:" in f for f in fired),
        rules_fired=fired,
        score_boost=min(boost, 0.40),   # cap total boost
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Account Takeover Detector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AtoResult:
    score:         float
    label:         int
    risk_level:    str
    action:        str
    reason_codes:  List[str]
    rule_result:   RuleResult
    threshold:     float

    def __str__(self) -> str:
        return (
            f"score={self.score:.4f} | risk={self.risk_level} | "
            f"action={self.action} | reasons={self.reason_codes}"
        )


class AccountTakeoverDetector:
    """
    Ensemble: Isolation Forest (unsupervised, 40%) + Random Forest (supervised, 60%)
    with a rule-based pre-filter layer.

    Why the ensemble?
    - IsoForest catches novel attack patterns not seen in training data.
    - Random Forest learns labeled historical patterns.
    - Rules provide instant deterministic blocks for obvious cases.
    """

    THRESHOLD      = 0.40
    ISO_WEIGHT     = 0.40
    RF_WEIGHT      = 0.60
    ISO_SHIFT      = 0.05   # IsoForest score centering
    ISO_SCALE      = 0.50   # IsoForest score scaling

    def __init__(self):
        self.fe        = LoginFeatureEngineer()
        self.scaler    = StandardScaler()
        self.isoforest = IsolationForest(
            n_estimators=300,
            contamination=0.03,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        self.rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "AccountTakeoverDetector":
        X = self.fe.fit_transform(df)
        y = X.pop("label")

        X_feat  = X[MODEL_FEATURES]
        X_sc    = self.scaler.fit_transform(X_feat)

        # Isolation Forest trains on all data (unsupervised)
        self.isoforest.fit(X_sc)

        # Random Forest trains on labeled split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sc, y, stratify=y, test_size=0.20, random_state=42
        )
        self.rf.fit(X_tr, y_tr)
        self._trained = True

        if verbose:
            val_proba = self._ensemble_score(X_val)
            val_pred  = (val_proba >= self.THRESHOLD).astype(int)
            print("── Validation (hold-out 20%) ──")
            print(f"  ROC-AUC        : {roc_auc_score(y_val, val_proba):.4f}")
            print(f"  Avg Precision  : {average_precision_score(y_val, val_proba):.4f}")
            print(classification_report(y_val, val_pred, digits=4))

        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _ensemble_score(self, X_sc: np.ndarray) -> np.ndarray:
        """Combine IsoForest anomaly score + RF class probability."""
        iso_raw  = -self.isoforest.score_samples(X_sc)            # higher = more anomalous
        iso_norm = np.clip((iso_raw - self.ISO_SHIFT) / self.ISO_SCALE, 0, 1)
        rf_prob  = self.rf.predict_proba(X_sc)[:, 1]
        return self.ISO_WEIGHT * iso_norm + self.RF_WEIGHT * rf_prob

    def _prepare_single(self, record: Dict) -> np.ndarray:
        row = {**RAW_SCHEMA, **record}   # fill missing with defaults
        df  = self.fe.transform(pd.DataFrame([row]))
        for f in MODEL_FEATURES:
            if f not in df.columns:
                df[f] = 0.0
        return self.scaler.transform(df[MODEL_FEATURES])

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, record: Dict) -> AtoResult:
        """Full prediction pipeline: rules → ML → final decision."""
        # Step 1: derive impossible_travel for rule check
        enriched = self._enrich_for_rules(record)

        # Step 2: hard rules
        rule_result = apply_hard_rules(enriched)

        # Step 3: ML score
        X_sc     = self._prepare_single(enriched)
        ml_score = float(self._ensemble_score(X_sc)[0])

        # Step 4: apply soft-rule boost, then hard-rule override
        final_score = min(ml_score + rule_result.score_boost, 1.0)
        if rule_result.triggered:
            final_score = 1.0

        label      = int(final_score >= self.THRESHOLD)
        risk_level = self._risk_level(final_score)
        action     = {"HIGH": "BLOCK", "MEDIUM": "REVIEW", "LOW": "ALLOW"}[risk_level]
        reasons    = self._reason_codes(enriched) + rule_result.rules_fired

        return AtoResult(
            score=round(final_score, 4),
            label=label,
            risk_level=risk_level,
            action=action,
            reason_codes=reasons,
            rule_result=rule_result,
            threshold=self.THRESHOLD,
        )

    @staticmethod
    def _enrich_for_rules(record: Dict) -> Dict:
        """Compute impossible_travel before passing to rule engine."""
        r = dict(record)
        dist = r.get("location_change_km", 0)
        hrs  = r.get("time_since_last_login_h", 24) + 0.01
        r["impossible_travel"] = int(dist / hrs > 900)
        return r

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.75:   return "HIGH"
        elif score >= 0.40: return "MEDIUM"
        else:               return "LOW"

    @staticmethod
    def _reason_codes(record: Dict) -> List[str]:
        codes = []
        if record.get("impossible_travel"):                       codes.append("IMPOSSIBLE_TRAVEL")
        if record.get("is_new_device"):                          codes.append("NEW_DEVICE")
        if record.get("ip_risk_score", 0) > 0.5:                codes.append("RISKY_IP")
        if record.get("vpn_used"):                               codes.append("VPN_DETECTED")
        if record.get("failed_attempts", 0) > 3:                codes.append("MULTIPLE_FAILED_LOGINS")
        if record.get("failed_attempts_last_24h", 0) > 10:      codes.append("HIGH_24H_FAILURES")
        if record.get("mfa_bypassed"):                           codes.append("MFA_BYPASSED")
        if record.get("sensitive_action"):                       codes.append("IMMEDIATE_SENSITIVE_ACTION")
        if record.get("session_anomaly_score", 0) > 0.6:        codes.append("ABNORMAL_SESSION_BEHAVIOUR")
        if record.get("asn_changed"):                            codes.append("ISP_CHANGED")
        if record.get("ip_country_mismatch"):                    codes.append("IP_COUNTRY_MISMATCH")
        if record.get("api_call_velocity", 0) > 60:             codes.append("HIGH_API_VELOCITY")
        return codes

    # ── Batch Evaluation ──────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> Dict:
        X = self.fe.transform(df)
        y_true = X.pop("label")
        for f in MODEL_FEATURES:
            if f not in X.columns:
                X[f] = 0.0
        X_sc   = self.scaler.transform(X[MODEL_FEATURES])
        proba  = self._ensemble_score(X_sc)
        y_pred = (proba >= self.THRESHOLD).astype(int)
        return {
            "roc_auc":       roc_auc_score(y_true, proba),
            "avg_precision": average_precision_score(y_true, proba),
            "confusion":     confusion_matrix(y_true, y_pred).tolist(),
            "report":        classification_report(y_true, y_pred, digits=4),
        }

    def feature_importance(self) -> pd.DataFrame:
        """Random Forest feature importance ranked."""
        return (
            pd.DataFrame({
                "feature":   MODEL_FEATURES,
                "importance": self.rf.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def tune_threshold(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns precision/recall/F1 across thresholds.
        Use this to pick the operating threshold for your recall/precision target.
        """
        from sklearn.metrics import precision_recall_curve
        X = self.fe.transform(df)
        y_true = X.pop("label")
        for f in MODEL_FEATURES:
            if f not in X.columns:
                X[f] = 0.0
        proba = self._ensemble_score(self.scaler.transform(X[MODEL_FEATURES]))
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        return pd.DataFrame({
            "threshold": np.append(thresholds, 1.0),
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
        })


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main — Train + Evaluate + Demo
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str):
    print(f"\n{'═'*64}\n  {title}\n{'═'*64}")


def main():
    _section("Generating Synthetic Login Data")
    df = LoginDataGenerator.generate(n_samples=20_000, fraud_rate=0.03)
    n_fraud = df["label"].sum()
    print(f"  Total   : {len(df):,} events")
    print(f"  Fraud   : {n_fraud:,} ({n_fraud/len(df):.1%})")
    print(f"  Features: {len(RAW_SCHEMA)} raw → {len(MODEL_FEATURES)} model features")

    _section("Training AccountTakeoverDetector")
    detector = AccountTakeoverDetector()
    detector.fit(df)

    _section("Full Dataset Evaluation")
    metrics = detector.evaluate(df)
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision  : {metrics['avg_precision']:.4f}")
    print(f"  Confusion Matrix: {metrics['confusion']}")
    print(metrics["report"])

    _section("Feature Importance (top 10)")
    print(detector.feature_importance().head(10).to_string(index=False))

    _section("Threshold Tuning — pick your operating point")
    tdf = detector.tune_threshold(df)
    # Show thresholds with F1 > 0.5
    good = tdf[tdf["f1"] > 0.50].sort_values("f1", ascending=False).head(5)
    print(good.to_string(index=False))

    _section("Live Inference Demo")

    cases = [
        ({
            "is_new_device": 1, "device_fingerprint_score": 0.1,
            "ip_risk_score": 0.92, "vpn_used": 1,
            "location_change_km": 9_000, "time_since_last_login_h": 0.3,
            "failed_attempts": 15, "failed_attempts_last_24h": 60,
            "mfa_bypassed": 1, "sensitive_action": 1,
            "hour": 3, "session_anomaly_score": 0.85,
        }, "Classic ATO — new device + impossible travel + credential stuffing"),

        ({
            "is_new_device": 1, "device_fingerprint_score": 0.6,
            "ip_risk_score": 0.30, "vpn_used": 0,
            "location_change_km": 250, "time_since_last_login_h": 48,
            "failed_attempts": 1, "failed_attempts_last_24h": 2,
            "mfa_bypassed": 0, "sensitive_action": 0,
            "hour": 10, "session_anomaly_score": 0.2,
        }, "Borderline — new device, travel, no other signals"),

        ({
            "is_new_device": 0, "device_fingerprint_score": 0.97,
            "ip_risk_score": 0.05, "vpn_used": 0,
            "location_change_km": 3, "time_since_last_login_h": 20,
            "failed_attempts": 0, "failed_attempts_last_24h": 0,
            "mfa_bypassed": 0, "sensitive_action": 0,
            "hour": 9, "session_anomaly_score": 0.05,
        }, "Legitimate — known device, home location, clean history"),

        ({
            "is_new_device": 1, "device_fingerprint_score": 0.2,
            "ip_risk_score": 0.88, "vpn_used": 1,
            "location_change_km": 500, "time_since_last_login_h": 2,
            "failed_attempts": 0, "failed_attempts_last_24h": 55,
            "mfa_bypassed": 0, "sensitive_action": 1,
            "hour": 2, "session_anomaly_score": 0.7,
            "ip_country_mismatch": 1, "asn_changed": 1,
        }, "Credential stuffing + session takeover"),
    ]

    for record, description in cases:
        result = detector.predict(record)
        print(f"\n  Case   : {description}")
        print(f"  Result : {result}")


if __name__ == "__main__":
    main()