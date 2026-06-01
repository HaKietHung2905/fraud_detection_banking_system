"""
Mobile Banking Fraud Detection
================================
File  : src/models/mobile_banking_detector.py
Covers: App takeover, overlay attacks, SIM swap, remote access trojans (RAT),
        jailbreak/root exploitation, and mobile social engineering.
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
    average_precision_score,
)
from sklearn.base import BaseEstimator, TransformerMixin

RAW_SCHEMA: Dict[str, object] = {
    # Device integrity
    "device_rooted_jailbroken":    0,      # 1 = device is rooted/jailbroken
    "emulator_detected":           0,      # 1 = running on emulator (automation)
    "developer_mode_on":           0,
    "screen_overlay_detected":     0,      # 1 = overlay app detected (phishing)
    "accessibility_abuse":         0,      # 1 = accessibility service abused
    "unknown_apk_installed":       0,      # 1 = sideloaded APK present

    # Session signals
    "session_duration_sec":        300,
    "tap_pattern_anomaly":         0.05,   # deviation from normal tapping [0-1]
    "swipe_velocity_anomaly":      0.05,
    "copy_paste_in_password":      0,      # 1 = password was pasted (not typed)
    "multiple_sessions_same_time": 0,      # 1 = concurrent sessions detected
    "app_in_background_during_tx": 0,      # 1 = app moved to background mid-transaction

    # SIM & network
    "sim_swap_recent":             0,      # 1 = SIM changed in last 48h
    "new_sim_first_login":         0,      # 1 = first login after SIM change
    "vpn_proxy_detected":          0,
    "tor_detected":                0,
    "ip_mismatch_gps":             0,      # 1 = IP location ≠ GPS location

    # Transaction behavior
    "amount":                      500.0,
    "new_payee":                   0,
    "payee_added_this_session":    0,      # 1 = payee added and paid in same session
    "tx_within_60s_of_login":      0,      # 1 = transaction immediately after login
    "max_transfer_limit_hit":      0,
    "mfa_otp_auto_filled":         0,      # 1 = OTP was auto-filled (may indicate RAT)

    # Behavioral baseline
    "time_of_day_risk":            0.1,
    "gps_location_change_km":      1.0,
    "device_fingerprint_score":    0.95,   # similarity to known device [0-1]
}

MODEL_FEATURES: List[str] = [
    "device_rooted_jailbroken", "emulator_detected", "developer_mode_on",
    "screen_overlay_detected", "accessibility_abuse", "unknown_apk_installed",
    "tap_pattern_anomaly", "swipe_velocity_anomaly", "copy_paste_in_password",
    "multiple_sessions_same_time", "app_in_background_during_tx",
    "sim_swap_recent", "new_sim_first_login", "vpn_proxy_detected",
    "tor_detected", "ip_mismatch_gps", "new_payee",
    "payee_added_this_session", "tx_within_60s_of_login",
    "max_transfer_limit_hit", "mfa_otp_auto_filled", "time_of_day_risk",
    # engineered
    "log_amount", "device_risk", "session_risk",
    "sim_risk", "behavior_anomaly", "log_session",
]


class MobileBankingFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, val in RAW_SCHEMA.items():
            if col not in df.columns: df[col] = val

        df["log_amount"]  = np.log1p(df["amount"])
        df["log_session"] = np.log1p(df["session_duration_sec"])

        df["device_risk"] = (
            df["device_rooted_jailbroken"] * 0.30
            + df["emulator_detected"] * 0.25
            + df["screen_overlay_detected"] * 0.20
            + df["accessibility_abuse"] * 0.15
            + df["unknown_apk_installed"] * 0.10
        )

        df["session_risk"] = (
            df["multiple_sessions_same_time"] * 0.30
            + df["app_in_background_during_tx"] * 0.25
            + df["payee_added_this_session"] * 0.20
            + df["tx_within_60s_of_login"] * 0.15
            + df["mfa_otp_auto_filled"] * 0.10
        )

        df["sim_risk"] = (
            df["sim_swap_recent"] * 0.55
            + df["new_sim_first_login"] * 0.30
            + df["ip_mismatch_gps"] * 0.15
        )

        df["behavior_anomaly"] = (
            df["tap_pattern_anomaly"] * 0.40
            + df["swipe_velocity_anomaly"] * 0.30
            + (1 - df["device_fingerprint_score"]) * 0.20
            + df["copy_paste_in_password"] * 0.10
        )

        return df


class MobileBankingDataGenerator:
    @staticmethod
    def generate(n=14_000, fraud_rate=0.03, seed=50):
        rng = np.random.default_rng(seed)
        nf = int(n * fraud_rate); nl = n - nf

        legit = pd.DataFrame({
            "device_rooted_jailbroken": rng.binomial(1, 0.05, nl),
            "emulator_detected": rng.binomial(1, 0.01, nl),
            "developer_mode_on": rng.binomial(1, 0.05, nl),
            "screen_overlay_detected": rng.binomial(1, 0.02, nl),
            "accessibility_abuse": rng.binomial(1, 0.01, nl),
            "unknown_apk_installed": rng.binomial(1, 0.03, nl),
            "session_duration_sec": rng.integers(60, 1800, nl),
            "tap_pattern_anomaly": rng.uniform(0.0, 0.15, nl),
            "swipe_velocity_anomaly": rng.uniform(0.0, 0.15, nl),
            "copy_paste_in_password": rng.binomial(1, 0.05, nl),
            "multiple_sessions_same_time": rng.binomial(1, 0.01, nl),
            "app_in_background_during_tx": rng.binomial(1, 0.03, nl),
            "sim_swap_recent": rng.binomial(1, 0.02, nl),
            "new_sim_first_login": rng.binomial(1, 0.02, nl),
            "vpn_proxy_detected": rng.binomial(1, 0.08, nl),
            "tor_detected": rng.binomial(1, 0.01, nl),
            "ip_mismatch_gps": rng.binomial(1, 0.03, nl),
            "amount": rng.lognormal(5.0, 1.2, nl),
            "new_payee": rng.binomial(1, 0.10, nl),
            "payee_added_this_session": rng.binomial(1, 0.03, nl),
            "tx_within_60s_of_login": rng.binomial(1, 0.05, nl),
            "max_transfer_limit_hit": rng.binomial(1, 0.03, nl),
            "mfa_otp_auto_filled": rng.binomial(1, 0.05, nl),
            "time_of_day_risk": rng.uniform(0.0, 0.2, nl),
            "gps_location_change_km": rng.exponential(2, nl),
            "device_fingerprint_score": rng.uniform(0.85, 1.0, nl),
            "label": np.zeros(nl, int),
        })

        fraud = pd.DataFrame({
            "device_rooted_jailbroken": rng.binomial(1, 0.80, nf),
            "emulator_detected": rng.binomial(1, 0.60, nf),
            "developer_mode_on": rng.binomial(1, 0.70, nf),
            "screen_overlay_detected": rng.binomial(1, 0.75, nf),
            "accessibility_abuse": rng.binomial(1, 0.65, nf),
            "unknown_apk_installed": rng.binomial(1, 0.80, nf),
            "session_duration_sec": rng.integers(5, 60, nf),
            "tap_pattern_anomaly": rng.uniform(0.6, 1.0, nf),
            "swipe_velocity_anomaly": rng.uniform(0.6, 1.0, nf),
            "copy_paste_in_password": rng.binomial(1, 0.70, nf),
            "multiple_sessions_same_time": rng.binomial(1, 0.70, nf),
            "app_in_background_during_tx": rng.binomial(1, 0.80, nf),
            "sim_swap_recent": rng.binomial(1, 0.60, nf),
            "new_sim_first_login": rng.binomial(1, 0.55, nf),
            "vpn_proxy_detected": rng.binomial(1, 0.75, nf),
            "tor_detected": rng.binomial(1, 0.40, nf),
            "ip_mismatch_gps": rng.binomial(1, 0.80, nf),
            "amount": rng.lognormal(8.5, 1.0, nf),
            "new_payee": rng.binomial(1, 0.90, nf),
            "payee_added_this_session": rng.binomial(1, 0.85, nf),
            "tx_within_60s_of_login": rng.binomial(1, 0.90, nf),
            "max_transfer_limit_hit": rng.binomial(1, 0.80, nf),
            "mfa_otp_auto_filled": rng.binomial(1, 0.75, nf),
            "time_of_day_risk": rng.uniform(0.5, 1.0, nf),
            "gps_location_change_km": rng.exponential(500, nf),
            "device_fingerprint_score": rng.uniform(0.0, 0.4, nf),
            "label": np.ones(nf, int),
        })

        return pd.concat([legit, fraud]).sample(frac=1, random_state=seed).reset_index(drop=True)


@dataclass
class MobileResult:
    score: float; label: int; risk_level: str
    action: str; attack_vector: str
    reason_codes: List[str]; threshold: float
    def __str__(self):
        return (f"score={self.score:.4f} | risk={self.risk_level} | "
                f"action={self.action} | vector={self.attack_vector} | "
                f"reasons={self.reason_codes}")


class MobileBankingDetector:
    THRESHOLD = 0.40
    ISO_WEIGHT = 0.35; RF_WEIGHT = 0.65

    def __init__(self):
        self.fe = MobileBankingFeatureEngineer()
        self.scaler = StandardScaler()
        self.isoforest = IsolationForest(n_estimators=250, contamination=0.03,
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

    def _attack_vector(self, record, score):
        if score < self.THRESHOLD: return "legitimate"
        if record.get("emulator_detected") or record.get("accessibility_abuse"):
            return "rat_malware"
        if record.get("sim_swap_recent"): return "sim_swap"
        if record.get("screen_overlay_detected"): return "overlay_phishing"
        return "account_takeover"

    def _reason_codes(self, record):
        codes = []
        if record.get("device_rooted_jailbroken"): codes.append("ROOTED_DEVICE")
        if record.get("emulator_detected"): codes.append("EMULATOR_DETECTED")
        if record.get("screen_overlay_detected"): codes.append("OVERLAY_ATTACK")
        if record.get("sim_swap_recent"): codes.append("SIM_SWAP")
        if record.get("multiple_sessions_same_time"): codes.append("CONCURRENT_SESSIONS")
        if record.get("payee_added_this_session"): codes.append("INSTANT_PAYEE_AND_TRANSFER")
        if record.get("tx_within_60s_of_login"): codes.append("IMMEDIATE_TRANSFER")
        if record.get("ip_mismatch_gps"): codes.append("IP_GPS_MISMATCH")
        if record.get("mfa_otp_auto_filled"): codes.append("OTP_AUTO_FILLED")
        return codes

    def predict(self, record: Dict) -> MobileResult:
        Xs = self._prepare(record)
        ml = float(self._score(Xs)[0])
        if record.get("emulator_detected") and record.get("screen_overlay_detected"):
            ml = max(ml, 0.95)
        if record.get("sim_swap_recent") and record.get("max_transfer_limit_hit"):
            ml = max(ml, 0.90)
        risk = "HIGH" if ml >= 0.75 else "MEDIUM" if ml >= self.THRESHOLD else "LOW"
        action = {"HIGH": "BLOCK_AND_LOCK", "MEDIUM": "STEP_UP_AUTH", "LOW": "ALLOW"}[risk]
        return MobileResult(score=round(ml, 4), label=int(ml >= self.THRESHOLD),
                             risk_level=risk, action=action,
                             attack_vector=self._attack_vector(record, ml),
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
    _s("Mobile Banking Fraud Detection")
    df = MobileBankingDataGenerator.generate(n=16_000, fraud_rate=0.03)
    print(f"  Total: {len(df):,} | Fraud: {df['label'].sum():,}")
    det = MobileBankingDetector(); det.fit(df)
    m = det.evaluate(df)
    print(f"\n  ROC-AUC: {m['roc_auc']:.4f} | AvgP: {m['avg_precision']:.4f}")
    print(m["report"])

    _s("Live Inference")
    cases = [
        ({"device_rooted_jailbroken": 1, "emulator_detected": 1, "screen_overlay_detected": 1,
          "amount": 50_000, "payee_added_this_session": 1, "tx_within_60s_of_login": 1,
          "sim_swap_recent": 1, "max_transfer_limit_hit": 1, "mfa_otp_auto_filled": 1,
          "ip_mismatch_gps": 1, "tap_pattern_anomaly": 0.9},
         "RAT + SIM swap + overlay (expected: BLOCK_AND_LOCK)"),
        ({"device_rooted_jailbroken": 0, "emulator_detected": 0, "amount": 200,
          "payee_added_this_session": 0, "sim_swap_recent": 0,
          "device_fingerprint_score": 0.97, "tap_pattern_anomaly": 0.05},
         "Normal mobile transaction (expected: ALLOW)"),
    ]
    for rec, desc in cases:
        r = det.predict(rec)
        print(f"\n  {desc}\n  {r}")

if __name__ == "__main__": main()