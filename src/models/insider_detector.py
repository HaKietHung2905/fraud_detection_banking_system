"""
Insider Risk Detection — Employee Behavioral Feature Engineering
================================================================
File  : src/models/insider_detector.py
Covers: Data exfiltration, privilege abuse, policy violations,
        off-hours access, anomalous system behaviour.

Architecture
------------
  Raw activity logs
      │
      ▼
  InsiderFeatureEngineer   ← derives 14 behavioural signals
      │
      ├─► apply_hard_rules()   ← deterministic blocks (USB + exfil, etc.)
      │
      └─► Ensemble
            ├── IsolationForest  (50%) — unsupervised; catches novel threats
            └── RandomForest     (50%) — supervised on labelled incidents
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
    precision_recall_curve,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw Feature Schema
# ─────────────────────────────────────────────────────────────────────────────

# All 28 raw fields an employee activity event should carry.
# Missing fields are filled with safe defaults during inference.
RAW_SCHEMA: Dict[str, object] = {

    # ── Access Timing ─────────────────────────────────────────────
    "access_outside_hours":      0,     # 1 = accessed system outside 07:00–20:00
    "weekend_access":            0,     # 1 = accessed on Saturday or Sunday
    "login_hour":                10,    # local hour of first login (0–23)
    "session_count_today":       1,     # number of sessions opened today

    # ── Data Volume & Movement ───────────────────────────────────
    "data_volume_mb":            10.0,  # MB read/written in session
    "download_count":            2,     # number of file downloads
    "upload_count":              0,     # number of file uploads to external
    "print_count":               0,     # pages printed
    "copy_to_clipboard_count":   0,     # clipboard copy events

    # ── Sensitive Resource Access ─────────────────────────────────
    "sensitive_records_accessed":0,     # count of PII/classified records opened
    "unique_tables_accessed":    2,     # distinct DB tables or file dirs touched
    "privileged_cmd_count":      0,     # sudo / admin commands executed
    "failed_access_attempts":    0,     # permission-denied events

    # ── System & Network Behaviour ────────────────────────────────
    "unique_systems_accessed":   2,     # distinct hosts/apps accessed
    "usb_used":                  0,     # USB storage device plugged in
    "cloud_upload_mb":           0.0,   # MB uploaded to personal cloud (GDrive, Dropbox)
    "remote_desktop_used":       0,     # RDP/TeamViewer session detected
    "new_software_installed":    0,     # unapproved software install event

    # ── Communication ─────────────────────────────────────────────
    "email_external_count":      1,     # emails sent to external domains
    "email_attachment_mb":       0.0,   # total attachment size sent externally
    "chat_external":             0,     # messages sent to external chat (Slack, Teams)

    # ── Behavioural Baseline ─────────────────────────────────────
    "deviation_from_baseline":   0.05,  # ML score vs. personal 30-day baseline [0–1]
    "peer_deviation_score":      0.05,  # deviation vs. peer group [0–1]
    "days_before_resignation":   999,   # 0–30 if employee gave notice (else 999)

    # ── HR & Context Signals ─────────────────────────────────────
    "recent_disciplinary":       0,     # disciplinary action in last 90 days
    "access_level_changed":      0,     # privilege escalation/change in last 7 days
    "terminated_account_active": 0,     # account still active post-termination
}

# Features passed to the ML model after engineering
MODEL_FEATURES: List[str] = [
    # raw pass-through (selected subset)
    "access_outside_hours", "weekend_access",
    "data_volume_mb", "download_count", "upload_count",
    "print_count", "copy_to_clipboard_count",
    "sensitive_records_accessed", "unique_tables_accessed",
    "privileged_cmd_count", "failed_access_attempts",
    "unique_systems_accessed", "usb_used", "cloud_upload_mb",
    "remote_desktop_used", "new_software_installed",
    "email_external_count", "email_attachment_mb", "chat_external",
    "deviation_from_baseline", "peer_deviation_score",
    "recent_disciplinary", "access_level_changed",
    "terminated_account_active",
    # engineered
    "log_data_volume",
    "log_download_count",
    "log_sensitive_records",
    "log_email_attachment",
    "log_cloud_upload",
    "exfil_composite",
    "access_anomaly",
    "time_risk",
    "resignation_risk",
    "behaviour_risk",
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering Transformer
# ─────────────────────────────────────────────────────────────────────────────

class InsiderFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Derives 10 high-signal composite features from raw employee activity logs.

    Engineered features
    -------------------
    log_data_volume       : log1p(data_volume_mb)          — reduce heavy-tail skew
    log_download_count    : log1p(download_count)
    log_sensitive_records : log1p(sensitive_records_accessed)
    log_email_attachment  : log1p(email_attachment_mb)
    log_cloud_upload      : log1p(cloud_upload_mb)

    exfil_composite       : weighted sum of exfiltration channels
                            (download + usb + cloud + email attachments + print)
    access_anomaly        : breadth of unusual resource access
                            (unique systems + tables + privileged commands)
    time_risk             : off-hours + weekend weighted by session count
    resignation_risk      : elevated score when notice period is imminent
    behaviour_risk        : personal + peer deviation composite
    """

    CAPS: Dict[str, float] = {
        "data_volume_mb":            50_000.0,
        "download_count":            2_000.0,
        "upload_count":              500.0,
        "print_count":               500.0,
        "sensitive_records_accessed":5_000.0,
        "unique_tables_accessed":    200.0,
        "privileged_cmd_count":      500.0,
        "unique_systems_accessed":   50.0,
        "cloud_upload_mb":           10_000.0,
        "email_external_count":      500.0,
        "email_attachment_mb":       2_000.0,
        "copy_to_clipboard_count":   1_000.0,
    }

    def fit(self, X: pd.DataFrame, y=None) -> "InsiderFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Fill missing raw columns with defaults
        for col, default in RAW_SCHEMA.items():
            if col not in df.columns:
                df[col] = default

        # Cap outliers
        for col, cap in self.CAPS.items():
            if col in df.columns:
                df[col] = df[col].clip(upper=cap)

        # ── Log-transforms ────────────────────────────────────────
        df["log_data_volume"]       = np.log1p(df["data_volume_mb"])
        df["log_download_count"]    = np.log1p(df["download_count"])
        df["log_sensitive_records"] = np.log1p(df["sensitive_records_accessed"])
        df["log_email_attachment"]  = np.log1p(df["email_attachment_mb"])
        df["log_cloud_upload"]      = np.log1p(df["cloud_upload_mb"])

        # ── Exfiltration composite ────────────────────────────────
        # Weighted sum of all channels an insider could use to remove data.
        # USB and cloud upload are weighted highest (harder to detect).
        df["exfil_composite"] = (
            df["log_download_count"]               * 0.20
            + df["usb_used"]                       * 0.25
            + df["log_cloud_upload"]               * 0.20
            + df["log_email_attachment"]            * 0.15
            + np.log1p(df["print_count"])          * 0.10
            + np.log1p(df["copy_to_clipboard_count"]) * 0.10
        )

        # ── Access anomaly ────────────────────────────────────────
        # How broadly did this employee reach across systems?
        df["access_anomaly"] = (
            np.log1p(df["unique_systems_accessed"]) * 0.40
            + np.log1p(df["unique_tables_accessed"]) * 0.30
            + np.log1p(df["privileged_cmd_count"])   * 0.20
            + np.log1p(df["failed_access_attempts"]) * 0.10
        )

        # ── Time risk ─────────────────────────────────────────────
        # Off-hours sessions multiplied by how many sessions (persistent threat)
        df["time_risk"] = (
            df["access_outside_hours"] * 0.50
            + df["weekend_access"]     * 0.30
            + np.log1p(df["session_count_today"]) * 0.20
        )

        # ── Resignation risk ──────────────────────────────────────
        # Employees who have given notice often exfiltrate in final 30 days.
        # days_before_resignation = 999 means no notice given.
        df["resignation_risk"] = np.where(
            df["days_before_resignation"] <= 30,
            1.0 - df["days_before_resignation"] / 30,   # 1.0 on last day, 0 at 30 days out
            0.0,
        )

        # ── Behaviour risk ────────────────────────────────────────
        # Combines personal baseline deviation and peer-group deviation.
        df["behaviour_risk"] = (
            df["deviation_from_baseline"] * 0.60
            + df["peer_deviation_score"]  * 0.40
        )

        return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Synthetic Data Generator
# ─────────────────────────────────────────────────────────────────────────────

class InsiderDataGenerator:
    """
    Generates three insider threat archetypes:
      - Data thief        : large exfil volume, USB, cloud upload
      - Privilege abuser  : admin commands, broad system access
      - Disgruntled leaver: resignation flag, gradual data drain
    """

    @staticmethod
    def generate(
        n_samples: int = 10_000,
        fraud_rate: float = 0.02,
        seed: int = 45,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n_fraud = int(n_samples * fraud_rate)
        n_legit = n_samples - n_fraud

        # ── Legitimate employees ──────────────────────────────────
        legit = pd.DataFrame({
            "access_outside_hours":       rng.binomial(1, 0.05,  n_legit),
            "weekend_access":             rng.binomial(1, 0.08,  n_legit),
            "login_hour":                 rng.integers(7, 19,    n_legit),
            "session_count_today":        rng.integers(1, 4,     n_legit),
            "data_volume_mb":             rng.lognormal(2.5, 1.0, n_legit),
            "download_count":             rng.integers(0, 15,    n_legit),
            "upload_count":               rng.integers(0, 3,     n_legit),
            "print_count":                rng.integers(0, 10,    n_legit),
            "copy_to_clipboard_count":    rng.integers(0, 20,    n_legit),
            "sensitive_records_accessed": rng.integers(0, 15,    n_legit),
            "unique_tables_accessed":     rng.integers(1, 6,     n_legit),
            "privileged_cmd_count":       rng.integers(0, 3,     n_legit),
            "failed_access_attempts":     rng.integers(0, 2,     n_legit),
            "unique_systems_accessed":    rng.integers(1, 5,     n_legit),
            "usb_used":                   rng.binomial(1, 0.02,  n_legit),
            "cloud_upload_mb":            rng.exponential(2.0,   n_legit),
            "remote_desktop_used":        rng.binomial(1, 0.05,  n_legit),
            "new_software_installed":     rng.binomial(1, 0.01,  n_legit),
            "email_external_count":       rng.integers(0, 8,     n_legit),
            "email_attachment_mb":        rng.exponential(3.0,   n_legit),
            "chat_external":              rng.binomial(1, 0.10,  n_legit),
            "deviation_from_baseline":    rng.uniform(0.0, 0.20, n_legit),
            "peer_deviation_score":       rng.uniform(0.0, 0.20, n_legit),
            "days_before_resignation":    np.full(n_legit, 999),
            "recent_disciplinary":        rng.binomial(1, 0.02,  n_legit),
            "access_level_changed":       rng.binomial(1, 0.03,  n_legit),
            "terminated_account_active":  np.zeros(n_legit, int),
            "label":                      np.zeros(n_legit, int),
        })

        # ── Insider threats (3 archetypes, equal split) ───────────
        n_each = n_fraud // 3
        n_rem  = n_fraud - n_each * 3

        # Archetype 1: Data thief — high exfil, USB, cloud upload
        thief = pd.DataFrame({
            "access_outside_hours":       rng.binomial(1, 0.70,  n_each),
            "weekend_access":             rng.binomial(1, 0.60,  n_each),
            "login_hour":                 rng.choice([0,1,2,3,4,5,21,22,23], n_each),
            "session_count_today":        rng.integers(3, 10,    n_each),
            "data_volume_mb":             rng.lognormal(6.0, 1.5, n_each),
            "download_count":             rng.integers(50, 500,  n_each),
            "upload_count":               rng.integers(10, 100,  n_each),
            "print_count":                rng.integers(20, 200,  n_each),
            "copy_to_clipboard_count":    rng.integers(50, 500,  n_each),
            "sensitive_records_accessed": rng.integers(100, 1000,n_each),
            "unique_tables_accessed":     rng.integers(10, 50,   n_each),
            "privileged_cmd_count":       rng.integers(0, 5,     n_each),
            "failed_access_attempts":     rng.integers(0, 3,     n_each),
            "unique_systems_accessed":    rng.integers(3, 10,    n_each),
            "usb_used":                   rng.binomial(1, 0.80,  n_each),
            "cloud_upload_mb":            rng.lognormal(5.0, 1.5, n_each),
            "remote_desktop_used":        rng.binomial(1, 0.20,  n_each),
            "new_software_installed":     rng.binomial(1, 0.10,  n_each),
            "email_external_count":       rng.integers(20, 200,  n_each),
            "email_attachment_mb":        rng.lognormal(4.5, 1.0, n_each),
            "chat_external":              rng.binomial(1, 0.50,  n_each),
            "deviation_from_baseline":    rng.uniform(0.60, 1.0, n_each),
            "peer_deviation_score":       rng.uniform(0.50, 1.0, n_each),
            "days_before_resignation":    np.full(n_each, 999),
            "recent_disciplinary":        rng.binomial(1, 0.10,  n_each),
            "access_level_changed":       rng.binomial(1, 0.10,  n_each),
            "terminated_account_active":  np.zeros(n_each, int),
            "label":                      np.ones(n_each, int),
        })

        # Archetype 2: Privilege abuser — admin commands, broad access
        abuser = pd.DataFrame({
            "access_outside_hours":       rng.binomial(1, 0.40,  n_each),
            "weekend_access":             rng.binomial(1, 0.35,  n_each),
            "login_hour":                 rng.integers(0, 24,    n_each),
            "session_count_today":        rng.integers(2, 8,     n_each),
            "data_volume_mb":             rng.lognormal(3.5, 1.2, n_each),
            "download_count":             rng.integers(5, 40,    n_each),
            "upload_count":               rng.integers(0, 5,     n_each),
            "print_count":                rng.integers(0, 20,    n_each),
            "copy_to_clipboard_count":    rng.integers(5, 50,    n_each),
            "sensitive_records_accessed": rng.integers(50, 500,  n_each),
            "unique_tables_accessed":     rng.integers(20, 100,  n_each),
            "privileged_cmd_count":       rng.integers(50, 500,  n_each),
            "failed_access_attempts":     rng.integers(5, 50,    n_each),
            "unique_systems_accessed":    rng.integers(10, 30,   n_each),
            "usb_used":                   rng.binomial(1, 0.20,  n_each),
            "cloud_upload_mb":            rng.exponential(10.0,  n_each),
            "remote_desktop_used":        rng.binomial(1, 0.60,  n_each),
            "new_software_installed":     rng.binomial(1, 0.40,  n_each),
            "email_external_count":       rng.integers(2, 20,    n_each),
            "email_attachment_mb":        rng.exponential(5.0,   n_each),
            "chat_external":              rng.binomial(1, 0.20,  n_each),
            "deviation_from_baseline":    rng.uniform(0.50, 1.0, n_each),
            "peer_deviation_score":       rng.uniform(0.60, 1.0, n_each),
            "days_before_resignation":    np.full(n_each, 999),
            "recent_disciplinary":        rng.binomial(1, 0.30,  n_each),
            "access_level_changed":       rng.binomial(1, 0.50,  n_each),
            "terminated_account_active":  np.zeros(n_each, int),
            "label":                      np.ones(n_each, int),
        })

        # Archetype 3: Disgruntled leaver — resignation window, gradual drain
        n_leaver = n_each + n_rem
        leaver = pd.DataFrame({
            "access_outside_hours":       rng.binomial(1, 0.50,  n_leaver),
            "weekend_access":             rng.binomial(1, 0.40,  n_leaver),
            "login_hour":                 rng.integers(0, 24,    n_leaver),
            "session_count_today":        rng.integers(2, 6,     n_leaver),
            "data_volume_mb":             rng.lognormal(5.0, 1.0, n_leaver),
            "download_count":             rng.integers(30, 300,  n_leaver),
            "upload_count":               rng.integers(5, 50,    n_leaver),
            "print_count":                rng.integers(10, 100,  n_leaver),
            "copy_to_clipboard_count":    rng.integers(20, 200,  n_leaver),
            "sensitive_records_accessed": rng.integers(30, 300,  n_leaver),
            "unique_tables_accessed":     rng.integers(5, 30,    n_leaver),
            "privileged_cmd_count":       rng.integers(0, 10,    n_leaver),
            "failed_access_attempts":     rng.integers(0, 5,     n_leaver),
            "unique_systems_accessed":    rng.integers(3, 12,    n_leaver),
            "usb_used":                   rng.binomial(1, 0.60,  n_leaver),
            "cloud_upload_mb":            rng.lognormal(4.0, 1.2, n_leaver),
            "remote_desktop_used":        rng.binomial(1, 0.15,  n_leaver),
            "new_software_installed":     rng.binomial(1, 0.05,  n_leaver),
            "email_external_count":       rng.integers(10, 100,  n_leaver),
            "email_attachment_mb":        rng.lognormal(3.5, 1.0, n_leaver),
            "chat_external":              rng.binomial(1, 0.40,  n_leaver),
            "deviation_from_baseline":    rng.uniform(0.55, 1.0, n_leaver),
            "peer_deviation_score":       rng.uniform(0.45, 1.0, n_leaver),
            "days_before_resignation":    rng.integers(0, 30,    n_leaver),
            "recent_disciplinary":        rng.binomial(1, 0.20,  n_leaver),
            "access_level_changed":       rng.binomial(1, 0.15,  n_leaver),
            "terminated_account_active":  rng.binomial(1, 0.30,  n_leaver),
            "label":                      np.ones(n_leaver, int),
        })

        return (
            pd.concat([legit, thief, abuser, leaver])
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
    Deterministic rules that fire BEFORE the ML model.
    Hard rules → immediate BLOCK regardless of ML score.
    Soft rules → score boost fed into final decision.
    """
    fired: List[str] = []
    boost = 0.0

    # ── Hard rules ────────────────────────────────────────────────
    hard = [
        (record.get("terminated_account_active", 0) == 1,
         "TERMINATED_ACCOUNT_ACTIVE"),

        (record.get("usb_used", 0) == 1
         and record.get("data_volume_mb", 0) > 1_000,
         "USB_LARGE_EXFIL"),

        (record.get("cloud_upload_mb", 0) > 5_000,
         "MASSIVE_CLOUD_UPLOAD"),

        (record.get("privileged_cmd_count", 0) > 200,
         "EXCESSIVE_PRIVILEGED_COMMANDS"),

        (record.get("sensitive_records_accessed", 0) > 2_000,
         "MASS_SENSITIVE_RECORD_ACCESS"),
    ]
    for condition, name in hard:
        if condition:
            fired.append(f"HARD:{name}")

    # ── Soft rules ────────────────────────────────────────────────
    soft = [
        (record.get("usb_used", 0) and record.get("access_outside_hours", 0),
         "SOFT:USB_OFF_HOURS", 0.20),

        (record.get("email_external_count", 0) > 50
         and record.get("email_attachment_mb", 0) > 100,
         "SOFT:BULK_EXTERNAL_EMAIL", 0.20),

        (record.get("days_before_resignation", 999) <= 14,
         "SOFT:IMMINENT_DEPARTURE", 0.25),

        (record.get("new_software_installed", 0)
         and record.get("remote_desktop_used", 0),
         "SOFT:REMOTE_TOOL_INSTALL", 0.15),

        (record.get("deviation_from_baseline", 0) > 0.80
         and record.get("peer_deviation_score", 0) > 0.80,
         "SOFT:DUAL_DEVIATION_SPIKE", 0.20),

        (record.get("recent_disciplinary", 0)
         and record.get("access_level_changed", 0),
         "SOFT:DISCIPLINARY_PLUS_PRIVILEGE_CHANGE", 0.15),
    ]
    for condition, name, b in soft:
        if condition:
            fired.append(name)
            boost += b

    return RuleResult(
        triggered=any("HARD:" in f for f in fired),
        rules_fired=fired,
        score_boost=min(boost, 0.45),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Insider Risk Detector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InsiderResult:
    score:        float
    label:        int
    risk_level:   str
    action:       str
    archetype:    str          # data_thief | privilege_abuser | leaver | unknown
    reason_codes: List[str]
    rule_result:  RuleResult
    threshold:    float

    def __str__(self) -> str:
        return (
            f"score={self.score:.4f} | risk={self.risk_level} | "
            f"action={self.action} | archetype={self.archetype} | "
            f"reasons={self.reason_codes}"
        )


class InsiderRiskDetector:
    """
    Ensemble: IsolationForest (50%) + RandomForest (50%)
    with rule pre-filter and archetype classification.

    Why IsoForest?
    - Labeled insider threat data is extremely sparse in production.
    - Novel techniques (e.g. steganography, staged exfil) won't match
      historical patterns; unsupervised anomaly detection adapts naturally.
    """

    THRESHOLD    = 0.45
    ISO_WEIGHT   = 0.50
    RF_WEIGHT    = 0.50
    ISO_SHIFT    = 0.05
    ISO_SCALE    = 0.50

    def __init__(self):
        self.fe        = InsiderFeatureEngineer()
        self.scaler    = RobustScaler()    # RobustScaler handles outliers better than Standard
        self.isoforest = IsolationForest(
            n_estimators=400,
            contamination=0.02,
            max_features=0.8,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )
        self.rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "InsiderRiskDetector":
        X = self.fe.fit_transform(df)
        y = X.pop("label") if "label" in X.columns else None

        for f in MODEL_FEATURES:
            if f not in X.columns:
                X[f] = 0.0

        X_feat = X[MODEL_FEATURES]
        X_sc   = self.scaler.fit_transform(X_feat)

        self.isoforest.fit(X_sc)

        if y is not None:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_sc, y, stratify=y, test_size=0.20, random_state=42
            )
            self.rf.fit(X_tr, y_tr)

            if verbose:
                val_proba = self._ensemble_score(X_val)
                val_pred  = (val_proba >= self.THRESHOLD).astype(int)
                print("── Validation (hold-out 20%) ──")
                print(f"  ROC-AUC        : {roc_auc_score(y_val, val_proba):.4f}")
                print(f"  Avg Precision  : {average_precision_score(y_val, val_proba):.4f}")
                print(classification_report(y_val, val_pred, digits=4))
        else:
            # No labels — fit RF on all data with IsoForest pseudo-labels
            pseudo = (
                (-self.isoforest.score_samples(X_sc) > self.ISO_SHIFT + self.ISO_SCALE * 0.5)
                .astype(int)
            )
            self.rf.fit(X_sc, pseudo)

        self._trained = True
        return self

    # ── Scoring helpers ───────────────────────────────────────────────────────

    def _ensemble_score(self, X_sc: np.ndarray) -> np.ndarray:
        iso_raw  = -self.isoforest.score_samples(X_sc)
        iso_norm = np.clip((iso_raw - self.ISO_SHIFT) / self.ISO_SCALE, 0, 1)
        rf_prob  = self.rf.predict_proba(X_sc)[:, 1]
        return self.ISO_WEIGHT * iso_norm + self.RF_WEIGHT * rf_prob

    def _prepare_single(self, record: Dict) -> np.ndarray:
        row = {**RAW_SCHEMA, **record}
        df  = self.fe.transform(pd.DataFrame([row]))
        for f in MODEL_FEATURES:
            if f not in df.columns:
                df[f] = 0.0
        return self.scaler.transform(df[MODEL_FEATURES])

    # ── Archetype classification ──────────────────────────────────────────────

    @staticmethod
    def _classify_archetype(record: Dict, score: float) -> str:
        if score < 0.45:
            return "benign"
        exfil_signals = (
            record.get("usb_used", 0)
            + (record.get("cloud_upload_mb", 0) > 100)
            + (record.get("email_attachment_mb", 0) > 50)
            + (record.get("download_count", 0) > 50)
        )
        priv_signals = (
            (record.get("privileged_cmd_count", 0) > 20)
            + (record.get("unique_systems_accessed", 0) > 10)
            + record.get("access_level_changed", 0)
            + (record.get("failed_access_attempts", 0) > 10)
        )
        leaver_signals = (
            (record.get("days_before_resignation", 999) <= 30)
            + record.get("recent_disciplinary", 0)
            + record.get("terminated_account_active", 0)
        )
        scores = {
            "data_thief":       exfil_signals,
            "privilege_abuser": priv_signals,
            "disgruntled_leaver": leaver_signals,
        }
        top = max(scores, key=scores.get)
        return top if scores[top] > 0 else "unknown"

    # ── Reason codes ──────────────────────────────────────────────────────────

    @staticmethod
    def _reason_codes(record: Dict) -> List[str]:
        codes = []
        if record.get("access_outside_hours"):                      codes.append("OFF_HOURS_ACCESS")
        if record.get("weekend_access"):                            codes.append("WEEKEND_ACCESS")
        if record.get("usb_used"):                                  codes.append("USB_DEVICE_USED")
        if record.get("data_volume_mb", 0) > 500:                  codes.append("HIGH_DATA_VOLUME")
        if record.get("download_count", 0) > 50:                   codes.append("EXCESSIVE_DOWNLOADS")
        if record.get("cloud_upload_mb", 0) > 100:                 codes.append("LARGE_CLOUD_UPLOAD")
        if record.get("sensitive_records_accessed", 0) > 100:      codes.append("MASS_SENSITIVE_ACCESS")
        if record.get("privileged_cmd_count", 0) > 20:             codes.append("EXCESSIVE_PRIV_COMMANDS")
        if record.get("unique_systems_accessed", 0) > 10:          codes.append("BROAD_SYSTEM_ACCESS")
        if record.get("email_external_count", 0) > 20:             codes.append("HIGH_EXTERNAL_EMAIL")
        if record.get("email_attachment_mb", 0) > 50:              codes.append("LARGE_EMAIL_ATTACHMENTS")
        if record.get("deviation_from_baseline", 0) > 0.60:        codes.append("BEHAVIOURAL_DEVIATION")
        if record.get("peer_deviation_score", 0) > 0.60:           codes.append("PEER_GROUP_DEVIATION")
        if record.get("days_before_resignation", 999) <= 30:       codes.append("RESIGNATION_WINDOW")
        if record.get("terminated_account_active"):                 codes.append("TERMINATED_ACCOUNT")
        if record.get("recent_disciplinary"):                       codes.append("RECENT_DISCIPLINARY")
        if record.get("access_level_changed"):                      codes.append("PRIVILEGE_CHANGED")
        if record.get("remote_desktop_used")                        \
                and record.get("new_software_installed"):           codes.append("REMOTE_TOOL_INSTALL")
        return codes

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self, record: Dict) -> InsiderResult:
        rule_result = apply_hard_rules(record)

        X_sc     = self._prepare_single(record)
        ml_score = float(self._ensemble_score(X_sc)[0])

        final_score = min(ml_score + rule_result.score_boost, 1.0)
        if rule_result.triggered:
            final_score = 1.0

        label     = int(final_score >= self.THRESHOLD)
        risk      = self._risk_level(final_score)
        action    = {"HIGH": "BLOCK", "MEDIUM": "REVIEW", "LOW": "ALLOW"}[risk]
        archetype = self._classify_archetype(record, final_score)
        reasons   = self._reason_codes(record) + rule_result.rules_fired

        return InsiderResult(
            score=round(final_score, 4),
            label=label,
            risk_level=risk,
            action=action,
            archetype=archetype,
            reason_codes=reasons,
            rule_result=rule_result,
            threshold=self.THRESHOLD,
        )

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.75:   return "HIGH"
        elif score >= 0.45: return "MEDIUM"
        else:               return "LOW"

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
        return (
            pd.DataFrame({
                "feature":    MODEL_FEATURES,
                "importance": self.rf.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def tune_threshold(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def cross_validate(self, df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
        """Stratified k-fold CV — more reliable than a single train/test split."""
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

            scaler = RobustScaler()
            X_tr_sc = scaler.fit_transform(X_tr[MODEL_FEATURES])
            X_val_sc = scaler.transform(X_val[MODEL_FEATURES])

            iso = IsolationForest(n_estimators=200, contamination=0.02,
                                  random_state=42, n_jobs=-1)
            iso.fit(X_tr_sc)

            rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                        class_weight="balanced",
                                        random_state=42, n_jobs=-1)
            rf.fit(X_tr_sc, y_tr)

            iso_raw  = -iso.score_samples(X_val_sc)
            iso_norm = np.clip((iso_raw - 0.05) / 0.5, 0, 1)
            rf_prob  = rf.predict_proba(X_val_sc)[:, 1]
            proba    = 0.5 * iso_norm + 0.5 * rf_prob
            y_pred   = (proba >= self.THRESHOLD).astype(int)

            records.append({
                "fold":          fold,
                "roc_auc":       roc_auc_score(y_val, proba),
                "avg_precision": average_precision_score(y_val, proba),
                "precision":     (y_pred[y_val == 1] == 1).mean() if y_pred.sum() else 0,
                "recall":        (y_pred[y_val == 1] == 1).mean(),
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
    _section("Generating Synthetic Insider Threat Data (3 archetypes)")
    df = InsiderDataGenerator.generate(n_samples=15_000, fraud_rate=0.02)
    n_fraud = df["label"].sum()
    print(f"  Total    : {len(df):,} activity records")
    print(f"  Fraud    : {n_fraud:,} ({n_fraud/len(df):.1%})")
    print(f"  Features : {len(RAW_SCHEMA)} raw → {len(MODEL_FEATURES)} model features")

    _section("Training InsiderRiskDetector")
    detector = InsiderRiskDetector()
    detector.fit(df)

    _section("Full Dataset Evaluation")
    metrics = detector.evaluate(df)
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision  : {metrics['avg_precision']:.4f}")
    print(f"  Confusion      : {metrics['confusion']}")
    print(metrics["report"])

    _section("Feature Importance (top 12)")
    print(detector.feature_importance().head(12).to_string(index=False))

    _section("Threshold Tuning")
    tdf = detector.tune_threshold(df)
    best = tdf[tdf["f1"] > 0.50].sort_values("f1", ascending=False).head(5)
    print(best.to_string(index=False))

    _section("Cross-Validation (5-fold stratified)")
    cv = detector.cross_validate(df, n_splits=5)
    print(cv.to_string())

    _section("Live Inference Demo")

    cases = [
        # ── Data thief ────────────────────────────────────────────
        ({
            "access_outside_hours": 1,  "weekend_access": 1,
            "data_volume_mb": 25_000,   "download_count": 350,
            "upload_count": 80,         "usb_used": 1,
            "cloud_upload_mb": 8_000,   "email_external_count": 120,
            "email_attachment_mb": 500, "sensitive_records_accessed": 800,
            "unique_systems_accessed": 8, "deviation_from_baseline": 0.90,
            "peer_deviation_score": 0.85, "days_before_resignation": 999,
            "recent_disciplinary": 0,   "access_level_changed": 0,
            "terminated_account_active": 0,
        }, "Data thief — USB + cloud + bulk download"),

        # ── Privilege abuser ──────────────────────────────────────
        ({
            "access_outside_hours": 1,  "weekend_access": 1,
            "data_volume_mb": 1_200,    "download_count": 20,
            "privileged_cmd_count": 350, "unique_systems_accessed": 22,
            "unique_tables_accessed": 60, "failed_access_attempts": 30,
            "remote_desktop_used": 1,   "new_software_installed": 1,
            "deviation_from_baseline": 0.75, "peer_deviation_score": 0.80,
            "access_level_changed": 1,  "recent_disciplinary": 1,
            "terminated_account_active": 0, "days_before_resignation": 999,
        }, "Privilege abuser — admin commands + broad lateral movement"),

        # ── Disgruntled leaver ────────────────────────────────────
        ({
            "access_outside_hours": 1,  "weekend_access": 1,
            "data_volume_mb": 9_000,    "download_count": 180,
            "usb_used": 1,              "email_external_count": 70,
            "email_attachment_mb": 300, "cloud_upload_mb": 2_000,
            "deviation_from_baseline": 0.80, "peer_deviation_score": 0.70,
            "days_before_resignation": 5,    "recent_disciplinary": 1,
            "terminated_account_active": 0,  "sensitive_records_accessed": 200,
        }, "Disgruntled leaver — 5 days from exit, draining data"),

        # ── Terminated account still active ───────────────────────
        ({
            "access_outside_hours": 1,  "data_volume_mb": 500,
            "download_count": 40,       "usb_used": 0,
            "deviation_from_baseline": 0.60, "terminated_account_active": 1,
            "days_before_resignation": 999,
        }, "Ghost account — terminated employee still accessing systems"),

        # ── Legitimate employee ───────────────────────────────────
        ({
            "access_outside_hours": 0,  "weekend_access": 0,
            "data_volume_mb": 45,       "download_count": 8,
            "usb_used": 0,              "cloud_upload_mb": 1.5,
            "email_external_count": 3,  "email_attachment_mb": 2,
            "sensitive_records_accessed": 5, "unique_systems_accessed": 2,
            "privileged_cmd_count": 0,  "deviation_from_baseline": 0.08,
            "peer_deviation_score": 0.05, "days_before_resignation": 999,
            "recent_disciplinary": 0,   "access_level_changed": 0,
            "terminated_account_active": 0,
        }, "Legitimate employee — normal daily activity"),
    ]

    for record, description in cases:
        result = detector.predict(record)
        print(f"\n  Case      : {description}")
        print(f"  Result    : {result}")


if __name__ == "__main__":
    main()