"""
train.py — Unified Fraud Detection Training Entry Point
========================================================
Supports all 10 fraud detectors with synthetic or real data.

Usage
-----
  # Train a single detector
  python train.py --dataset card --synthetic
  python train.py --dataset card --data-path data/raw/creditcard.csv

  # Train multiple detectors
  python train.py --dataset card login aml --synthetic

  # Train all 10 at once
  python train.py --all --synthetic

  # Save models to models/
  python train.py --all --synthetic --save

  # List available detectors
  python train.py --list
"""

import argparse
import sys
import os
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, precision_recall_curve,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
# Registry of all 10 detectors
# ─────────────────────────────────────────────────────────────────────────────

DETECTOR_REGISTRY = {
    # key : (description, fraud_rate, has_real_loader)
    "card":       ("Credit Card Fraud",          0.020, True),
    "login":      ("Account Takeover",            0.030, False),
    "loan":       ("Loan Fraud",                  0.050, False),
    "insider":    ("Insider Threat",              0.020, False),
    "aml":        ("Money Laundering (AML)",      0.030, False),
    "wire":       ("Wire Transfer / BEC Fraud",   0.040, False),
    "synthetic_id":("Synthetic Identity Fraud",   0.060, False),
    "atm":        ("ATM & Card Skimming",         0.025, False),
    "mobile":     ("Mobile Banking Fraud",        0.030, False),
    "merchant":   ("Merchant Fraud",              0.050, False),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    bar = "═" * 62
    print(f"\n{bar}\n  {title}\n{bar}")

def save_model(artifact: dict, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved  → {path}  ({size_kb:.0f} KB)")

def print_metrics(name, roc, ap, threshold, cm=None):
    print(f"\n  ── {name.upper()} ──")
    print(f"  ROC-AUC        : {roc:.4f}")
    print(f"  Avg Precision  : {ap:.4f}")
    print(f"  Threshold      : {threshold:.4f}")
    if cm is not None:
        print(f"  Confusion      :")
        print(f"    Legit: {cm[0,0]:>7,} correct | {cm[0,1]:>5,} false alarms")
        print(f"    Fraud: {cm[1,1]:>7,} caught  | {cm[1,0]:>5,} missed")

def best_threshold(y_true, proba):
    """Pick threshold maximizing F1 on the positive class."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    idx = f1.argmax()
    return float(thresholds[idx]) if idx < len(thresholds) else 0.5

def scale_pos_weight(fraud_rate: float) -> int:
    return max(1, int((1 - fraud_rate) / fraud_rate))


# ─────────────────────────────────────────────────────────────────────────────
# Real dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_card_real(path: str):
    """ULB/Kaggle creditcard.csv  (V1-V28 PCA + Amount + Class)"""
    df = pd.read_csv(path)
    df["log_amount"]  = np.log1p(df["Amount"])
    df["hour"]        = (df["Time"] / 3600).astype(int) % 24
    df["is_night"]    = df["hour"].between(0, 5).astype(int)
    df["amount_norm"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1)
    v_cols    = [f"V{i}" for i in range(1, 29)]
    feat_cols = v_cols + ["log_amount", "is_night", "amount_norm"]
    df["label"] = df["Class"]
    return df, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic generators (one per detector)
# ─────────────────────────────────────────────────────────────────────────────

def _gen(seed): return np.random.default_rng(seed)

def gen_card(n=20_000, fraud_rate=0.02):
    rng = _gen(42); nf = int(n*fraud_rate); nl = n-nf
    avg = rng.lognormal(3.8, 0.6, nl)
    legit = pd.DataFrame({
        "amount": np.clip(rng.normal(avg, avg*0.3), 1, None),
        "hour": rng.integers(7, 22, nl), "merchant_risk": rng.uniform(0, 0.15, nl),
        "tx_count_1h": rng.integers(0, 3, nl), "tx_count_24h": rng.integers(1, 8, nl),
        "country_mismatch": rng.binomial(1, 0.01, nl), "distance_km": rng.exponential(8, nl),
        "card_present": rng.binomial(1, 0.85, nl), "is_new_merchant": rng.binomial(1, 0.08, nl),
        "declined_24h": rng.integers(0, 1, nl), "avg_amount_30d": avg, "label": np.zeros(nl, int)})
    avg_f = rng.lognormal(3.8, 0.6, nf)
    fraud = pd.DataFrame({
        "amount": rng.lognormal(5.5, 1.2, nf), "hour": rng.choice([0,1,2,3,4,22,23], nf),
        "merchant_risk": rng.uniform(0.5, 1, nf), "tx_count_1h": rng.integers(4, 15, nf),
        "tx_count_24h": rng.integers(8, 30, nf), "country_mismatch": rng.binomial(1, 0.7, nf),
        "distance_km": rng.exponential(800, nf), "card_present": rng.binomial(1, 0.2, nf),
        "is_new_merchant": rng.binomial(1, 0.8, nf), "declined_24h": rng.integers(1, 6, nf),
        "avg_amount_30d": avg_f, "label": np.ones(nf, int)})
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    df["log_amount"] = np.log1p(df["amount"]); df["is_night"] = df["hour"].between(0,5).astype(int)
    df["amount_vs_avg"] = df["amount"] / (df["avg_amount_30d"] + 1)
    feats = ["merchant_risk","tx_count_1h","tx_count_24h","country_mismatch","card_present",
             "is_new_merchant","declined_24h","log_amount","is_night","amount_vs_avg"]
    return df, feats

def gen_login(n=15_000, fraud_rate=0.03):
    rng = _gen(43); nf = int(n*fraud_rate); nl = n-nf
    legit = pd.DataFrame({
        "failed_attempts": rng.integers(0,2,nl), "hour": rng.integers(7,22,nl),
        "is_new_device": rng.binomial(1,0.05,nl), "ip_risk_score": rng.uniform(0,0.2,nl),
        "location_km": rng.exponential(5,nl), "time_since_h": rng.exponential(24,nl),
        "vpn_used": rng.binomial(1,0.05,nl), "sensitive_action": rng.binomial(1,0.1,nl),
        "mfa_bypassed": rng.binomial(1,0.02,nl), "label": np.zeros(nl,int)})
    fraud = pd.DataFrame({
        "failed_attempts": rng.integers(3,20,nf), "hour": rng.choice([0,1,2,3,4,5],nf),
        "is_new_device": rng.binomial(1,0.9,nf), "ip_risk_score": rng.uniform(0.6,1,nf),
        "location_km": rng.exponential(2000,nf), "time_since_h": rng.exponential(0.5,nf),
        "vpn_used": rng.binomial(1,0.7,nf), "sensitive_action": rng.binomial(1,0.8,nf),
        "mfa_bypassed": rng.binomial(1,0.55,nf), "label": np.ones(nf,int)})
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=43).reset_index(drop=True)
    df["log_dist"] = np.log1p(df["location_km"]); df["is_night"] = df["hour"].between(0,5).astype(int)
    df["impossible_travel"] = (df["location_km"] / (df["time_since_h"]+0.01) > 900).astype(int)
    feats = ["failed_attempts","is_new_device","ip_risk_score","vpn_used",
             "sensitive_action","mfa_bypassed","log_dist","is_night","impossible_travel"]
    return df, feats

def gen_loan(n=8_000, fraud_rate=0.05):
    rng = _gen(44); nf = int(n*fraud_rate); nl = n-nf
    legit = pd.DataFrame({
        "lti_ratio": rng.uniform(0.1,0.5,nl), "credit_score": rng.integers(580,850,nl),
        "employ_months": rng.integers(12,240,nl), "address_months": rng.integers(12,120,nl),
        "apps_30d": rng.integers(0,2,nl), "doc_completeness": rng.uniform(0.85,1,nl),
        "inquiries_6m": rng.integers(0,4,nl), "income_verified": rng.binomial(1,0.9,nl),
        "label": np.zeros(nl,int)})
    fraud = pd.DataFrame({
        "lti_ratio": rng.uniform(0.8,5,nf), "credit_score": rng.integers(400,620,nf),
        "employ_months": rng.integers(0,6,nf), "address_months": rng.integers(0,3,nf),
        "apps_30d": rng.integers(3,15,nf), "doc_completeness": rng.uniform(0.3,0.75,nf),
        "inquiries_6m": rng.integers(5,20,nf), "income_verified": rng.binomial(1,0.15,nf),
        "label": np.ones(nf,int)})
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=44).reset_index(drop=True)
    feats = ["lti_ratio","credit_score","employ_months","address_months",
             "apps_30d","doc_completeness","inquiries_6m","income_verified"]
    return df, feats

def gen_insider(n=10_000, fraud_rate=0.02):
    rng = _gen(45); nf = int(n*fraud_rate); nl = n-nf
    legit = pd.DataFrame({
        "off_hours": rng.binomial(1,0.05,nl), "data_mb": rng.lognormal(2,1,nl),
        "sensitive_records": rng.integers(0,20,nl), "downloads": rng.integers(0,10,nl),
        "systems": rng.integers(1,5,nl), "usb": rng.binomial(1,0.02,nl),
        "email_ext": rng.integers(0,5,nl), "deviation": rng.uniform(0,0.2,nl),
        "label": np.zeros(nl,int)})
    fraud = pd.DataFrame({
        "off_hours": rng.binomial(1,0.8,nf), "data_mb": rng.lognormal(5,1.5,nf),
        "sensitive_records": rng.integers(50,500,nf), "downloads": rng.integers(20,200,nf),
        "systems": rng.integers(8,20,nf), "usb": rng.binomial(1,0.6,nf),
        "email_ext": rng.integers(10,100,nf), "deviation": rng.uniform(0.6,1,nf),
        "label": np.ones(nf,int)})
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=45).reset_index(drop=True)
    df["log_mb"] = np.log1p(df["data_mb"]); df["log_rec"] = np.log1p(df["sensitive_records"])
    feats = ["off_hours","log_mb","log_rec","downloads","systems","usb","email_ext","deviation"]
    return df, feats

def gen_aml(n=12_000, fraud_rate=0.03):
    rng = _gen(46); nf = int(n*fraud_rate); nl = n-nf
    legit = pd.DataFrame({
        "amount": rng.lognormal(6.5,1.2,nl), "is_cash": rng.binomial(1,0.15,nl),
        "tx_count_48h": rng.integers(1,5,nl), "structured_count_30d": rng.integers(0,1,nl),
        "round_amount": rng.binomial(1,0.15,nl), "account_hops": rng.integers(1,3,nl),
        "beneficiaries_7d": rng.integers(1,5,nl), "sender_risk": rng.uniform(0,0.2,nl),
        "receiver_risk": rng.uniform(0,0.2,nl), "cross_border": rng.binomial(1,0.1,nl),
        "high_risk_country": rng.binomial(1,0.02,nl), "is_shell": rng.binomial(1,0.02,nl),
        "pep": rng.binomial(1,0.01,nl), "kyc_risk": rng.uniform(0,0.2,nl),
        "deviation": rng.uniform(0,0.15,nl), "unusual_biz": rng.binomial(1,0.03,nl),
        "label": np.zeros(nl,int)})
    fraud = pd.DataFrame({
        "amount": rng.uniform(8000,9999,nf), "is_cash": rng.binomial(1,0.7,nf),
        "tx_count_48h": rng.integers(5,20,nf), "structured_count_30d": rng.integers(2,15,nf),
        "round_amount": rng.binomial(1,0.6,nf), "account_hops": rng.integers(3,12,nf),
        "beneficiaries_7d": rng.integers(5,30,nf), "sender_risk": rng.uniform(0.5,1,nf),
        "receiver_risk": rng.uniform(0.4,1,nf), "cross_border": rng.binomial(1,0.8,nf),
        "high_risk_country": rng.binomial(1,0.6,nf), "is_shell": rng.binomial(1,0.7,nf),
        "pep": rng.binomial(1,0.3,nf), "kyc_risk": rng.uniform(0.5,1,nf),
        "deviation": rng.uniform(0.6,1,nf), "unusual_biz": rng.binomial(1,0.8,nf),
        "label": np.ones(nf,int)})
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=46).reset_index(drop=True)
    df["log_amount"] = np.log1p(df["amount"])
    df["near_ctr"]   = df["amount"].between(8000,9999).astype(int)
    df["geo_risk"]   = ((df["sender_risk"]+df["receiver_risk"])/2*(1+df["cross_border"]*0.5)).clip(0,1)
    df["entity_risk"]= (df["pep"]*0.35 + df["is_shell"]*0.35 + df["kyc_risk"]*0.30)
    feats = ["is_cash","tx_count_48h","structured_count_30d","round_amount",
             "account_hops","beneficiaries_7d","cross_border","high_risk_country",
             "is_shell","pep","deviation","unusual_biz","log_amount","near_ctr","geo_risk","entity_risk"]
    return df, feats

def gen_wire(n=10_000, fraud_rate=0.04):
    rng = _gen(47); nf = int(n*fraud_rate); nl = n-nf
    legit = pd.DataFrame({
        "amount": rng.lognormal(10.5,1.2,nl), "is_international": rng.binomial(1,0.25,nl),
        "is_new_beneficiary": rng.binomial(1,0.08,nl), "beneficiary_age_days": rng.integers(180,3650,nl),
        "via_email": rng.binomial(1,0.3,nl), "urgency": rng.binomial(1,0.05,nl),
        "out_of_policy": rng.binomial(1,0.02,nl), "after_hours": rng.binomial(1,0.05,nl),
        "domain_age_days": rng.integers(180,3650,nl), "amount_vs_avg": rng.uniform(0.5,2,nl),
        "bene_change_7d": rng.binomial(1,0.02,nl), "dual_approval_bypassed": rng.binomial(1,0.02,nl),
        "email_anomaly": rng.binomial(1,0.02,nl), "domain_lookalike": rng.binomial(1,0.01,nl),
        "phone_verified": rng.binomial(1,0.85,nl), "prior_bec": rng.integers(0,1,nl),
        "label": np.zeros(nl,int)})
    fraud = pd.DataFrame({
        "amount": rng.lognormal(12,1,nf), "is_international": rng.binomial(1,0.8,nf),
        "is_new_beneficiary": rng.binomial(1,0.9,nf), "beneficiary_age_days": rng.integers(1,30,nf),
        "via_email": rng.binomial(1,0.95,nf), "urgency": rng.binomial(1,0.9,nf),
        "out_of_policy": rng.binomial(1,0.8,nf), "after_hours": rng.binomial(1,0.6,nf),
        "domain_age_days": rng.integers(1,14,nf), "amount_vs_avg": rng.uniform(3,20,nf),
        "bene_change_7d": rng.binomial(1,0.8,nf), "dual_approval_bypassed": rng.binomial(1,0.7,nf),
        "email_anomaly": rng.binomial(1,0.85,nf), "domain_lookalike": rng.binomial(1,0.75,nf),
        "phone_verified": rng.binomial(1,0.05,nf), "prior_bec": rng.integers(1,5,nf),
        "label": np.ones(nf,int)})
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=47).reset_index(drop=True)
    df["log_amount"]  = np.log1p(df["amount"])
    df["domain_risk"] = (df["domain_age_days"] < 30).astype(int)
    df["comm_risk"]   = (df["email_anomaly"]*0.4 + df["domain_lookalike"]*0.4 + (1-df["phone_verified"])*0.2)
    df["proc_risk"]   = (df["urgency"]*0.3 + df["out_of_policy"]*0.3 + df["dual_approval_bypassed"]*0.4)
    feats = ["is_international","is_new_beneficiary","via_email","urgency","out_of_policy",
             "after_hours","bene_change_7d","dual_approval_bypassed","email_anomaly",
             "domain_lookalike","prior_bec","log_amount","domain_risk","comm_risk","proc_risk"]
    return df, feats

def gen_synthetic_id(n=8_000, fraud_rate=0.06):
    rng = _gen(48); nf = int(n*fraud_rate); nl = n-nf
    legit = pd.DataFrame({
        "age_credit_gap": rng.integers(0,5,nl), "ssn_anomaly": rng.binomial(1,0.02,nl),
        "thin_file": rng.binomial(1,0.1,nl), "au_count": rng.integers(0,2,nl),
        "inquiry_spike": rng.binomial(1,0.05,nl), "inquiries_90d": rng.integers(0,3,nl),
        "new_accounts_6m": rng.integers(0,2,nl), "po_box": rng.binomial(1,0.03,nl),
        "apps_30d": rng.integers(0,2,nl), "employer_unverifiable": rng.binomial(1,0.05,nl),
        "bureau_discrepancies": rng.integers(0,1,nl), "credit_build_pattern": rng.binomial(1,0.05,nl),
        "credit_file_age_m": rng.integers(24,360,nl), "label": np.zeros(nl,int)})
    fraud = pd.DataFrame({
        "age_credit_gap": rng.integers(15,50,nf), "ssn_anomaly": rng.binomial(1,0.8,nf),
        "thin_file": rng.binomial(1,0.8,nf), "au_count": rng.integers(3,10,nf),
        "inquiry_spike": rng.binomial(1,0.8,nf), "inquiries_90d": rng.integers(8,25,nf),
        "new_accounts_6m": rng.integers(4,12,nf), "po_box": rng.binomial(1,0.5,nf),
        "apps_30d": rng.integers(5,20,nf), "employer_unverifiable": rng.binomial(1,0.85,nf),
        "bureau_discrepancies": rng.integers(3,8,nf), "credit_build_pattern": rng.binomial(1,0.9,nf),
        "credit_file_age_m": rng.integers(6,36,nf), "label": np.ones(nf,int)})
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=48).reset_index(drop=True)
    df["log_file_age"] = np.log1p(df["credit_file_age_m"])
    df["id_inconsistency"] = (df["ssn_anomaly"]*0.4 + df["bureau_discrepancies"].clip(upper=5)/5*0.3 + df["age_credit_gap"].clip(upper=40)/40*0.3)
    df["app_velocity"] = df["apps_30d"].clip(upper=20)/20
    feats = ["age_credit_gap","ssn_anomaly","thin_file","au_count","inquiry_spike",
             "inquiries_90d","new_accounts_6m","po_box","employer_unverifiable",
             "bureau_discrepancies","credit_build_pattern","log_file_age","id_inconsistency","app_velocity"]
    return df, feats

def gen_atm(n=15_000, fraud_rate=0.025):
    rng = _gen(49); nf = int(n*fraud_rate); nl = n-nf
    legit = pd.DataFrame({
        "amount": rng.choice([20,40,60,100,200,300,500],nl), "is_max_withdrawal": rng.binomial(1,0.05,nl),
        "pin_speed_ms": rng.integers(1500,8000,nl), "chip_used": rng.binomial(1,0.9,nl),
        "atm_tamper": rng.binomial(1,0.01,nl), "fraud_reports_24h": rng.integers(0,1,nl),
        "cards_same_atm_1h": rng.integers(1,5,nl), "atm_hops_1h": rng.integers(1,2,nl),
        "failed_pin": rng.integers(0,1,nl), "country_mismatch": rng.binomial(1,0.02,nl),
        "magstripe_fallback": rng.binomial(1,0.02,nl), "card_compromised": np.zeros(nl,int),
        "deviation": rng.uniform(0,0.15,nl), "label": np.zeros(nl,int)})
    fraud = pd.DataFrame({
        "amount": rng.choice([200,300,400,500,1000],nf), "is_max_withdrawal": rng.binomial(1,0.85,nf),
        "pin_speed_ms": rng.integers(200,700,nf), "chip_used": rng.binomial(1,0.1,nf),
        "atm_tamper": rng.binomial(1,0.6,nf), "fraud_reports_24h": rng.integers(3,20,nf),
        "cards_same_atm_1h": rng.integers(10,50,nf), "atm_hops_1h": rng.integers(3,10,nf),
        "failed_pin": rng.integers(0,2,nf), "country_mismatch": rng.binomial(1,0.7,nf),
        "magstripe_fallback": rng.binomial(1,0.8,nf), "card_compromised": rng.binomial(1,0.4,nf),
        "deviation": rng.uniform(0.7,1,nf), "label": np.ones(nf,int)})
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=49).reset_index(drop=True)
    df["log_amount"]   = np.log1p(df["amount"])
    df["fast_pin"]     = (df["pin_speed_ms"] < 800).astype(int)
    df["cluster_risk"] = (np.log1p(df["cards_same_atm_1h"])*0.5 + df["atm_tamper"]*0.3 + df["fraud_reports_24h"].clip(upper=10)/10*0.2)
    feats = ["is_max_withdrawal","chip_used","atm_tamper","fraud_reports_24h","cards_same_atm_1h",
             "atm_hops_1h","failed_pin","country_mismatch","magstripe_fallback","card_compromised",
             "deviation","log_amount","fast_pin","cluster_risk"]
    return df, feats

def gen_mobile(n=14_000, fraud_rate=0.03):
    rng = _gen(50); nf = int(n*fraud_rate); nl = n-nf
    legit = pd.DataFrame({
        "rooted": rng.binomial(1,0.05,nl), "emulator": rng.binomial(1,0.01,nl),
        "overlay": rng.binomial(1,0.02,nl), "accessibility_abuse": rng.binomial(1,0.01,nl),
        "tap_anomaly": rng.uniform(0,0.15,nl), "concurrent_sessions": rng.binomial(1,0.01,nl),
        "sim_swap": rng.binomial(1,0.02,nl), "vpn": rng.binomial(1,0.08,nl),
        "ip_gps_mismatch": rng.binomial(1,0.03,nl), "amount": rng.lognormal(5,1.2,nl),
        "new_payee": rng.binomial(1,0.1,nl), "payee_added_this_session": rng.binomial(1,0.03,nl),
        "tx_60s_login": rng.binomial(1,0.05,nl), "max_limit_hit": rng.binomial(1,0.03,nl),
        "otp_auto_filled": rng.binomial(1,0.05,nl), "device_score": rng.uniform(0.85,1,nl),
        "label": np.zeros(nl,int)})
    fraud = pd.DataFrame({
        "rooted": rng.binomial(1,0.8,nf), "emulator": rng.binomial(1,0.6,nf),
        "overlay": rng.binomial(1,0.75,nf), "accessibility_abuse": rng.binomial(1,0.65,nf),
        "tap_anomaly": rng.uniform(0.6,1,nf), "concurrent_sessions": rng.binomial(1,0.7,nf),
        "sim_swap": rng.binomial(1,0.6,nf), "vpn": rng.binomial(1,0.75,nf),
        "ip_gps_mismatch": rng.binomial(1,0.8,nf), "amount": rng.lognormal(8.5,1,nf),
        "new_payee": rng.binomial(1,0.9,nf), "payee_added_this_session": rng.binomial(1,0.85,nf),
        "tx_60s_login": rng.binomial(1,0.9,nf), "max_limit_hit": rng.binomial(1,0.8,nf),
        "otp_auto_filled": rng.binomial(1,0.75,nf), "device_score": rng.uniform(0,0.4,nf),
        "label": np.ones(nf,int)})
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=50).reset_index(drop=True)
    df["log_amount"]  = np.log1p(df["amount"])
    df["device_risk"] = (df["rooted"]*0.3+df["emulator"]*0.25+df["overlay"]*0.2+df["accessibility_abuse"]*0.15+(1-df["device_score"])*0.1)
    df["session_risk"]= (df["concurrent_sessions"]*0.3+df["payee_added_this_session"]*0.25+df["tx_60s_login"]*0.2+df["otp_auto_filled"]*0.15+df["max_limit_hit"]*0.1)
    feats = ["rooted","emulator","overlay","accessibility_abuse","tap_anomaly","concurrent_sessions",
             "sim_swap","vpn","ip_gps_mismatch","new_payee","payee_added_this_session",
             "tx_60s_login","otp_auto_filled","log_amount","device_risk","session_risk"]
    return df, feats

def gen_merchant(n=10_000, fraud_rate=0.05):
    rng = _gen(51); nf = int(n*fraud_rate); nl = n-nf
    legit = pd.DataFrame({
        "merchant_age_days": rng.integers(90,3650,nl), "mcc_risk": rng.uniform(0,0.15,nl),
        "verified": rng.binomial(1,0.95,nl), "kyb_score": rng.uniform(0.7,1,nl),
        "chargeback_rate": rng.uniform(0,0.01,nl), "chargeback_count_7d": rng.integers(0,2,nl),
        "decline_rate": rng.uniform(0,0.05,nl), "refund_rate": rng.uniform(0,0.03,nl),
        "micro_tx_1h": rng.integers(0,1,nl), "seq_card_attempt": rng.binomial(1,0.01,nl),
        "velocity_spike": rng.binomial(1,0.05,nl), "unique_cards_1h": rng.integers(1,20,nl),
        "country_count_24h": rng.integers(1,3,nl), "ship_bill_mismatch": rng.uniform(0,0.05,nl),
        "refund_sales_ratio": rng.uniform(0,0.03,nl), "label": np.zeros(nl,int)})
    fraud = pd.DataFrame({
        "merchant_age_days": rng.integers(1,30,nf), "mcc_risk": rng.uniform(0.5,1,nf),
        "verified": rng.binomial(1,0.1,nf), "kyb_score": rng.uniform(0,0.3,nf),
        "chargeback_rate": rng.uniform(0.05,0.5,nf), "chargeback_count_7d": rng.integers(5,50,nf),
        "decline_rate": rng.uniform(0.2,0.8,nf), "refund_rate": rng.uniform(0.15,0.8,nf),
        "micro_tx_1h": rng.integers(50,500,nf), "seq_card_attempt": rng.binomial(1,0.8,nf),
        "velocity_spike": rng.binomial(1,0.9,nf), "unique_cards_1h": rng.integers(50,500,nf),
        "country_count_24h": rng.integers(5,30,nf), "ship_bill_mismatch": rng.uniform(0.3,0.9,nf),
        "refund_sales_ratio": rng.uniform(0.3,1,nf), "label": np.ones(nf,int)})
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=51).reset_index(drop=True)
    df["log_age"] = np.log1p(df["merchant_age_days"])
    df["card_test_score"] = (np.log1p(df["micro_tx_1h"])*0.4+df["seq_card_attempt"]*0.3+df["decline_rate"].clip(upper=1)*0.3)
    df["chargeback_risk"]  = df["chargeback_rate"].clip(upper=0.3)/0.3
    feats = ["mcc_risk","verified","kyb_score","chargeback_rate","chargeback_count_7d",
             "decline_rate","refund_rate","micro_tx_1h","seq_card_attempt","velocity_spike",
             "unique_cards_1h","country_count_24h","ship_bill_mismatch","refund_sales_ratio",
             "log_age","card_test_score","chargeback_risk"]
    return df, feats


SYNTHETIC_GENERATORS = {
    "card":         gen_card,
    "login":        gen_login,
    "loan":         gen_loan,
    "insider":      gen_insider,
    "aml":          gen_aml,
    "wire":         gen_wire,
    "synthetic_id": gen_synthetic_id,
    "atm":          gen_atm,
    "mobile":       gen_mobile,
    "merchant":     gen_merchant,
}


# ─────────────────────────────────────────────────────────────────────────────
# Model factory — pick algorithm per detector
# ─────────────────────────────────────────────────────────────────────────────

def build_model(ds_key: str, fraud_rate: float):
    """
    Algorithm selection per detector type:
      XGBoost        → card, loan, aml, wire, synthetic_id, merchant  (rich labeled data)
      IsoForest + RF → login, insider, atm, mobile  (sparse labels / novel patterns)
    """
    spw = scale_pos_weight(fraud_rate)

    if ds_key in {"card", "loan", "aml", "wire", "synthetic_id", "merchant"}:
        return ("xgb", xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=0.1,
            eval_metric="aucpr", random_state=42, n_jobs=-1,
        ))
    else:  # login, insider, atm, mobile
        iso = IsolationForest(n_estimators=200, contamination=fraud_rate,
                              random_state=42, n_jobs=-1)
        rf  = RandomForestClassifier(n_estimators=250, max_depth=10,
                                     class_weight="balanced",
                                     random_state=42, n_jobs=-1)
        return ("ensemble", (iso, rf))


# ─────────────────────────────────────────────────────────────────────────────
# Core training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(ds_key, df, feat_cols, args):
    desc, fraud_rate, _ = DETECTOR_REGISTRY[ds_key]
    section(f"Training — {desc}")

    X = df[feat_cols]
    y = df["label"]
    print(f"  Samples    : {len(df):,}")
    print(f"  Fraud      : {y.sum():,} ({y.mean():.2%})")
    print(f"  Features   : {len(feat_cols)}")

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, stratify=y, test_size=0.2, random_state=42)

    model_type, model_obj = build_model(ds_key, fraud_rate)
    t0 = time.time()

    if model_type == "xgb":
        model_obj.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        proba = model_obj.predict_proba(Xte)[:, 1]
    else:
        iso, rf = model_obj
        iso.fit(Xtr)
        rf.fit(Xtr, ytr)
        iso_norm = np.clip((-iso.score_samples(Xte) - 0.05) / 0.5, 0, 1)
        rf_prob  = rf.predict_proba(Xte)[:, 1]
        proba    = 0.40 * iso_norm + 0.60 * rf_prob

    elapsed = time.time() - t0

    thr   = best_threshold(yte, proba)
    pred  = (proba >= thr).astype(int)
    roc   = roc_auc_score(yte, proba)
    ap    = average_precision_score(yte, proba)
    cm    = confusion_matrix(yte, pred)

    print_metrics(desc, roc, ap, thr, cm)
    print(f"\n  Trained in : {elapsed:.1f}s")
    print(f"\n{classification_report(yte, pred, digits=4)}")

    # Feature importance (XGBoost only)
    if model_type == "xgb":
        fi = sorted(zip(feat_cols, model_obj.feature_importances_),
                    key=lambda x: x[1], reverse=True)[:8]
        print("  Feature importance (top 8):")
        for fname, imp in fi:
            bar = "█" * int(imp * 200)
            print(f"    {fname:30s} {imp:.4f}  {bar}")

    artifact = {
        "model_type": model_type, "model": model_obj,
        "scaler": scaler, "feat_cols": feat_cols,
        "threshold": thr, "roc_auc": roc, "avg_precision": ap,
        "detector": ds_key,
    }

    if args.save:
        os.makedirs("models", exist_ok=True)
        save_model(artifact, f"models/{ds_key}_detector.pkl")

    return {"detector": ds_key, "description": desc,
            "roc_auc": roc, "avg_precision": ap, "threshold": thr}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train all 10 fraud detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --list
  python train.py --dataset card --synthetic
  python train.py --dataset card --data-path data/raw/creditcard.csv --save
  python train.py --dataset card login aml --synthetic
  python train.py --all --synthetic --save
        """
    )
    p.add_argument("--dataset",   nargs="+", choices=list(DETECTOR_REGISTRY),
                   help="One or more detector keys")
    p.add_argument("--all",       action="store_true", help="Train all 10 detectors")
    p.add_argument("--list",      action="store_true", help="List available detectors")
    p.add_argument("--data-path", type=str, help="Path to real dataset CSV")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    p.add_argument("--save",      action="store_true", help="Save models to models/")
    p.add_argument("--n-samples", type=int, default=0,
                   help="Override default synthetic sample count")
    return p.parse_args()


def main():
    args = parse_args()

    # ── --list ───────────────────────────────────────────────────────────────
    if args.list:
        section("Available Fraud Detectors")
        print(f"  {'Key':15s} {'Description':35s} {'Fraud rate':12s} {'Real data':10s}")
        print(f"  {'─'*15} {'─'*35} {'─'*12} {'─'*10}")
        for key, (desc, fr, has_real) in DETECTOR_REGISTRY.items():
            print(f"  {key:15s} {desc:35s} {fr:.1%}       {'✓' if has_real else '─'}")
        print(f"\n  Run: python train.py --dataset <key> --synthetic")
        print(f"       python train.py --all --synthetic --save")
        return

    if not args.dataset and not args.all:
        print("ERROR: specify --dataset <key> or --all  (use --list to see options)")
        sys.exit(1)

    targets = list(DETECTOR_REGISTRY.keys()) if args.all else args.dataset
    results = []

    for ds_key in targets:
        desc, fraud_rate, has_real = DETECTOR_REGISTRY[ds_key]

        # ── Load data ─────────────────────────────────────────────────────
        if args.data_path and has_real and not args.synthetic:
            section(f"Loading real data for {desc}")
            if ds_key == "card":
                df, feat_cols = load_card_real(args.data_path)
            else:
                print(f"  No real loader for '{ds_key}' yet — using synthetic.")
                gen_fn = SYNTHETIC_GENERATORS[ds_key]
                n = args.n_samples if args.n_samples else None
                df, feat_cols = gen_fn() if not n else gen_fn(n=n, fraud_rate=fraud_rate)
        elif args.synthetic or not args.data_path:
            gen_fn = SYNTHETIC_GENERATORS[ds_key]
            n = args.n_samples if args.n_samples else None
            df, feat_cols = gen_fn() if not n else gen_fn(n=n, fraud_rate=fraud_rate)
        else:
            print(f"  Skipping {ds_key}: no --data-path and no --synthetic flag.")
            continue

        result = train_and_evaluate(ds_key, df, feat_cols, args)
        results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────────
    if len(results) > 1:
        section("Summary — All Detectors")
        print(f"  {'Detector':15s} {'Description':35s} {'ROC-AUC':9s} {'Avg-P':8s} {'Threshold':10s}")
        print(f"  {'─'*15} {'─'*35} {'─'*9} {'─'*8} {'─'*10}")
        for r in results:
            print(f"  {r['detector']:15s} {r['description']:35s} "
                  f"{r['roc_auc']:.4f}   {r['avg_precision']:.4f}   {r['threshold']:.4f}")

    if args.save:
        print(f"\n  Models saved in: models/")


if __name__ == "__main__":
    main()