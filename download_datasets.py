"""
download_datasets.py — Real Dataset Downloader
================================================
Downloads real datasets for all 10 fraud detectors.

Auto-download (no credentials needed):
  card         → creditcard.csv from GitHub

Kaggle download (requires kaggle API key):
  loan         → home-credit-default-risk
  aml          → paysim1 (synthetic financial transactions)
  merchant     → ieee-fraud-detection
  synthetic_id → (uses credit card default UCI dataset)
  atm          → (uses creditcard.csv + ATM filter)

Registration required (manual download):
  login/mobile → CICIDS 2017 (UNB)
  insider      → CERT Insider Threat Dataset (CMU SEI)
  wire         → Uses synthetic (no public dataset)

Usage
-----
  # Download everything possible automatically
  python download_datasets.py --auto

  # Download with Kaggle API (requires ~/.kaggle/kaggle.json)
  python download_datasets.py --kaggle

  # Download specific datasets
  python download_datasets.py --datasets card loan aml

  # Show status of all datasets
  python download_datasets.py --status
"""

import argparse
import os
import sys
import time
import ssl
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Registry
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "card": {
        "description": "Credit Card Fraud (ULB/Kaggle)",
        "method":      "direct",
        "url":         "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv",
        "output":      "data/raw/creditcard.csv",
        "size_mb":     98,
        "rows":        284_807,
        "fraud_rate":  "0.17%",
        "label_col":   "Class",
        "detectors":   ["card"],
    },
    "paysim": {
        "description": "PaySim Financial Transactions (AML proxy)",
        "method":      "kaggle",
        "kaggle_id":   "ealaxi/paysim1",
        "kaggle_file": "PS_20174392719_1491204439457_log.csv",
        "output":      "data/raw/paysim.csv",
        "size_mb":     470,
        "rows":        6_362_620,
        "fraud_rate":  "0.13%",
        "label_col":   "isFraud",
        "detectors":   ["aml"],
    },
    "home_credit": {
        "description": "Home Credit Default Risk (Loan fraud proxy)",
        "method":      "kaggle",
        "kaggle_id":   "competitions/home-credit-default-risk",
        "kaggle_file": "application_train.csv",
        "output":      "data/raw/home_credit.csv",
        "size_mb":     166,
        "rows":        307_511,
        "fraud_rate":  "8.07%",
        "label_col":   "TARGET",
        "detectors":   ["loan"],
    },
    "ieee_fraud": {
        "description": "IEEE-CIS Fraud Detection (Card + Merchant)",
        "method":      "kaggle",
        "kaggle_id":   "competitions/ieee-fraud-detection",
        "kaggle_file": "train_transaction.csv",
        "output":      "data/raw/ieee_fraud.csv",
        "size_mb":     450,
        "rows":        590_540,
        "fraud_rate":  "3.5%",
        "label_col":   "isFraud",
        "detectors":   ["card", "merchant"],
    },
    "give_credit": {
        "description": "Give Me Some Credit (Loan default)",
        "method":      "kaggle",
        "kaggle_id":   "competitions/GiveMeSomeCredit",
        "kaggle_file": "cs-training.csv",
        "output":      "data/raw/give_credit.csv",
        "size_mb":     20,
        "rows":        150_000,
        "fraud_rate":  "6.7%",
        "label_col":   "SeriousDlqin2yrs",
        "detectors":   ["loan", "synthetic_id"],
    },
    "cert_insider": {
        "description": "CERT Insider Threat Dataset v6.2",
        "method":      "manual",
        "url":         "https://www.cmu.edu/sei/our-work/cert-insider-threat",
        "output":      "data/raw/cert_insider/",
        "size_mb":     500,
        "rows":        None,
        "fraud_rate":  "~2%",
        "label_col":   "insider",
        "detectors":   ["insider"],
        "instructions": [
            "1. Go to: https://www.cmu.edu/sei/our-work/cert-insider-threat",
            "2. Register and request dataset access (CERT Insider Threat Dataset v6.2)",
            "3. Download and extract to data/raw/cert_insider/",
        ],
    },
    "cicids": {
        "description": "CICIDS 2017 (Login anomaly / Account Takeover)",
        "method":      "manual",
        "url":         "https://www.unb.ca/cic/datasets/ids-2017.html",
        "output":      "data/raw/cicids/",
        "size_mb":     8_000,
        "rows":        2_830_743,
        "fraud_rate":  "~20% (attack traffic)",
        "label_col":   "Label",
        "detectors":   ["login", "mobile"],
        "instructions": [
            "1. Go to: https://www.unb.ca/cic/datasets/ids-2017.html",
            "2. Download 'CICIDS2017.zip' (requires registration)",
            "3. Extract to data/raw/cicids/",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'═'*62}\n  {title}\n{'═'*62}")

def check_kaggle():
    """Check if kaggle CLI is configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        return False, "~/.kaggle/kaggle.json not found"
    try:
        result = subprocess.run(["kaggle", "--version"],
                                capture_output=True, text=True, timeout=5)
        return result.returncode == 0, result.stdout.strip()
    except FileNotFoundError:
        return False, "kaggle CLI not installed (pip install kaggle)"

def file_exists(path):
    p = Path(path)
    return p.exists() and (p.stat().st_size > 1024 if p.is_file() else any(p.iterdir()))

def download_direct(url, output_path, size_mb):
    """Download file directly from URL with progress."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    downloaded = [0]
    start = time.time()

    def progress(count, block_size, total):
        downloaded[0] = count * block_size
        mb = downloaded[0] / 1e6
        pct = (downloaded[0] / total * 100) if total > 0 else 0
        speed = downloaded[0] / (time.time() - start + 0.01) / 1e6
        print(f"\r  {mb:.1f} / {total/1e6:.0f} MB  ({pct:.0f}%)  {speed:.1f} MB/s", end="", flush=True)

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    urllib.request.urlretrieve(url, output_path, reporthook=progress)
    print(f"\r  Downloaded {Path(output_path).stat().st_size / 1e6:.1f} MB → {output_path}")

def download_kaggle(kaggle_id, kaggle_file, output_path):
    """Download dataset using kaggle CLI."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("data/raw/_tmp_kaggle")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    is_competition = kaggle_id.startswith("competitions/")
    if is_competition:
        comp_name = kaggle_id.replace("competitions/", "")
        cmd = ["kaggle", "competitions", "download", "-c", comp_name,
               "-f", kaggle_file, "-p", str(tmp_dir)]
    else:
        cmd = ["kaggle", "datasets", "download", "-d", kaggle_id,
               "-f", kaggle_file, "-p", str(tmp_dir), "--unzip"]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        return False

    # Move file to final location
    downloaded = list(tmp_dir.glob("**/*.csv"))
    if downloaded:
        shutil.move(str(downloaded[0]), output_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  Moved → {output_path}")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Column adapters: map real dataset cols → detector feature names
# ─────────────────────────────────────────────────────────────────────────────

def adapt_paysim(path):
    """
    PaySim → AML detector features.
    Columns: step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,
             nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
    """
    import pandas as pd, numpy as np
    print("  Adapting PaySim → AML features...")
    df = pd.read_csv(path, nrows=200_000)  # 200k rows for speed

    adapted = pd.DataFrame()
    adapted["amount"]               = df["amount"]
    adapted["is_cash"]              = df["type"].isin(["CASH_OUT","CASH_IN"]).astype(int)
    adapted["is_wire"]              = (df["type"] == "TRANSFER").astype(int)
    adapted["is_crypto"]            = np.zeros(len(df), int)
    adapted["tx_count_48h"]         = np.random.default_rng(42).integers(1, 6, len(df))
    adapted["structured_count_30d"] = np.zeros(len(df), int)
    adapted["round_amount"]         = (df["amount"] % 1000 == 0).astype(int)
    adapted["account_hops"]         = (df["type"] == "TRANSFER").astype(int) + 1
    adapted["beneficiaries_7d"]     = np.random.default_rng(42).integers(1, 8, len(df))
    adapted["sender_risk"]          = np.where(df["oldbalanceOrg"] == 0, 0.8, 0.1)
    adapted["receiver_risk"]        = np.where(df["oldbalanceDest"] == 0, 0.6, 0.1)
    adapted["cross_border"]         = np.zeros(len(df), int)
    adapted["high_risk_country"]    = np.zeros(len(df), int)
    adapted["is_shell"]             = (df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]).abs() < 1
    adapted["is_shell"]             = adapted["is_shell"].astype(int)
    adapted["pep"]                  = np.zeros(len(df), int)
    adapted["kyc_risk"]             = np.where(df["amount"] > 200_000, 0.8, 0.1)
    adapted["deviation"]            = np.clip(df["amount"] / (df["oldbalanceOrg"] + 1), 0, 1)
    adapted["unusual_biz"]          = (df["type"] == "CASH_OUT").astype(int)
    adapted["label"]                = df["isFraud"]
    adapted["log_amount"]           = np.log1p(df["amount"])
    adapted["near_ctr"]             = df["amount"].between(8000, 9999).astype(int)
    adapted["geo_risk"]             = adapted["sender_risk"]
    adapted["entity_risk"]          = adapted["is_shell"] * 0.5 + adapted["kyc_risk"] * 0.5

    out = path.replace(".csv", "_adapted.csv")
    adapted.to_csv(out, index=False)
    print(f"  Saved adapted → {out} ({len(adapted):,} rows, {adapted['label'].sum():,} fraud)")
    return out

def adapt_home_credit(path):
    """Home Credit → Loan detector features."""
    import pandas as pd, numpy as np
    print("  Adapting Home Credit → Loan features...")
    df = pd.read_csv(path, nrows=100_000)

    adapted = pd.DataFrame()
    adapted["lti_ratio"]        = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    adapted["credit_score"]     = np.where(df["EXT_SOURCE_2"].notna(), df["EXT_SOURCE_2"] * 850, 600).astype(int)
    adapted["employ_months"]    = df["DAYS_EMPLOYED"].abs().fillna(0) / 30
    adapted["address_months"]   = df["DAYS_REGISTRATION"].abs().fillna(0) / 30
    adapted["apps_30d"]         = df.get("AMT_REQ_CREDIT_BUREAU_MON", pd.Series(np.zeros(len(df)))).fillna(0)
    adapted["doc_completeness"] = 1 - df.filter(like="FLAG_DOC").mean(axis=1).fillna(0)
    adapted["inquiries_6m"]     = df.get("AMT_REQ_CREDIT_BUREAU_QRT", pd.Series(np.zeros(len(df)))).fillna(0)
    adapted["income_verified"]  = (df["NAME_INCOME_TYPE"] != "Unemployed").astype(int)
    adapted["label"]            = df["TARGET"]

    out = path.replace(".csv", "_adapted.csv")
    adapted.to_csv(out, index=False)
    print(f"  Saved adapted → {out} ({len(adapted):,} rows, {adapted['label'].sum():,} fraud)")
    return out

def adapt_ieee_fraud(path):
    """IEEE-CIS → Card detector features."""
    import pandas as pd, numpy as np
    print("  Adapting IEEE-CIS → Card features...")
    df = pd.read_csv(path, nrows=200_000)

    adapted = pd.DataFrame()
    adapted["merchant_risk"]    = np.random.default_rng(42).uniform(0, 0.3, len(df))
    adapted["tx_count_1h"]      = np.random.default_rng(42).integers(0, 5, len(df))
    adapted["tx_count_24h"]     = np.random.default_rng(42).integers(1, 10, len(df))
    adapted["country_mismatch"] = np.zeros(len(df), int)
    adapted["card_present"]     = (df.get("card6", "credit") == "debit").astype(int)
    adapted["is_new_merchant"]  = np.zeros(len(df), int)
    adapted["declined_24h"]     = np.zeros(len(df), int)
    adapted["log_amount"]       = np.log1p(df["TransactionAmt"])
    adapted["is_night"]         = ((df["TransactionDT"] // 3600) % 24).between(0, 5).astype(int)
    adapted["amount_vs_avg"]    = df["TransactionAmt"] / (df["TransactionAmt"].mean() + 1)
    adapted["label"]            = df["isFraud"]

    out = path.replace(".csv", "_adapted.csv")
    adapted.to_csv(out, index=False)
    print(f"  Saved adapted → {out} ({len(adapted):,} rows, {adapted['label'].sum():,} fraud)")
    return out


ADAPTERS = {
    "paysim":      adapt_paysim,
    "home_credit": adapt_home_credit,
    "ieee_fraud":  adapt_ieee_fraud,
}


# ─────────────────────────────────────────────────────────────────────────────
# Status display
# ─────────────────────────────────────────────────────────────────────────────

def show_status():
    section("Dataset Status")
    kaggle_ok, kaggle_msg = check_kaggle()

    print(f"  Kaggle CLI : {'✓ ' + kaggle_msg if kaggle_ok else '✗  ' + kaggle_msg}")
    print()

    header = f"  {'Dataset':16s} {'Description':42s} {'Status':12s} {'Size':8s} {'Detectors'}"
    print(header)
    print("  " + "─" * 100)

    for key, info in DATASETS.items():
        exists = file_exists(info["output"])
        if exists:
            status = "✓ ready"
        elif info["method"] == "direct":
            status = "→ auto"
        elif info["method"] == "kaggle":
            status = "→ kaggle" if kaggle_ok else "✗ need key"
        else:
            status = "✗ manual"

        size = f"{info['size_mb']:,} MB" if info["size_mb"] < 1000 else f"{info['size_mb']/1000:.1f} GB"
        dets = ", ".join(info["detectors"])
        print(f"  {key:16s} {info['description']:42s} {status:12s} {size:8s} {dets}")

    print()
    print("  Detectors with NO public dataset → use --synthetic:")
    print("    wire, atm, mobile  (synthetic data is sufficient for these)")


# ─────────────────────────────────────────────────────────────────────────────
# Download functions
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(key, force=False, kaggle_ok=False):
    info = DATASETS[key]
    output = info["output"]

    if file_exists(output) and not force:
        size = Path(output).stat().st_size / 1e6 if Path(output).is_file() else 0
        print(f"  [{key}] Already exists ({size:.0f} MB) — skip. Use --force to re-download.")
        return True

    print(f"\n  [{key}] {info['description']}")

    if info["method"] == "direct":
        try:
            print(f"  Downloading from GitHub ({info['size_mb']} MB)...")
            download_direct(info["url"], output, info["size_mb"])
            return True
        except Exception as e:
            print(f"  ERROR: {e}")
            return False

    elif info["method"] == "kaggle":
        if not kaggle_ok:
            print(f"  SKIP: Kaggle credentials not found.")
            print(f"  Run:  pip install kaggle")
            print(f"        # Get API key from kaggle.com → Account → API → Create Token")
            print(f"        mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
            print(f"  Then: kaggle {'competitions' if 'competitions' in info['kaggle_id'] else 'datasets'} download "
                  f"{'--competition ' + info['kaggle_id'].replace('competitions/','') if 'competitions' in info['kaggle_id'] else '-d ' + info['kaggle_id']} "
                  f"-f {info['kaggle_file']} -p data/raw/")
            return False
        try:
            return download_kaggle(info["kaggle_id"], info["kaggle_file"], output)
        except Exception as e:
            print(f"  ERROR: {e}")
            return False

    elif info["method"] == "manual":
        print(f"  MANUAL DOWNLOAD REQUIRED:")
        for step in info.get("instructions", []):
            print(f"    {step}")
        return False

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Run adapters on downloaded data
# ─────────────────────────────────────────────────────────────────────────────

def run_adapters(datasets_downloaded):
    section("Adapting Real Datasets → Detector Features")
    for key in datasets_downloaded:
        if key in ADAPTERS:
            info = DATASETS[key]
            if file_exists(info["output"]):
                print(f"\n  [{key}]")
                try:
                    out = ADAPTERS[key](info["output"])
                    print(f"  ✓ Adapted: {out}")
                except Exception as e:
                    print(f"  ✗ Adapter failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Verify downloads and show train commands
# ─────────────────────────────────────────────────────────────────────────────

def show_train_commands():
    section("Ready to Train — Commands")

    commands = {
        "card (real)":         "python train.py --dataset card --data-path data/raw/creditcard.csv --save",
        "aml (real)":          "python train.py --dataset aml  --data-path data/raw/paysim_adapted.csv --save",
        "loan (real)":         "python train.py --dataset loan --data-path data/raw/home_credit_adapted.csv --save",
        "card+merchant (IEEE)":"python train.py --dataset card merchant --data-path data/raw/ieee_fraud_adapted.csv --save",
        "all (synthetic)":     "python train.py --all --synthetic --save",
        "mix (real+synthetic)":"python train.py --dataset card --data-path data/raw/creditcard.csv --save && python train.py --dataset login loan insider aml wire synthetic_id atm mobile merchant --synthetic --save",
    }
    for desc, cmd in commands.items():
        print(f"\n  # {desc}")
        print(f"  {cmd}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Download real datasets for all 10 fraud detectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py --status
  python download_datasets.py --auto
  python download_datasets.py --kaggle
  python download_datasets.py --all
  python download_datasets.py --datasets card paysim
        """
    )
    p.add_argument("--status",   action="store_true", help="Show download status of all datasets")
    p.add_argument("--auto",     action="store_true", help="Download only auto-downloadable datasets (no credentials)")
    p.add_argument("--kaggle",   action="store_true", help="Download Kaggle datasets (requires ~/.kaggle/kaggle.json)")
    p.add_argument("--all",      action="store_true", help="Download all datasets (auto + kaggle)")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), help="Download specific datasets")
    p.add_argument("--adapt",    action="store_true", help="Run column adapters on downloaded data")
    p.add_argument("--force",    action="store_true", help="Re-download even if file exists")
    return p.parse_args()


def main():
    args = parse_args()

    if args.status or (not any([args.auto, args.kaggle, args.all, args.datasets])):
        show_status()
        show_train_commands()
        return

    kaggle_ok, kaggle_msg = check_kaggle()
    downloaded = []

    targets = []
    if args.datasets:
        targets = args.datasets
    elif args.all:
        targets = list(DATASETS.keys())
    elif args.kaggle:
        targets = [k for k, v in DATASETS.items() if v["method"] in ("direct", "kaggle")]
    elif args.auto:
        targets = [k for k, v in DATASETS.items() if v["method"] == "direct"]

    section(f"Downloading {len(targets)} dataset(s)")

    for key in targets:
        ok = download_dataset(key, force=args.force, kaggle_ok=kaggle_ok)
        if ok:
            downloaded.append(key)

    if args.adapt or downloaded:
        run_adapters(downloaded)

    section("Summary")
    for key in targets:
        info = DATASETS[key]
        exists = file_exists(info["output"])
        status = "✓ ready" if exists else "✗ missing"
        print(f"  {key:16s} {status}")

    show_train_commands()


if __name__ == "__main__":
    main()