import os
import json
import pandas as pd
from pathlib import Path
import hashlib

def sha256_file(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()

def audit_dataset(input_dir: Path, work_dir: Path):
    """
    Validates dataset integrity by checking ASSET_MANIFEST.json 
    and ensuring test cases count matches.
    """
    print("[AUDIT] Starting Dataset Audit...")
    audit_results = {}
    
    # Locate ASSET_MANIFEST.json (Optional but recommended)
    asset_manifest_matches = list(input_dir.rglob("ASSET_MANIFEST.json"))
    dataset_root = input_dir
    
    if len(asset_manifest_matches) == 0:
        print("[AUDIT] WARNING: ASSET_MANIFEST.json not found in /kaggle/input.")
    else:
        dataset_root = asset_manifest_matches[0].parent
        with open(asset_manifest_matches[0], 'r') as f:
            manifest = json.load(f)
            print(f"[AUDIT] Loaded ASSET_MANIFEST.json from {asset_manifest_matches[0]}")
            audit_results['manifest'] = "OK"

    # Independently locate SEALED test labels (Required)
    test_labels_matches = list(input_dir.rglob("SEALED_test_label_key.csv"))
    if len(test_labels_matches) == 0:
        raise FileNotFoundError(
            "Could not locate SEALED_test_label_key.csv in /kaggle/input.\n"
            "Did you forget to attach the dataset containing the test labels using 'Add Data'?"
        )
    
    test_label_path = test_labels_matches[0]
    # If we didn't set dataset_root from manifest, use the test label's parent
    if len(asset_manifest_matches) == 0:
        dataset_root = test_label_path.parent
        
    test_df = pd.read_csv(test_label_path)
    test_count = len(test_df)
    print(f"[AUDIT] Found {test_count} cases in SEALED_test_label_key.csv")
    
    if test_count < 600:
        print(f"[AUDIT] WARNING: Test count ({test_count}) is unusually low. Expected ~624.")
    
    audit_results['test_cases'] = test_count
    audit_results['dataset_root'] = str(dataset_root)
    
    # Save audit log
    audit_log_path = work_dir / "audit.csv"
    pd.DataFrame([audit_results]).to_csv(audit_log_path, index=False)
    print(f"[AUDIT] Audit Complete. Saved to {audit_log_path}")
    
    return dataset_root, test_df

if __name__ == "__main__":
    import importlib.util
    env_path = Path(__file__).with_name("01_environment.py")
    spec = importlib.util.spec_from_file_location("pipeline_environment", env_path)
    env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_module)
    setup_environment = env_module.setup_environment
    is_k, in_dir, w_dir = setup_environment()
    audit_dataset(in_dir, w_dir)
