import os
import sys
import shutil
import zipfile
from pathlib import Path

# Add current directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

def download_kaggle_data():
    input_dir = Path("./input")
    input_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = [
        'hintrngia/gate46-development-only-cxr',
        'hintrngia/gate7-test-seal',
        'hintrngia/cxr-lung-masks'
    ]
    
    print("[LOCAL SETUP] Checking Kaggle datasets...")
    for ds in datasets:
        ds_name = ds.split('/')[-1]
        marker = input_dir / ds_name
        if not marker.exists():
            print(f"[KAGGLE CLI] Downloading {ds} into ./input ...")
            ret = os.system(f"kaggle datasets download -d {ds} --unzip -p ./input")
            if ret == 0:
                marker.touch()
                print(f"[KAGGLE CLI] Successfully downloaded {ds_name}")
            else:
                print(f"[KAGGLE CLI] Warning: Kaggle CLI returned non-zero code {ret}. Make sure kaggle is installed and kaggle.json is in ~/.kaggle/ or %USERPROFILE%\\.kaggle\\")
        else:
            print(f"[KAGGLE CLI] Dataset {ds_name} already downloaded.")

def check_dependencies():
    required_checks = [('albumentations', 'albumentations'), ('pytorch_grad_cam', 'grad-cam'), ('cv2', 'opencv-python'), ('sklearn', 'scikit-learn')]
    missing = []
    for mod, pkg in required_checks:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
            
    if missing:
        print(f"[LOCAL SETUP] Installing missing packages: {missing} ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("[LOCAL SETUP] Dependencies installed successfully!")

def run_local():
    print("="*60)
    print("      LOCAL XAI & EVALUATION RUNNER")
    print("="*60)
    
    check_dependencies()
    
    # 1. Download Kaggle datasets if needed
    try:
        download_kaggle_data()
    except Exception as e:
        print(f"[LOCAL SETUP] Dataset download check failed: {e}")
        
    # 2. Check for models directory
    models_dir = Path("./outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    existing_pt = list(models_dir.glob("*.pt"))
    if not existing_pt:
        # Check fallback ./local_work/models
        fallback_dir = Path("./local_work/models")
        if fallback_dir.exists() and list(fallback_dir.glob("*.pt")):
            models_dir = fallback_dir
            existing_pt = list(models_dir.glob("*.pt"))

    if not existing_pt:
        print("\n[NOTICE] No .pt model files found in ./outputs/models/")
        print("Please copy your 6 trained model checkpoints (.pt files) into ./outputs/models/\n")
    else:
        print(f"\n[LOCAL SETUP] Found {len(existing_pt)} model checkpoints in {models_dir.absolute()}:")
        for pt in existing_pt:
            print(f"  - {pt.name}")

    # 3. Execute main pipeline
    print("\n[LOCAL RUN] Launching Pipeline...")
    src_dir = Path(__file__).parent / "src"
    files_to_stitch = [
        "01_environment.py",
        "02_config.py",
        "03_audit.py",
        "04_dataset.py",
        "05_dataloader.py",
        "06_models.py",
        "07_training.py",
        "08_validation.py",
        "09_calibration.py",
        "10_xai.py",
        "11_test_evaluation.py",
        "12_export.py"
    ]
    
    combined_code = []
    for fname in files_to_stitch:
        fpath = src_dir / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                code = f.read()
                combined_code.append(code)
                
    from build_notebook import MAIN_CODE
    combined_code.append(MAIN_CODE)
    
    full_script = "\n\n".join(combined_code)
    
    global_scope = {"__name__": "__main__", "__file__": str(Path(__file__).absolute())}
    exec(full_script, global_scope)

if __name__ == "__main__":
    run_local()
