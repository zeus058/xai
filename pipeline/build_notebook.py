import json
from pathlib import Path

MAIN_CODE = """
# ==========================================
# MAIN EXECUTION ORCHESTRATOR
# ==========================================
import os
import zipfile
import traceback
import pandas as pd
from sklearn.model_selection import train_test_split

def run_main_pipeline():
    is_kaggle, input_dir, work_dir = setup_environment(seed=CONFIG.seeds[0])
    export_dirs = setup_export_directories(work_dir)
    
    # Audit & Load Data
    dataset_root, test_df = audit_dataset(input_dir, work_dir)
    
    # ---- Step 1: Extract GATE46_DEVELOPMENT.g46blob (training images) ----
    dev_blob_root = None
    blob_matches = list(input_dir.rglob("GATE46_DEVELOPMENT.g46blob"))
    if len(blob_matches) > 0:
        print(f"\\n[ENV] Found development blob at {blob_matches[0]}. Extracting...")
        # CRITICAL FIX: Extract to local disk instead of Drive to prevent massive IO bottlenecks
        dev_blob_root = Path("/content/gate46_development") if not is_kaggle and Path("/content").exists() else work_dir / "gate46_development"
        try:
            # Count existing images to decide if re-extraction needed
            existing_images = list(dev_blob_root.rglob("*.jpeg")) + list(dev_blob_root.rglob("*.jpg")) if dev_blob_root.exists() else []
            if len(existing_images) < 100:
                import shutil
                if dev_blob_root.exists():
                    shutil.rmtree(dev_blob_root)
                dev_blob_root.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(blob_matches[0], 'r') as archive:
                    archive.extractall(dev_blob_root)
                # Verify extraction
                extracted_images = list(dev_blob_root.rglob("*.jpeg")) + list(dev_blob_root.rglob("*.jpg"))
                print(f"[ENV] Blob extracted: {len(extracted_images)} images in {dev_blob_root}")
                # Print top-level structure for diagnostics
                for item in sorted(dev_blob_root.iterdir()):
                    print(f"[ENV]   {item.name}/ ({len(list(item.rglob('*')))} items)" if item.is_dir() else f"[ENV]   {item.name}")
            else:
                print(f"[ENV] Blob already extracted: {len(existing_images)} images found in {dev_blob_root}")
        except Exception as e:
            print(f"[ENV] ERROR extracting blob: {e}")
            dev_blob_root = None
    else:
        print("[ENV] WARNING: GATE46_DEVELOPMENT.g46blob not found.")
    
    # ---- Step 2: Extract opaque_test_images.zip (test images) if needed ----
    test_zips = list(input_dir.rglob("opaque_test_images.zip"))
    if len(test_zips) > 0:
        try:
            print(f"\\n[ENV] Found opaque_test_images.zip. Extracting...")
            extracted_dir = Path("/content/test_images") if not is_kaggle and Path("/content").exists() else work_dir / "test_images"
            extracted_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(test_zips[0], 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
        except Exception as e:
            print(f"[ENV] WARNING: Could not extract test images zip: {e}")
    
    # ---- Step 3: Build image map across all known locations ----
    print("\\n[PIPELINE] Scanning for images across dataset...")
    image_map = {}
    valid_exts = {'.jpeg', '.jpg', '.png'}
    # Scan /kaggle/input (test images)
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_exts:
            if 'mask' in path.parent.name.lower():
                continue
            image_map[path.name] = path
    # Scan /kaggle/working (extracted blob images + test images) or local Colab disk
    search_dirs = [work_dir]
    if not is_kaggle and Path("/content").exists():
        search_dirs.extend([Path("/content/gate46_development"), Path("/content/test_images")])
        
    for s_dir in search_dirs:
        if s_dir.exists():
            for path in s_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in valid_exts:
                    if 'mask' in path.parent.name.lower():
                        continue
                    image_map[path.name] = path
    print(f"[PIPELINE] Found {len(image_map)} images total ({len(image_map)} unique filenames).")
    
    # ---- Step 4: Load development manifest ----
    dev_path = None
    if dev_blob_root is not None and dev_blob_root.exists():
        # Search recursively inside extracted blob for any development manifest
        all_csvs_in_blob = list(dev_blob_root.rglob("*.csv"))
        print(f"[PIPELINE] Found {len(all_csvs_in_blob)} CSVs inside extracted blob:")
        for c in all_csvs_in_blob:
            print(f"  - {c.relative_to(dev_blob_root)}")
        dev_matches = [p for p in all_csvs_in_blob if 'clean' in p.name.lower() and 'development' in p.name.lower()]
        if not dev_matches:
            dev_matches = [p for p in all_csvs_in_blob if 'development' in p.name.lower()]
        if dev_matches:
            dev_path = dev_matches[0]
            print(f"[PIPELINE] Loaded development manifest from blob: {dev_path.name}")
    
    if dev_path is None:
        # Fallback: search across all /kaggle/input
        all_csvs = list(input_dir.rglob("*.csv"))
        dev_matches = [p for p in all_csvs if 'development' in p.name.lower()]
        if dev_matches:
            clean_matches = [m for m in dev_matches if 'clean' in m.name.lower()]
            dev_path = clean_matches[0] if clean_matches else dev_matches[0]
            print(f"[PIPELINE] Found development manifest via fallback: {dev_path}")
    
    if dev_path is None:
        raise FileNotFoundError(
            "[FATAL] Development manifest not found in /kaggle/input! "
            "Cannot safely create Train/Val splits. Halting to prevent Data Leakage."
        )
    else:
        dev_df = pd.read_csv(dev_path)
        if 'final_split' in dev_df.columns:
            train_df = dev_df[dev_df['final_split'] == 'train'].reset_index(drop=True)
            val_df = dev_df[dev_df['final_split'] == 'tuning'].reset_index(drop=True)
            print(f"[PIPELINE] Using canonical split: {len(train_df)} train, {len(val_df)} val")
        else:
            train_df, val_df = train_test_split(dev_df, test_size=0.2, random_state=42,
                                                 stratify=dev_df.get('label_index', dev_df.get('label')))

    
    print("\\n[PIPELINE] Initializing Datasets and Dataloaders...")
    
    # ---- Find Lung Masks Directory ----
    lung_masks_dir = None
    mask_matches = list(input_dir.rglob("lung_masks"))
    if mask_matches:
        lung_masks_dir = mask_matches[0]
        print(f"[PIPELINE] Found lung masks at: {lung_masks_dir}")
    else:
        print("[PIPELINE] WARNING: lung_masks directory not found. Mask loss will use zero masks.")
        
    try:
        train_ds = ChestXRayDataset(train_df, image_map, CONFIG, split='train', mask_dir=lung_masks_dir)
        val_ds = ChestXRayDataset(val_df, image_map, CONFIG, split='val', mask_dir=lung_masks_dir)
    except Exception as e:
        print(f"[PIPELINE] ERROR building train/val datasets: {e}")
        raise
    try:
        test_ds = ChestXRayDataset(test_df, image_map, CONFIG, split='test', mask_dir=lung_masks_dir)
    except Exception as e:
        print(f"[PIPELINE] ERROR building test dataset: {e}")
        raise
    
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds, CONFIG, use_sampler=True)
    device = get_device()
    run_ledger = []  # Track success/failure of each (arch, seed) run
    
    for seed in CONFIG.seeds:
        setup_environment(seed)
        
        for arch in CONFIG.models_to_train:
            run_key = f"{arch}_seed{seed}"
            print(f"\\n{'='*50}\\nSTARTING PIPELINE: {arch} | SEED: {seed}\\n{'='*50}")
            
            try:
                # 1. Model & Training
                model = ClassificationModel(
                    arch, 
                    freeze_percent=CONFIG.densenet_freeze_percent if arch=="DenseNet121" else CONFIG.resnet_freeze_percent,
                    use_cbam=CONFIG.use_cbam
                )
                model = train_pipeline(model, train_loader, val_loader, device, CONFIG, seed)
                
                # 2. Calibration
                print("\\n[PIPELINE] Running Calibration...")
                try:
                    temperature = calibrate_model(model, val_loader, device)
                except Exception as e:
                    print(f"[CALIBRATION] WARNING: Calibration failed ({e}). Using temperature=1.0")
                    temperature = 1.0
                
                _, val_probs, val_labels = validate(model, val_loader, nn.BCEWithLogitsLoss(), device, CONFIG)
                val_probs_flat = val_probs.squeeze()
                val_labels_flat = val_labels.squeeze()
                optimal_threshold = find_optimal_threshold(val_labels_flat, val_probs_flat, metric='youden')
                print(f"[PIPELINE] Optimal Threshold: {optimal_threshold:.4f}")
                
                # 3. Test Evaluation (Single Pass)
                test_metrics, test_preds, test_labels_out = evaluate_test_set(model, test_loader, device, CONFIG, optimal_threshold, temperature)
                export_metrics(test_metrics, arch, seed, export_dirs, stage="test")
                
                # 4. XAI (Optional, requires lung masks)
                if lung_masks_dir is not None and lung_masks_dir.exists():
                    try:
                        # Convert optimal_threshold (probability) to logit space for XAI
                        import math
                        p = max(1e-6, min(1 - 1e-6, optimal_threshold))
                        logit_threshold = math.log(p / (1 - p))
                        xai_results = generate_explanations(
                            model, test_loader, device, lung_masks_dir,
                            save_figures=True, output_dir=export_dirs["figures"], max_figures=20,
                            threshold=logit_threshold, max_samples=100
                        )
                        export_xai_results(xai_results, arch, seed, export_dirs)
                    except Exception as e:
                        print(f"[XAI] WARNING: XAI failed for {arch} seed {seed}: {e}. Skipping XAI step.")
                else:
                    print(f"[XAI] Skipping XAI: lung_masks_dir not found.")
                
                # 5. Save model checkpoint
                try:
                    model_path = export_dirs["models"] / f"{arch}_seed{seed}_best.pt"
                    torch.save(model.state_dict(), model_path)
                    print(f"[EXPORT] Model saved to {model_path.name}")
                except Exception as e:
                    print(f"[EXPORT] WARNING: Could not save model: {e}")
                
                run_ledger.append({"run": run_key, "status": "SUCCESS", "auroc": test_metrics.get("auroc", "N/A"), "accuracy": test_metrics.get("accuracy", "N/A")})
                print(f"\\n[LEDGER] ✓ {run_key} completed successfully.")
                
            except Exception as e:
                error_msg = traceback.format_exc()
                print(f"\\n[LEDGER] ✗ {run_key} FAILED: {e}")
                print(f"[LEDGER] Full traceback:\\n{error_msg}")
                run_ledger.append({"run": run_key, "status": "FAILED", "error": str(e)})
                # Continue to next (arch, seed) — do NOT abort the whole pipeline
                continue
            
            finally:
                # Always free GPU memory after each run regardless of success/failure
                try:
                    del model
                    torch.cuda.empty_cache()
                except:
                    pass
    
    # ---- Final Summary Report ----
    print("\\n" + "="*60)
    print("[PIPELINE] === FINAL RUN SUMMARY ===")
    successful = [r for r in run_ledger if r["status"] == "SUCCESS"]
    failed = [r for r in run_ledger if r["status"] == "FAILED"]
    print(f"  Total runs attempted: {len(run_ledger)}")
    print(f"  ✓ Successful: {len(successful)}")
    print(f"  ✗ Failed:     {len(failed)}")
    for r in run_ledger:
        if r["status"] == "SUCCESS":
            print(f"    ✓ {r['run']:35s} | AUROC={r['auroc']:.4f} | ACC={r['accuracy']:.4f}")
        else:
            print(f"    ✗ {r['run']:35s} | ERROR: {r['error']}")
    
    # Save ledger to CSV
    pd.DataFrame(run_ledger).to_csv(export_dirs["csv"] / "run_ledger.csv", index=False)
    print(f"\\n[PIPELINE] Run ledger saved to csv/run_ledger.csv")
    
    # Aggregating XAI Metrics
    try:
        xai_csvs = list(export_dirs["csv"].glob("*_xai_lrr.csv"))
        if xai_csvs:
            print("\\n[PIPELINE] === AGGREGATING XAI METRICS (Grad-CAM vs Guided Grad-CAM) ===")
            xai_dfs = []
            for f in xai_csvs:
                parts = f.stem.split("_")
                arch = parts[0]
                seed = parts[1].replace("seed", "")
                df_item = pd.read_csv(f)
                df_item["arch"] = arch
                df_item["seed"] = seed
                xai_dfs.append(df_item)
            
            combined_xai = pd.concat(xai_dfs, ignore_index=True)
            combined_xai.to_csv(export_dirs["csv"] / "XAI_ALL_SAMPLES_COMBINED.csv", index=False)
            
            xai_summary = combined_xai.groupby("arch").agg(
                gcam_lrr_mean=("gcam_lrr_plus", "mean"),
                gcam_lrr_std=("gcam_lrr_plus", "std"),
                guided_gcam_lrr_mean=("guided_gcam_lrr_plus", "mean"),
                guided_gcam_lrr_std=("guided_gcam_lrr_plus", "std"),
                lrr_chance_mean=("lrr_chance", "mean"),
                gcam_del_auc_mean=("gcam_del_auc", "mean"),
                guided_del_auc_mean=("guided_del_auc", "mean"),
                gcam_ins_auc_mean=("gcam_ins_auc", "mean"),
                guided_ins_auc_mean=("guided_ins_auc", "mean"),
                gcam_stab_mean=("gcam_stab", "mean"),
                guided_stab_mean=("guided_stab", "mean"),
                gcam_rand_mean=("gcam_rand", "mean"),
                guided_rand_mean=("guided_rand", "mean")
            ).reset_index()
            
            xai_summary.to_csv(export_dirs["csv"] / "XAI_AGGREGATED_SUMMARY.csv", index=False)
            print("[PIPELINE] XAI Aggregated Summary:")
            for _, row in xai_summary.iterrows():
                print(f"  Architecture: {row['arch']:15s} | Grad-CAM LRR+: {row['gcam_lrr_mean']:.4f} ± {row['gcam_lrr_std']:.4f} | Guided Grad-CAM LRR+: {row['guided_gcam_lrr_mean']:.4f} ± {row['guided_gcam_lrr_std']:.4f} (Chance: {row['lrr_chance_mean']:.4f})")
                print(f"    Grad-CAM -> Del AUC: {row['gcam_del_auc_mean']:.4f} | Ins AUC: {row['gcam_ins_auc_mean']:.4f} | Stab: {row['gcam_stab_mean']:.4f} | Rand: {row['gcam_rand_mean']:.4f}")
                print(f"    Guided   -> Del AUC: {row['guided_del_auc_mean']:.4f} | Ins AUC: {row['guided_ins_auc_mean']:.4f} | Stab: {row['guided_stab_mean']:.4f} | Rand: {row['guided_rand_mean']:.4f}")
    except Exception as e:
        print(f"[PIPELINE] WARNING: Could not aggregate XAI metrics: {e}")
        
    print("\\n[PIPELINE] All Tasks Completed Successfully!")
"""

def create_notebook(output_path: Path):
    """
    Stitches all python modules into a single Jupyter Notebook.
    """
    src_dir = Path("src")
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
    
    cells = []
    
    # 1. Install dependencies cell
    pip_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q albumentations grad-cam\n"
        ]
    }
    cells.append(pip_cell)
    
    colab_setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "\n",
            "# Automatically mount Google Drive and download Kaggle datasets if running in Colab\n",
            "if 'google.colab' in sys.modules:\n",
            "    print('[ENV] Google Colab detected. Setting up workspace...')\n",
            "    import torch\n",
            "    if not torch.cuda.is_available():\n",
            "        print('[ENV] NOTICE: GPU not detected. Running in CPU mode for Evaluation & XAI...')\n",
            "    else:\n",
            "        print('[ENV] GPU detected. Running on GPU acceleration.')\n",
            "    \n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive')\n",
            "    \n",
            "    kaggle_json = '/content/drive/MyDrive/kaggle.json'\n",
            "    if not os.path.exists(kaggle_json):\n",
            "        print('[ENV] kaggle.json not found in Drive. Please upload it now:')\n",
            "        from google.colab import files\n",
            "        uploaded = files.upload()\n",
            "        if 'kaggle.json' in uploaded:\n",
            "            os.system('mv kaggle.json /content/drive/MyDrive/kaggle.json')\n",
            "            print('[ENV] Saved kaggle.json to your Google Drive for future use!')\n",
            "        else:\n",
            "            print('[FATAL] You must upload kaggle.json to download datasets!')\n",
            "            sys.exit(1)\n",
            "            \n",
            "    if os.path.exists(kaggle_json):\n",
            "        print('[ENV] Setting up Kaggle API...')\n",
            "        os.makedirs('/root/.kaggle', exist_ok=True)\n",
            "        os.system(f'cp {kaggle_json} /root/.kaggle/')\n",
            "        os.system('chmod 600 /root/.kaggle/kaggle.json')\n",
            "        \n",
            "        print('[ENV] Downloading datasets from Kaggle (this may take a few minutes)...')\n",
            "        os.makedirs('/content/input', exist_ok=True)\n",
            "        os.chdir('/content/input')\n",
            "        \n",
            "        # Download datasets\n",
            "        datasets = [\n",
            "            'hintrngia/gate46-development-only-cxr',\n",
            "            'hintrngia/gate7-test-seal',\n",
            "            'hintrngia/cxr-lung-masks'\n",
            "        ]\n",
            "        \n",
            "        for ds in datasets:\n",
            "            ds_name = ds.split('/')[-1]\n",
            "            if not os.path.exists(ds_name):\n",
            "                print(f'Downloading {ds}...')\n",
            "                os.system(f'kaggle datasets download -d {ds} --unzip -p /content/input')\n",
            "                os.system(f'touch {ds_name}')\n",
            "            else:\n",
            "                print(f'{ds_name} already exists.')\n",
            "        \n",
            "        print('[ENV] Workspace setup complete!')\n",
            "        os.chdir('/content')\n"
        ]
    }
    cells.append(colab_setup_cell)
    
    # 2. Add each module
    for filename in files_to_stitch:
        filepath = src_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found.")
            continue
            
        with open(filepath, "r") as f:
            code = f.read()
            
        markdown_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## Module: {filename}"]
        }
        cells.append(markdown_cell)
        
        code_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.split("\n")]
        }
        cells.append(code_cell)
        
    # 3. Add Main execution cell
    main_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in MAIN_CODE.split("\n")]
    }
    cells.append(main_cell)
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
        
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    create_notebook(Path("Pneumonia_XAI_Pipeline.ipynb"))
