import os
import json
import pandas as pd
from pathlib import Path

def setup_export_directories(work_dir: Path):
    """Creates the standard directory structure for artifacts."""
    dirs = {
        "models": work_dir / "models",
        "logs": work_dir / "logs",
        "plots": work_dir / "plots",
        "csv": work_dir / "csv",
        "json": work_dir / "json",
        "latex_tables": work_dir / "latex_tables",
        "figures": work_dir / "figures"
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    return dirs

def export_metrics(metrics: dict, architecture: str, seed: int, export_dirs: dict, stage: str = "test"):
    """
    Exports metrics to JSON and appends to a CSV summary.
    """
    # Save JSON
    json_path = export_dirs["json"] / f"{architecture}_seed{seed}_{stage}_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Append to CSV
    csv_path = export_dirs["csv"] / f"all_results_{stage}.csv"
    
    row = {"architecture": architecture, "seed": seed}
    row.update(metrics)
    
    df = pd.DataFrame([row])
    
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
        
    print(f"[EXPORT] Saved metrics for {architecture} (Seed: {seed}) to {json_path.name}")

def export_xai_results(xai_results: list, architecture: str, seed: int, export_dirs: dict):
    """
    Exports XAI LRR results to CSV.
    """
    if not xai_results:
        return
        
    df = pd.DataFrame(xai_results)
    csv_path = export_dirs["csv"] / f"{architecture}_seed{seed}_xai_lrr.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"[EXPORT] Saved XAI results to {csv_path.name}")
