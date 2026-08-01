from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
RESULTS_DIR = Path("results")
def main():
    summary_files = sorted(RESULTS_DIR.glob("*_summary.csv"))
    if not summary_files:
        print("No summary files found in results/. Run scripts/02_run_experiment.py for each config first.")
        return
    frames = [pd.read_csv(f) for f in summary_files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("f1", ascending=False)
    out_path = RESULTS_DIR / "comparison_summary.csv"
    combined.to_csv(out_path, index=False)
    print(combined.to_string(index=False))
    print(f"\nWritten to {out_path}")
if __name__ == "__main__":
    main()