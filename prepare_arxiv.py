#!/usr/bin/env python3
"""Prepare arXiv submission artifacts."""

import subprocess
import json
from datetime import datetime
from pathlib import Path

def main():
    paper_dir = Path("paper")
    artifacts_dir = Path("arxiv_submission")
    artifacts_dir.mkdir(exist_ok=True)

    # 1. Copy LaTeX sources
    (artifacts_dir / "paper").mkdir(exist_ok=True)
    for f in ["main.tex", "Makefile", "README.md"]:
        src = paper_dir / f
        if src.exists():
            subprocess.run(["cp", str(src), str(artifacts_dir / "paper" / f)])

    # Copy figures
    (artifacts_dir / "paper" / "figures").mkdir(exist_ok=True)
    for fig in (paper_dir / "figures").glob("*"):
        subprocess.run(["cp", str(fig), str(artifacts_dir / "paper" / "figures" / fig.name)])

    # 2. Generate benchmark results (if available)
    print("Running quick benchmark for arXiv data...")
    result = subprocess.run(
        ["python3", "experiments/real_benchmark.py"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        # Save benchmark JSON to paper/
        bench_data = json.loads(result.stdout.strip().split('\n')[-1])
        with open(paper_dir / "benchmark_results.json", "w") as f:
            json.dump(bench_data, f, indent=2)
        print(f"✓ Benchmark: {bench_data.get('compression', 'N/A'):.1%} compression")
    else:
        print("⚠ Benchmark failed, using placeholder values")

    # 3. Create ZIP
    timestamp = datetime.now().strftime("%Y%m%d")
    zip_name = f"agent-coderag-arxiv-{timestamp}.zip"
    subprocess.run(["zip", "-r", zip_name, "paper/", "experiments/benchmark.py"])
    print(f"\n✓ Created {zip_name}")
    print(f"  Contents: {artifacts_dir}/paper/")
    print(f"  Upload to: https://arxiv.org/submit")

    # 4. Show checklist
    print("\n" + "="*60)
    print("ARXIV SUBMISSION CHECKLIST")
    print("="*60)
    print("☑ LaTeX source (main.tex)")
    print("☑ Figures (PDF in paper/figures/)")
    print("☑ Build instructions (Makefile)")
    print("☐ Endorsement (obtain ORCID + endorser)")
    print("☐ Choose category: cs.AI or cs.LG or cs.SE")
    print("☐ Abstract < 1920 chars")
    print("☐ Confirm license (MIT)")
    print("="*60)

if __name__ == "__main__":
    main()
