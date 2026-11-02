"""Run panel DML-DID for one or more studies.

    python run_dml.py configs/demo_synthetic.yaml
    python run_dml.py configs/*.yaml
    python run_dml.py configs/broadband_investment.yaml --learners ridge elastic_net --n-rep 5
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from dml_did import load_config, run_study
from dml_did.config import REPO_ROOT
from dml_did.learners import LEARNERS
from dml_did.report import save_results, to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="YAML config file(s); glob patterns are allowed")
    parser.add_argument("--learners", nargs="+", choices=list(LEARNERS), help="override the learners in the config")
    parser.add_argument("--n-rep", type=int, help="override the number of cross-fitting repetitions")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results", help="output directory")
    args = parser.parse_args()

    paths = sorted({p for pattern in args.configs for p in glob.glob(pattern)})
    if not paths:
        parser.error(f"no config files match {args.configs}")

    for path in paths:
        cfg = load_config(path)
        if args.learners:
            cfg.learners = args.learners
        if args.n_rep:
            cfg.n_rep = args.n_rep

        if not cfg.data_path.exists():
            print(f"[{cfg.name}] skipped: data file not found at {cfg.data_path}")
            continue

        results, info = run_study(cfg)
        save_results(
            results,
            info,
            name=cfg.name,
            title=cfg.description or cfg.name,
            out_dir=args.out,
            true_effect=cfg.true_effect,
        )

        print(f"\n[{cfg.name}] results\n{to_markdown(results)}\n")
        print(f"[{cfg.name}] saved to {args.out / cfg.name}.{{csv,md,png}}")


if __name__ == "__main__":
    main()
