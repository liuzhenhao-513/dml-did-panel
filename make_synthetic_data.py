"""Write the synthetic demo panel to data/synthetic_did.csv."""

from dml_did.config import REPO_ROOT
from dml_did.synthetic import make_did_panel

if __name__ == "__main__":
    out = REPO_ROOT / "data" / "synthetic_did.csv"
    out.parent.mkdir(exist_ok=True)
    df = make_did_panel()
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows ({df['firm_id'].nunique()} firms) to {out}")
