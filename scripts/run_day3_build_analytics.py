import sys
from pathlib import Path

# add src to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bootcamp_data.transforms import (
    enforce_schema,
    parse_datetime,
    add_time_parts,
    iqr_bounds,
    winsorize,
    add_outlier_flag,
)
from bootcamp_data.joins import safe_left_join

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raw = ROOT / "data" / "raw"
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    # load
    orders = pd.read_csv(raw / "orders.csv")
    users = pd.read_csv(raw / "users.csv")

    # ✅ unify join key types
    users["user_id"] = users["user_id"].astype("string")

    # schema + datetime parts
    orders = enforce_schema(orders)
    orders = parse_datetime(orders, "created_at", utc=True)
    orders = add_time_parts(orders, "created_at")

    # outliers
    lo, hi = iqr_bounds(orders["amount"], k=1.5)
    orders = orders.assign(amount_winsor=winsorize(orders["amount"], lo, hi))
    orders = add_outlier_flag(orders, "amount", k=1.5)

    # join
    analytics = safe_left_join(orders, users, on="user_id", validate="many_to_one")

    # ✅ ترتيب (غيّر العمود إذا تبغى)
    analytics = analytics.sort_values("created_at", na_position="last").reset_index(drop=True)

    # ✅ طباعة كـ Table في التيرمنال
    cols_to_show = [
        "order_id",
        "user_id",
        "created_at",
        "amount",
        "amount_winsor",
        "quantity",
        "is_outlier",
        "status",
        "country",
        "signup_date",
    ]
    # خذ فقط الأعمدة اللي موجودة فعلاً (عشان ما يصير KeyError)
    cols_to_show = [c for c in cols_to_show if c in analytics.columns]

    pd.set_option("display.width", 2000)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_colwidth", 60)

    print("\n=== Analytics table (sorted) - first 20 rows ===")
    print(analytics.loc[:, cols_to_show].head(20).to_string(index=False))

    # write output
    out_path = processed / "analytics_table.parquet"
    analytics.to_parquet(out_path, index=False)
    print("\nWrote:", out_path)


if __name__ == "__main__":
    main()
