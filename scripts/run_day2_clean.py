# scripts/run_day2_clean.py

import logging
import sys
from pathlib import Path

# make `src/` importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bootcamp_data.config import make_paths
from bootcamp_data.io import read_orders_csv, read_users_csv, write_parquet
from bootcamp_data.transforms import (
    enforce_schema,
    missingness_report,
    add_missing_flags,
    normalize_text,
    apply_mapping,
)
from bootcamp_data.quality import (
    require_columns,
    assert_non_empty,
    assert_in_range,
)

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Paths
    p = make_paths(ROOT)

    log.info("Loading raw inputs")
    orders_raw = read_orders_csv(p.raw / "orders.csv")
    users = read_users_csv(p.raw / "users.csv")
    log.info("Rows: orders_raw=%s, users=%s", len(orders_raw), len(users))

  
    require_columns(
        orders_raw,
        ["order_id", "user_id", "amount", "quantity", "created_at", "status"],
    )
    require_columns(users, ["user_id", "country", "signup_date"])

    assert_non_empty(orders_raw, "orders_raw")
    assert_non_empty(users, "users")

 
    orders = enforce_schema(orders_raw)

   
    rep = missingness_report(orders)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rep_path = reports_dir / "missingness_orders.csv"
    rep.to_csv(rep_path, index=True)
    log.info("Wrote missingness report: %s", rep_path)

    
    status_norm = normalize_text(orders["status"])
    mapping = {
        "paid": "paid",
        "refund": "refund",
        "refunded": "refund",
    }
    status_clean = apply_mapping(status_norm, mapping)

    # -------------------------------------------------
    # Build cleaned orders
    # -------------------------------------------------
    orders_clean = (
        orders
        .assign(status_clean=status_clean)
        .pipe(add_missing_flags, cols=["amount", "quantity"])
    )

    # -------------------------------------------------
    # Range checks (fail fast)
    # -------------------------------------------------
    assert_in_range(orders_clean["quantity"], lo=0, name="quantity")
    assert_in_range(orders_clean["amount"], lo=0, name="amount")

    # -------------------------------------------------
    # Write outputs
    # -------------------------------------------------
    write_parquet(orders_clean, p.processed / "orders_clean.parquet")
    write_parquet(users, p.processed / "users.parquet")
    log.info("Wrote processed outputs: %s", p.processed)


if __name__ == "__main__":
    main()