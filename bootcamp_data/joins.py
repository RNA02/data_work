import pandas as pd

def safe_left_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    *,
    validate: str = "many_to_one",
    suffixes: tuple[str, str] = ("", "_right"),
) -> pd.DataFrame:
    out = left.merge(
        right,
        how="left",
        on=on,
        validate=validate,
        suffixes=suffixes,
    )
    if len(out) != len(left):
        raise ValueError(f"Row count changed after left join: {len(left)} -> {len(out)}")
    return out
