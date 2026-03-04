from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Splitter:
    cv: object
    groups: np.ndarray | None

    def split(self, x: pd.DataFrame, y: np.ndarray):
        if self.groups is None:
            yield from self.cv.split(x, y)
        else:
            yield from self.cv.split(x, y, groups=self.groups)


def make_splitter(
    *,
    cv: str,
    n_splits: int,
    seed: int,
    groups_df: pd.DataFrame | None,
) -> Splitter:
    if cv == "stratified":
        from sklearn.model_selection import StratifiedKFold

        return Splitter(
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
            groups=None,
        )

    if cv == "group_ab":
        if groups_df is None:
            raise ValueError("groups_df is required for cv='group_ab'")
        from sklearn.model_selection import GroupKFold

        groups = (groups_df["Group_A"].astype("int64") * 1_000_000) + groups_df[
            "Group_B"
        ].astype("int64")
        return Splitter(cv=GroupKFold(n_splits=n_splits), groups=groups.to_numpy())

    raise ValueError(f"Unknown cv={cv!r}")

