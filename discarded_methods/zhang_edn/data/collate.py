"""Custom collation for variable-length RF data batches."""

from __future__ import annotations

from typing import Dict, List

import torch


def collate_variable_length(batch: List[Dict]) -> Dict:
    """
    Zero-pad RF data to max length in batch.

    Use when mixing datasets with different n_samples::

        loader = DataLoader(dataset, collate_fn=collate_variable_length)
    """
    max_d = max(item["rf_data"].shape[1] for item in batch)

    padded_rf = []
    for item in batch:
        rf = item["rf_data"]
        K, D = rf.shape
        if D < max_d:
            rf = torch.cat([rf, torch.zeros(K, max_d - D)], dim=1)
        padded_rf.append(rf)

    return {
        "rf_data": torch.stack(padded_rf),
        "angle_deg": torch.stack([item["angle_deg"] for item in batch]),
        "sequence_name": [item["sequence_name"] for item in batch],
        "category": [item["category"] for item in batch],
        "subcategory": [item["subcategory"] for item in batch],
    }
