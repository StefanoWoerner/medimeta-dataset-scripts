import os

import pandas as pd
import yaml

from config import config as cfg


task_target_mapping = {
    "MULTICLASS_CLASSIFICATION": "multi-class classification",
    "BINARY_CLASSIFICATION": "binary classification",
    "MULTILABEL_CLASSIFICATION": "multi-label classification",
    "ORDINAL_REGRESSION": "ordinal regression",
    "REGRESSION": "regression",
    "BINARY_SEGMENTATION": "binary segmentation",
    "MULTILABEL_SEGMENTATION": "multi-label segmentation",
}


def main():
    dataset_ids = [
        d
        for d in os.listdir(cfg.unified_data_base_path)
        if os.path.isdir(os.path.join(cfg.unified_data_base_path, d))
    ]

    columns = [
        "Dataset Name",
        "Domain",
        "\\# Training Images",
        "\\# Validation Images",
        "\\# Test Images",
        "Task Name",
        "Task Target",
        "\\# Labels",
    ]
    rows = []

    for dataset_id in sorted(dataset_ids):
        # load random images
        info_path = os.path.join(cfg.unified_data_base_path, dataset_id, "info.yaml")
        with open(info_path, "r") as f:
            info = yaml.safe_load(f)
        dataset_name = info["name"]
        domain = info["domain"]
        num_train = info["splits_num_samples"]["train"]
        num_val = info["splits_num_samples"]["val"]
        num_test = info["splits_num_samples"]["test"]
        for task_info in info["tasks"]:
            task_name = task_info["task_name"]
            task_target = task_info["task_target"].replace(" ", "_")
            num_labels = len(task_info["labels"])
            rows.append(
                [
                    dataset_name,
                    domain,
                    num_train,
                    num_val,
                    num_test,
                    task_name,
                    task_target,
                    num_labels,
                ]
            )

    df = pd.DataFrame(rows, columns=columns)
    df["Task Target"] = df["Task Target"].map(task_target_mapping)

    # format so that common information between tasks appears as a multirow cell in latex
    df = df.set_index(
        [
            "Dataset Name",
            "Domain",
            "\\# Training Images",
            "\\# Validation Images",
            "\\# Test Images",
            "Task Name",
        ]
    )
    latex = df.style.to_latex(hrules=True)
    print(latex)


if __name__ == "__main__":
    main()
