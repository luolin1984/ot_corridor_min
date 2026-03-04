#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json
import numpy as np
import laspy
from sklearn.metrics import classification_report, confusion_matrix

def main():
    ap = argparse.ArgumentParser("Evaluate point-wise labels")
    ap.add_argument("--gt_las", required=True)
    ap.add_argument("--pred_las", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    gt = laspy.read(args.gt_las)
    pr = laspy.read(args.pred_las)
    if gt.header.point_count != pr.header.point_count:
        raise RuntimeError("Point counts differ; ensure prediction kept original ordering.")

    y_true = np.asarray(gt.classification).astype(int)
    if "PredLabel" in pr.point_format.extra_dimension_names:
        y_pred = np.asarray(pr["PredLabel"]).astype(int)
    else:
        y_pred = np.asarray(pr.classification).astype(int)

    labels_sorted = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels_sorted, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted).tolist()
    out = {"labels": labels_sorted, "report": report, "confusion_matrix": cm}

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"OK: metrics -> {args.out_json}")

if __name__ == "__main__":
    main()
