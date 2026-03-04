#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def main():
    ap = argparse.ArgumentParser("Train a simple RF on sampled points")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--out_meta", default=None)
    ap.add_argument("--n_estimators", type=int, default=200)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X, y = data["X"], data["y"].astype(int)
    feat_names = list(data["feat_names"])

    # 简单下采样/权重：用 class_weight='balanced' 交给 RF 处理
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    clf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, class_weight="balanced", random_state=0)
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    print(classification_report(yte, ypred, digits=3))

    # 保存
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.out_model)
    meta = {
        "feat_names": feat_names,
        "labels_note": {"ground":2, "vegetation":5, "wire":14, "tower":15}
    }
    meta_path = args.out_meta or (str(Path(args.out_model).with_suffix(".json")))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"OK: model -> {args.out_model}\nOK: meta  -> {meta_path}")

if __name__ == "__main__":
    main()
