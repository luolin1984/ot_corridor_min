import os, json, argparse
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def build_net(backbone: str, classes: int = 2, in_channels: int = 5):
    enc = dict(resnet50="resnet50", convnext_tiny="timm-convnext_tiny")[backbone]
    return smp.FPN(encoder_name=enc, encoder_weights=None, in_channels=in_channels, classes=classes)

def norm_stack(R, G, B, CHM, SLP):
    x = np.stack([R, G, B, CHM, SLP], axis=0).astype(np.float32)  # (5,H,W)
    x[:3] /= 10000.0
    for k in (3,4):
        m = np.nanmean(x[k]); s = np.nanstd(x[k]) + 1e-6
        x[k] = (x[k] - m) / s
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--chm", required=True)
    ap.add_argument("--slope", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out_prob", required=True)
    ap.add_argument("--out_mask", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    meta = json.load(open(args.meta, "r"))
    backbone = meta.get("backbone", "resnet50")

    ds_rgb = rasterio.open(args.rgb)
    ds_chm = rasterio.open(args.chm)
    ds_slp = rasterio.open(args.slope)
    assert ds_rgb.transform == ds_chm.transform == ds_slp.transform, "RGB/CHM/SLOPE未对齐"
    H, W = ds_rgb.height, ds_rgb.width

    net = build_net(backbone=backbone).to(args.device)
    sd = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(sd, strict=False)
    net.eval()

    prob_sum = np.zeros((H, W), np.float32)
    prob_cnt = np.zeros((H, W), np.float32)

    xs = list(range(0, W - args.tile + 1, args.stride))
    ys = list(range(0, H - args.tile + 1, args.stride))

    for y in ys:
        for x in xs:
            w = Window(x, y, args.tile, args.tile)
            R, G, B = [ds_rgb.read(b, window=w) for b in (1,2,3)]
            CHM = ds_chm.read(1, window=w)
            SLP = ds_slp.read(1, window=w)
            arr = norm_stack(R, G, B, CHM, SLP)
            tin = torch.from_numpy(arr).unsqueeze(0).to(args.device)  # (1,5,t,t)
            logits = net(tin)
            p = F.softmax(logits, dim=1)[0,1].detach().cpu().numpy().astype(np.float32)  # (t,t)
            prob_sum[y:y+args.tile, x:x+args.tile] += p
            prob_cnt[y:y+args.tile, x:x+args.tile] += 1.0

    prob = prob_sum / np.maximum(prob_cnt, 1.0)
    mask = (prob >= args.thr).astype(np.uint8)

    prof = ds_rgb.profile.copy()
    prof.update(count=1, dtype="float32", compress="deflate", nodata=0.0)
    os.makedirs(os.path.dirname(args.out_prob), exist_ok=True)
    with rasterio.open(args.out_prob, "w", **prof) as dst:
        dst.write(prob, 1)

    prof_m = ds_rgb.profile.copy()
    prof_m.update(count=1, dtype="uint8", compress="deflate", nodata=0)
    with rasterio.open(args.out_mask, "w", **prof_m) as dst:
        dst.write(mask, 1)

    print("[OK] wrote:", args.out_prob)
    print("[OK] wrote:", args.out_mask)

if __name__ == "__main__":
    main()