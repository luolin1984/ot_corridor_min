# 功能：把训练好的 FPN-SSL 用滑窗方式跑全图，输出 outputs/semseg_risk.tif（0/1 前景）。
import os, json, argparse
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def build(backbone, classes=2):
    return smp.FPN(encoder_name={'resnet50':'resnet50','convnext_tiny':'timm-convnext_tiny'}[backbone],
                   encoder_weights=None, in_channels=5, classes=classes)

def load_5ch(rgb, chm, slope, window):
    with rasterio.open(rgb) as R:
        R1,R2,R3 = [R.read(b, window=window) for b in (1,2,3)]
        tr = R.transform; crs=R.crs; prof=R.profile
    with rasterio.open(chm) as C:
        CHM = C.read(1, window=window)
    with rasterio.open(slope) as S:
        SLP = S.read(1, window=window)
    img = np.stack([R1,R2,R3,CHM,SLP],2).astype(np.float32)
    img[:,:,:3] /= 10000.0
    for j in (3,4):
        m = np.nanmean(img[...,j]); s=np.nanstd(img[...,j])+1e-6
        img[...,j]=(img[...,j]-m)/s
    return img.transpose(2,0,1).copy(), tr, crs, prof

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb', default='data/imagery/aoi_rgb.tif')
    ap.add_argument('--chm', default='outputs/aoi_chm.tif')
    ap.add_argument('--slope', default='outputs/aoi_slope.tif')
    ap.add_argument('--model', default='outputs/model_ssl/fpn_ssl.pth')
    ap.add_argument('--meta', default='outputs/model_ssl/meta.json')
    ap.add_argument('--out', default='outputs/semseg_risk.tif')
    ap.add_argument('--tile', type=int, default=512)
    ap.add_argument('--stride', type=int, default=512)
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    net = build(meta['backbone'], classes=meta.get('classes',2))
    net.load_state_dict(torch.load(args.model, map_location='cpu'), strict=True)
    net.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    with rasterio.open(args.rgb) as R:
        W,H = R.width, R.height
        tr, crs, prof = R.transform, R.crs, R.profile

    pred = np.zeros((H,W), dtype=np.uint8)
    tile, stride = args.tile, args.stride
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            w = Window(x,y,tile,tile)
            img,_,_,_ = load_5ch(args.rgb, args.chm, args.slope, w)
            t = torch.from_numpy(img).unsqueeze(0).to(device)
            with torch.no_grad():
                p = F.softmax(net(t),1)[0,1].cpu().numpy()
            pred[y:y+tile, x:x+tile] = (p>=0.5).astype(np.uint8)

    prof.update(count=1, dtype='uint8', compress='lzw')
    with rasterio.open(args.out, 'w', **prof) as dst:
        dst.write(pred, 1)
    print("[OK] Wrote", args.out)

if __name__ == '__main__':
    main()
