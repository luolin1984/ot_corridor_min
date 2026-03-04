#1、读取 5 通道栅格（RGB+CHM+Slope），切成重叠小块（默认 512×512）。
#2、监督样本：你现在可直接用风险栅格 outputs/aoi_risk_hag_gt6m.tif当作弱标注（1/0），它覆盖“高于阈值的植被/越线风险”，先训练单类前景（最小改造）。
#   若你后续有更细粒度标签（如导线/地线/塔材/金具），这个脚本也支持多类（只要把标签做成多类整数栅格）。
#无监督样本：同域未标注区域，在训练中用弱增强/强增强的一致性约束 + 教师 EMA 伪标签阈值控制。
#损失：监督端 Focal + Lovász-Softmax，无监督端 KL，总损失加权。
import os, math, json, random, argparse, csv, time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ---------- Lovasz-Softmax (简化版，多类可用；二类同样适用) ----------
def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    if gts == 0:
        return torch.zeros_like(gt_sorted)
    p = len(gt_sorted)
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probs, labels):
    C = probs.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if len(losses) == 0:
        return torch.tensor(0., device=probs.device)
    return torch.mean(torch.stack(losses))

def lovasz_softmax(probs, labels):
    probs = probs.permute(0,2,3,1).contiguous().view(-1, probs.size(1))
    labels = labels.view(-1)
    return lovasz_softmax_flat(probs, labels)

# ---------- Focal ----------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        focal = (self.alpha * (1-pt)**self.gamma) * ce
        mask = (target != self.ignore_index).float()
        return (focal * mask).sum() / mask.sum().clamp(min=1)

# ---------- 数据集（读 5 通道：RGB+CHM+Slope；标签可选） ----------
class TileDataset(Dataset):
    def __init__(self, rgb_path, chm_path, slope_path, label_path=None, tile=512, stride=512, augment=False, ignore_index=255):
        self.rgb = rasterio.open(rgb_path)
        self.chm = rasterio.open(chm_path)
        self.slp = rasterio.open(slope_path)
        assert self.rgb.transform == self.chm.transform == self.slp.transform, "RGB/CHM/SLOPE 地理参考不一致"
        self.W = self.rgb.width
        self.H = self.rgb.height
        self.tile = tile
        self.stride = stride
        self.label_path = label_path
        self.ignore_index = ignore_index
        self.au = augment
        if label_path and os.path.exists(label_path):
            self.lab = rasterio.open(label_path)
            assert self.lab.transform == self.rgb.transform, "Label 与影像未对齐"
        else:
            self.lab = None
        xs = list(range(0, self.W - tile + 1, stride))
        ys = list(range(0, self.H - tile + 1, stride))
        self.windows = [(x, y) for y in ys for x in xs]
        self.t_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(p=0.1),
        ]) if augment else None

    def __len__(self): return len(self.windows)

    def __getitem__(self, i):
        x, y = self.windows[i]
        w = Window(x, y, self.tile, self.tile)
        R,G,B = [self.rgb.read(b, window=w) for b in (1,2,3)]
        CHM = self.chm.read(1, window=w)
        SLP = self.slp.read(1, window=w)
        img = np.stack([R,G,B,CHM,SLP], axis=2).astype(np.float32)
        # 简单归一化：RGB / 10000, CHM 与 SLP 标准化
        img[..., :3] = img[..., :3] / 10000.0
        for j in (3,4):
            m = np.nanmean(img[..., j]); s = np.nanstd(img[..., j]) + 1e-6
            img[..., j] = (img[..., j]-m)/s

        if self.t_aug is not None:
            out = self.t_aug(image=img)
            img = out['image']

        img = torch.from_numpy(img).permute(2,0,1).contiguous()

        if self.lab is None:
            lab = torch.full((self.tile, self.tile), fill_value=self.ignore_index, dtype=torch.long)
        else:
            yv = self.lab.read(1, window=w)
            # 风险栅格是 0/1：把 1 作为前景类（class=1），背景 0
            yv = np.where(yv==1, 1, 0).astype(np.int64)
            lab = torch.from_numpy(yv)

        return img, lab

# ---------- EMA ----------
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self):
        msd = self.model.state_dict()
        for k, v in msd.items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
                continue
            if not torch.is_floating_point(v):
                self.shadow[k].copy_(v)
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

def build_net(backbone='resnet50', classes=2):
    enc = dict(resnet50='resnet50', convnext_tiny='timm-convnext_tiny')[backbone]
    model = smp.FPN(
        encoder_name=enc,
        encoder_weights=None,
        in_channels=5,     # RGB+CHM+Slope
        classes=classes
    )
    return model

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb', default='data/imagery/aoi_rgb.tif')
    ap.add_argument('--chm', default='outputs/aoi_chm.tif')
    ap.add_argument('--slope', default='outputs/aoi_slope.tif')
    ap.add_argument('--label', default='outputs/aoi_risk_hag_gt6m.tif')  # 弱标注
    ap.add_argument('--outdir', default='outputs/model_ssl')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--tile', type=int, default=512)
    ap.add_argument('--stride', type=int, default=512)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--backbone', default='resnet50', choices=['resnet50','convnext_tiny'])
    ap.add_argument('--pseudo_thr', type=float, default=0.7)
    ap.add_argument('--lambda_u', type=float, default=0.5)  # 无监督权重
    # ---- 新增：训练日志 ----
    ap.add_argument('--log_csv', default=None, help='保存训练曲线的 CSV（默认 outdir/train_log.csv）')
    ap.add_argument('--tb', action='store_true', help='启用 TensorBoard 日志（可选）')
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 日志文件准备
    log_csv = args.log_csv or os.path.join(args.outdir, "train_log.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","sup_loss","unsup_loss","total_loss","miou_sup","acc_sup","time_sec"])

    # TensorBoard（可选）
    tb = None
    if args.tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=os.path.join(args.outdir, "tb"))
        except Exception as e:
            print("[WARN] TensorBoard 初始化失败：", e)

    sup_ds = TileDataset(args.rgb, args.chm, args.slope, label_path=args.label,
                         tile=args.tile, stride=args.stride, augment=True)
    unsup_ds = TileDataset(args.rgb, args.chm, args.slope, label_path=None,
                           tile=args.tile, stride=args.stride, augment=True)
    sup_dl = DataLoader(sup_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    uns_dl = DataLoader(unsup_ds, batch_size=args.batch, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_s = build_net(args.backbone).to(device)
    net_t = build_net(args.backbone).to(device)
    net_t.load_state_dict(net_s.state_dict(), strict=True)
    ema = EMA(net_t, decay=0.99)

    opt = torch.optim.AdamW(net_s.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    focal = FocalLoss(gamma=2.0, alpha=0.25)
    ignore = 255

    for ep in range(1, args.epochs+1):
        t0 = time.time()
        net_s.train(); net_t.eval()
        it_uns = iter(uns_dl)

        # 累计器
        sup_loss_sum = 0.0
        unsup_loss_sum = 0.0
        total_loss_sum = 0.0
        inter_sum = 0
        union_sum = 0
        correct_sum = 0
        count_sum = 0

        for imgs, labs in sup_dl:
            imgs = imgs.to(device); labs = labs.to(device)
            # ---- 监督损失
            logits = net_s(imgs)
            loss_sup = focal(logits, labs) + lovasz_softmax(F.softmax(logits,1), labs)

            # ---- 无监督一致性（teacher -> pseudo）
            try:
                u_imgs, _ = next(it_uns)
            except StopIteration:
                it_uns = iter(uns_dl)
                u_imgs, _ = next(it_uns)
            u_imgs = u_imgs.to(device)
            with torch.no_grad():
                pseudo = F.softmax(net_t(u_imgs), 1)
                conf, plab = pseudo.max(1)
                mask = (conf >= args.pseudo_thr).float()   # [N,H,W]
            stu_logits = net_s(u_imgs)
            l_kl = F.kl_div(F.log_softmax(stu_logits,1), pseudo, reduction='none').sum(1)  # [N,H,W]
            loss_uns = (l_kl * mask).mean()

            loss = loss_sup + args.lambda_u*loss_uns
            opt.zero_grad(); loss.backward(); opt.step()

            # EMA 更新 teacher（对 net_t 的影子进行更新）
            ema.update()

            # 统计监督批次的 IoU/Acc（忽略 255）
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)[:,1]  # 前景通道
                pred  = (probs > 0.5)
                valid = (labs != ignore)
                if valid.any():
                    yb = (labs == 1)
                    tp = (pred & yb & valid).sum().item()
                    fp = (pred & (~yb) & valid).sum().item()
                    fn = ((~pred) & yb & valid).sum().item()
                    inter_sum += tp
                    union_sum += (tp + fp + fn)
                    correct_sum += (pred.eq(yb) & valid).sum().item()
                    count_sum  += valid.sum().item()

            # 累计损失
            sup_loss_sum += float(loss_sup.item())
            unsup_loss_sum += float(loss_uns.item())
            total_loss_sum += float(loss.item())

        sched.step()

        miou_sup = (inter_sum / union_sum) if union_sum > 0 else 0.0
        acc_sup  = (correct_sum / count_sum) if count_sum > 0 else 0.0
        tsec = time.time() - t0

        # 写 CSV
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, sup_loss_sum, unsup_loss_sum, total_loss_sum, miou_sup, acc_sup, tsec])

        # TensorBoard（可选）
        if tb is not None:
            tb.add_scalar("loss/supervised",   sup_loss_sum,  ep)
            tb.add_scalar("loss/unsupervised", unsup_loss_sum,ep)
            tb.add_scalar("loss/total",        total_loss_sum,ep)
            tb.add_scalar("metric/mIoU_sup",   miou_sup,      ep)
            tb.add_scalar("metric/acc_sup",    acc_sup,       ep)

        print(f"[SSL] epoch {ep}/{args.epochs} done. "
              f"loss_sup={sup_loss_sum:.3f} loss_unsup={unsup_loss_sum:.3f} "
              f"miou_sup={miou_sup:.3f} acc_sup={acc_sup:.3f}")

    # 训练完成：把 EMA 权重拷回 teacher（更平滑）
    ema.apply_to(net_t)
    torch.save(net_t.state_dict(), os.path.join(args.outdir, 'fpn_ssl.pth'))
    meta = dict(backbone=args.backbone, in_channels=5, classes=2, tile=args.tile, stride=args.stride)
    json.dump(meta, open(os.path.join(args.outdir,'meta.json'),'w'), indent=2, ensure_ascii=False)
    print("[OK] Saved:", args.outdir, "\nCSV:", log_csv, "\nTensorBoard:", os.path.join(args.outdir, "tb"))

if __name__ == '__main__':
    main()
