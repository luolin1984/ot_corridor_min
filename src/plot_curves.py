import csv, matplotlib.pyplot as plt, os

csv_path = "outputs/model_ssl/train_log.csv"
epochs, sup, unsup, total, miou, acc = [], [], [], [], [], []
with open(csv_path, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        epochs.append(int(row["epoch"]))
        sup.append(float(row["sup_loss"]))
        unsup.append(float(row["unsup_loss"]))
        total.append(float(row["total_loss"]))
        miou.append(float(row["miou_sup"]))
        acc.append(float(row["acc_sup"]))

def plot(x,y,title,ylabel,out):
    plt.figure(); plt.plot(x,y,marker="o"); plt.grid(True,ls="--",alpha=.4)
    plt.title(title); plt.xlabel("epoch"); plt.ylabel(ylabel); plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True); plt.savefig(out,dpi=180); plt.close()

base = "outputs/model_ssl"
plot(epochs, sup,   "Supervised loss", "loss", f"{base}/curve_sup_loss.png")
plot(epochs, unsup, "Unsupervised loss", "loss", f"{base}/curve_unsup_loss.png")
plot(epochs, total, "Total loss", "loss", f"{base}/curve_total_loss.png")
plot(epochs, miou,  "mIoU (supervised tiles)", "mIoU", f"{base}/curve_miou.png")
plot(epochs, acc,   "Accuracy (supervised tiles)", "acc", f"{base}/curve_acc.png")
print("Saved curves under", base)
plt.show()
