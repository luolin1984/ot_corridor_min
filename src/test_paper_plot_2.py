import matplotlib
matplotlib.use("TkAgg")  # 如果你要保存图不用显示的话
import matplotlib.pyplot as plt
import csv

def read_csv(path):
    e, loss = [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            e.append(int(row["epoch"]))
            loss.append(float(row["unsup_loss"]))
    return e, loss

base = "outputs"
m1_e, m1_l = read_csv(f"../{base}/model_ssl/train_log_prop.csv")         # PIF-SV
m2_e, m2_l = read_csv(f"../{base}/model_ssl_lidar/train_log.csv")   # lidar_only
m3_e, m3_l = read_csv(f"../{base}/model_ssl_rgb/train_log.csv")     # rgb_only
m4_e, m4_l = read_csv(f"../{base}/model_sup_fusion/train_log.csv")  # supervise_fusion

fig, ax1 = plt.subplots(figsize=(7, 4))

# 左轴：三个小量级的
ax1.plot(m1_e, m1_l, "o-", label="PIF-SV")
ax1.plot(m2_e, m2_l, "s--", label="lidar_only")
ax1.plot(m3_e, m3_l, "d-", label="rgb_only")
ax1.set_xlabel("迭代轮次")
ax1.set_ylabel("损失值（半监督类）")
ax1.grid(True, ls="--", alpha=0.4)

# 右轴：一个大量级的
ax2 = ax1.twinx()
ax2.plot(m4_e, m4_l, "v:", color="crimson", label="supervise_fusion")
ax2.set_ylabel("损失值（监督融合）")

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.tight_layout()
#plt.savefig("paper_total_loss_dual.png", dpi=300, bbox_inches="tight")
plt.show()
