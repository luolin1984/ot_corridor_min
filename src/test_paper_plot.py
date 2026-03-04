import matplotlib
matplotlib.use("TkAgg")  # 没 GUI 就留着，有的话可以删
import csv, os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示为方块的问题

# 这四个路径按你的实际目录改
logs = {
    "PIF-SV":        "../outputs/model_ssl/train_log_prop.csv",
    "lidar_only":      "../outputs/model_ssl_lidar/train_log.csv",
    "rgb_only":        "../outputs/model_ssl_rgb/train_log.csv",
    "supervise_fusion":      "../outputs/model_sup_fusion/train_log.csv",
}

# 想展示的列
target_col = "miou_sup"   # 换成 "miou_sup" 就是mIoU图
                            # total_loss：整体优化效果
                            # sup_loss：监督分支学得怎么样
                            # unsup_loss：半监督一致性学得怎么样
                            # miou_sup：监督样本上的 mIoU
                            # acc_sup：监督样本上的像素精度

# 给每个方法分配风格
styles = {
    "PIF-SV":   {"color": "C0", "linestyle": "-",  "marker": "o"},
    "lidar_only": {"color": "C1", "linestyle": "--", "marker": "s"},
    "rgb_only":   {"color": "C2", "linestyle": "-.", "marker": "d"},
    "supervise_fusion": {"color": "C3", "linestyle": ":",  "marker": "^"},
}

plt.figure(figsize=(6, 4))

for name, path in logs.items():
    if not os.path.exists(path):
        print(f"[WARN] {name} log not found: {path}")
        continue

    epochs, vals = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            vals.append(float(row[target_col]))

    st = styles.get(name, {})
    plt.plot(
        epochs,
        vals,
        label=name,
        color=st.get("color"),
        linestyle=st.get("linestyle"),
        marker=st.get("marker"),
    )

plt.xlabel("迭代轮次")
plt.ylabel("mIoU值")
#plt.title("Total loss comparison")
plt.grid(True, ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
#plt.savefig("../paper_unsup_loss_all.png", dpi=300, bbox_inches="tight")
plt.show()
print("[OK] saved ../paper_unsup_loss_all.png")
