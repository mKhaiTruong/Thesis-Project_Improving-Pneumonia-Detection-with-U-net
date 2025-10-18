import os, glob, re
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.2)

# ----------------------
# Parse SAM-Adapter logs
# ----------------------
def parse_sam_adapter_logs(log_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        with open(f, "r", encoding="utf-8") as infile:
            for line in infile:
                m = re.search(
                    r"Epoch (\d+) took ([0-9.]+) minutes \| Train Loss = ([0-9.]+).*GPU usage = c_nvmlMemory_t\(total: [0-9]+ B, free: ([0-9]+) B, used: [0-9]+ B\).*CPU usage = ([0-9.]+)%.*Metrics = ({.*})",
                    line
                )
                if m:
                    epoch, time_min, train_loss, gpu_free, cpu, metrics_str  = m.groups()
                    metrics = eval(metrics_str.replace("'", '"'))  # {'iou': '0.7878', 'dice': '0.8585'}
                    rows.append({
                        "epoch": int(epoch),
                        "time_min": float(time_min),
                        "train_loss": float(train_loss),
                        "iou": float(metrics["iou"]),
                        "dice": float(metrics["dice"]),
                        "gpu_usage": 100 - (float(gpu_free) / 1e9 * 100),  # convert B to GB % used
                        "cpu_usage": float(cpu)
                    })
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["cumulative_time"] = df["time_min"].cumsum()
    return df

# ----------------------
# Log directory
# ----------------------
log_dir = r"..\PneunomiaSeg\SAM\CUS SAM\results\models\sam_adapter_exp1\logs"

df = parse_sam_adapter_logs(log_dir)
if df is None:
    raise ValueError("No valid logs found!")

# ----------------------
# Visualization
# ----------------------
metrics = ["train_loss", "iou", "dice", "gpu_usage", "cpu_usage", "cumulative_time"]
titles  = {
    "train_loss": "Training Loss",
    "iou": "IOU Score",
    "dice": "Dice Score",
    "gpu_usage": "GPU Usage (%)",
    "cpu_usage": "CPU Usage (%)",
    "cumulative_time": "Cumulative Training Time (minutes)"
}

colors = sns.color_palette("tab10", len(metrics))

for i, metric in enumerate(metrics):
    plt.figure(figsize=(7, 5))
    plt.plot(
        df["epoch"], df[metric],
        label=f"SAM-Adapter",
        color=colors[i % len(colors)],
        linewidth=2,
        marker="o", markersize=4
    )
    plt.title(titles[metric])
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
