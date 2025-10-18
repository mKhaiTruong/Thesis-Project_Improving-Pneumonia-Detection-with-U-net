import os, glob, re
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.2)

# ----------------------
# Parse training logs
# ----------------------
def parse_unet_logs(log_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        with open(f, "r", encoding="utf-8") as infile:
            for line in infile:
                m = re.search(
                    r"Epoch (\d+).*took ([0-9.]+) minutes.*Train Loss = ([0-9.]+).*Valid Loss = ([0-9.]+).*Current IOU score = ([0-9.]+)",
                    line,
                )
                if m:
                    epoch, time_min, train_loss, valid_loss, iou = m.groups()
                    rows.append({
                        "epoch": int(epoch),
                        "time_min": float(time_min),
                        "train_loss": float(train_loss),
                        "valid_loss": float(valid_loss),
                        "iou": float(iou),
                    })
    if not rows:
        return None
    
    df = pd.DataFrame(rows)
    df["cumulative_time"] = df["time_min"].cumsum()
    return df

# ----------------------
# Model log directories
# ----------------------
model_results = {
    'deeplabv3+_b3' : r'..\PneunomiaSeg\Segmentation\results\models\unet_deeplabv3plus_efficientnet-b3_exp1\logs',
    'deeplabv3+_mit': r'..\PneunomiaSeg\Segmentation\results\models\unet_deeplabv3plus_mit_b1_exp1\logs',
    'segformer_b3'  : r'..\PneunomiaSeg\Segmentation\results\models\unet_segformer_efficientnet-b3_exp1\logs',
    'segformer_mit' : r'..\PneunomiaSeg\Segmentation\results\models\unet_segformer_mit_b1_exp1\logs',
    'unetpp_b3'     : r'..\PneunomiaSeg\Segmentation\results\models\unet_unetpp_efficientnet-b3_exp1\logs',
}

# ----------------------
# Load all logs
# ----------------------
df_dict = {}
for name, path in model_results.items():
    df = parse_unet_logs(path)
    if df is not None:
        df_dict[name] = df

if not df_dict:
    raise ValueError("No valid logs found!")

# ----------------------
# Visualization
# ----------------------
metrics = ["train_loss", "valid_loss", "iou", "cumulative_time"]
titles  = {
    "train_loss": "Training Loss",
    "valid_loss": "Validation Loss",
    "iou": "IOU Score",
    "cumulative_time": "Cumulative Training Time (minutes)"
}

colors = sns.color_palette("tab10", len(df_dict))

for metric in metrics:
    plt.figure(figsize=(7, 5))
    for i, (name, df) in enumerate(df_dict.items()):
        plt.plot(
            df["epoch"], df[metric],
            label=name,
            color=colors[i],
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
