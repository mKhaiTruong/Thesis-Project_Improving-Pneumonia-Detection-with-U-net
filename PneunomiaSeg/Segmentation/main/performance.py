import os, glob, re, math
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.2)

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
    
    df = pd.DataFrame(rows)
    df["cumulative_time"] = df["time_min"].cumsum()
    return df

# Parse logs
model_results = {
    'deeplabv3+_b3' : r'..\PneunomiaSeg\Segmentation\results\models\unet_deeplabv3plus_efficientnet-b3_exp1\logs',
    'deeplabv3+_mit': r'..\PneunomiaSeg\Segmentation\results\models\unet_deeplabv3plus_mit_b1_exp1\logs',
    'segformer_b3'  : r'..\PneunomiaSeg\Segmentation\results\models\unet_segformer_efficientnet-b3_exp1\logs',
    'segformer_mit' : r'..\PneunomiaSeg\Segmentation\results\models\unet_segformer_mit_b1_exp1\logs',
    'unetpp_b3'     : r'..\PneunomiaSeg\Segmentation\results\models\unet_unetpp_efficientnet-b3_exp1\logs',
}

df_dict = {}
for name, path in model_results.items():
    df = parse_unet_logs(path)
    if df is not None:
        df_dict[name] = df

n_models   = len(df_dict)
n_max_cols = 3
nrows      = math.ceil(n_models / n_max_cols)
fig, axs   = plt.subplots(nrows, n_max_cols, figsize=(4*n_max_cols, 3*nrows))
axs = axs.flatten()

# # --- Visualization ---
for idx, (name, df) in enumerate(df_dict.items()):
    ax = axs[idx]
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o", linewidth=2)
    ax.plot(df["epoch"], df["valid_loss"], label="Valid Loss", marker="s", linewidth=2)
    ax.plot(df["epoch"], df["iou"], label="IOU", marker="^", linewidth=2)

    ax.set_title(f"Model: {name}")
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

for j in range(idx+1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()