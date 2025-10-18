import torch
import json
import os, logging
import shutil
import numpy as np
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

HYPERPARAM_KEYS = [
    # training
    "lr", "batch_size", "decay", "patience", "epochs", "seed", "print_freq",
    # model
    "name", "encoder", "encoder_weights", "activation", "classes",
    # loss
    "loss_type", "bce_weight", "tversky_weight", "pos_weight",
    "alpha", "beta", "smooth",
    # misc
    "task", "device"
]


def get_logger(filename, verbosity=1, name=None):
    level_dict = {
        0: logging.DEBUG,
        1: logging.INFO,
        2: logging.WARNING,
    }
    
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

def save_hparams_to_json(args, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    cfg_path = os.path.join(base_dir, f"{args.task}_config.json")
    payload  = {}
    
    for k in HYPERPARAM_KEYS:
        v = getattr(args, k, None)
        payload[k] = str(v) if hasattr(v, "__class__") and v.__class__.__name__ == "device" else v
    
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_metrics_jsonl(task, epoch, train_loss, val_loss, iou, best_iou, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    path   = os.path.join(base_dir, f"{task}_metrics.jsonl")
    record = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "mean_iou": float(iou),
        "best_iou": float(best_iou),
    }
    
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def data_to_device(data):
    img, dens_map = data
    return img.to(DEVICE), dens_map.to(DEVICE)

def add_in_file(model_name, BASE_DIR, text):
    log_path = os.path.join(BASE_DIR, f"logs_{model_name}.txt")
    
    with open(log_path, "a+") as f:
        f.write(text)

def save_checkpoint(state, is_better, task, out_dir, file_name='checkpoint.pth.tar'):
    ckpt_name = f"{task}_{file_name}"
    ckpt_path = os.path.join(out_dir, ckpt_name)
    
    torch.save(state, ckpt_path)
    
    if is_better:
        best_name = f"{task}_model_best.pth.tar"
        best_path = os.path.join(out_dir, best_name)
        shutil.copyfile(ckpt_path, best_path)
        

        
if __name__ == "__main__":
    # Test log
    test_model = "quicktest_model"
    add_in_file(test_model, "This is a quick test line.\n")

    # Test save checkpoint
    dummy_state = {"epoch": 0}
    save_checkpoint(dummy_state, is_best=True, task_id="quicktest_")