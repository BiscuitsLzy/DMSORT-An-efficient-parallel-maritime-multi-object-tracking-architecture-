import os
import yaml
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import torch.nn.functional as F
from torch.amp import GradScaler,autocast
from get_feature import build_model
from get_data import build_transform, CustomDataset, PKSampler, collate_fn
from torch.amp import GradScaler, autocast  
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
import torch.nn as nn
import torch.nn.functional as F

import time  
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_cos_sim = F.cosine_similarity(anchor, positive)
        neg_cos_sim = F.cosine_similarity(anchor, negative)
        pos_dist = 1 - pos_cos_sim
        neg_dist = 1 - neg_cos_sim

        losses = F.relu(pos_dist - neg_dist + self.margin*2)
        losses_squar=losses**2
        return losses_squar.mean()


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_experiment_dir(base_dir="experiments"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("exp")]
    max_num = max([int(d[3:]) for d in existing if d[3:].isdigit()], default=0)
    new_dir = os.path.join(base_dir, f"exp{max_num + 1}")
    os.makedirs(new_dir, exist_ok=False)
    return new_dir


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for anchors, positives, negatives in val_loader:  
            anchors = anchors.cuda()
            positives = positives.cuda()
            negatives = negatives.cuda()

            with autocast("cuda"):
                anchor_feat = model(anchors)
                positive_feat = model(positives)
                negative_feat = model(negatives)
                loss = criterion(anchor_feat, positive_feat, negative_feat)

            val_loss += loss.item() * anchors.size(0)
    return val_loss / len(val_loader.dataset)


def main():
    cfg = load_config('basepath\\DMSORT\\feature_asso\\configs\\litae.yaml')
    exp_dir = create_experiment_dir()
    print(f"Experiment directory created at: {exp_dir}")


    with open(os.path.join(exp_dir, "config.yaml"), 'w') as f:
        yaml.dump(cfg, f)

    train_set = CustomDataset(
        root_dir=cfg['data']['root_dir'],
        mode='train',
        transform=build_transform(is_train=True)
    )
    val_set = CustomDataset(
        root_dir=cfg['data']['root_dir'],
        mode='test',
        transform=build_transform(is_train=False)
    )

    train_sampler = PKSampler(train_set, P=cfg['sampler']['P'], K=cfg['sampler']['K'])
    val_sampler = PKSampler(val_set, P=cfg['sampler']['P'], K=cfg['sampler']['K'])

    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,  
        num_workers=cfg['train']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True
    )

    # 模型初始化
    model = build_model(cfg).cuda()
    criterion = BatchHardTripletLoss(margin=float(cfg['train']['margin']))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg['train']['lr']),
        weight_decay=float(cfg['train']['weight_decay'])
    )

    # 学习率调度
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg['train']['epochs']) - int(cfg['train']['warmup_epochs']),
        eta_min=float(cfg['train']['lr']) * 0.01
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=int(cfg['train']['warmup_epochs']),
        after_scheduler=cosine_scheduler
    )

    scaler = GradScaler("cuda")
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(int(cfg['train']['epochs'])):
        model.train()
        train_loss = 0.0

        # 修改训练循环解包方式
        for anchors, positives, negatives in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            anchors = anchors.cuda()
            positives = positives.cuda()
            negatives = negatives.cuda()

            optimizer.zero_grad()

            with autocast("cuda"):
                anchor_feat = model(anchors)
                positive_feat = model(positives)
                negative_feat = model(negatives)
                loss = criterion(anchor_feat, positive_feat, negative_feat)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * anchors.size(0)

        # 验证
        val_loss = validate(model, val_loader, criterion)
        scheduler.step()

        # 记录历史
        avg_train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

        # 打印日志
        print(f"\nEpoch {epoch + 1}/{cfg['train']['epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"LR: {history['lr'][-1]:.2e}")

    # 保存最终结果
    torch.save(model.state_dict(), os.path.join(exp_dir, "final_model.pth"))
    torch.save(history, os.path.join(exp_dir, "history.pt"))

    # 绘制曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title("Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(exp_dir, "loss_curve.png"))
    plt.close()


if __name__ == "__main__":
    main()
