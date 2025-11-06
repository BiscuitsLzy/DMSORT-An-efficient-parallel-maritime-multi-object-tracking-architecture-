
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
import random
import timm.data


class CustomDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root = os.path.join(root_dir, mode)
        self.transform = transform
        self.samples = []
        self.imWidth, self.imHeight = 1920, 1080

        self.current_max_id = 0  # 记录当前最大ID
        self._process_videos()
        self._build_id_mappings()

    def _process_videos(self):
        video_dirs = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        for video_dir in video_dirs:
            video_path = os.path.join(self.root, video_dir)
            img_dir = os.path.join(video_path, 'img1')
            gt_dir = os.path.join(video_path, 'gt')

            max_orig_id = 0
            gt_file = os.path.join(gt_dir, [f for f in os.listdir(gt_dir) if f.endswith('.txt')][0])

            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    orig_id = int(parts[1])
                    if orig_id > max_orig_id:
                        max_orig_id = orig_id

            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue

                    frame_idx = int(parts[0])
                    orig_id = int(parts[1])
                    x, y, w, h = map(int, parts[2:6])

                    global_id = self.current_max_id + orig_id

                    img_name = f"{frame_idx:06d}.jpg"
                    img_path = os.path.join(img_dir, img_name)

                    if os.path.exists(img_path) and w > 0 and h > 0:
                        self.samples.append((img_path, (x, y, w, h, global_id)))

            self.current_max_id += max_orig_id
    def _build_id_mappings(self):
        self.id_to_indices = defaultdict(list)
        self.ids = []

        for idx, (_, (_, _, _, _, obj_id)) in enumerate(self.samples):
            self.id_to_indices[obj_id].append(idx)

        self.ids = sorted(self.id_to_indices.keys())
        print(f"Loaded {len(self)} samples with {len(self.ids)} unique IDs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor = self._get_single_item(index)
        _, _, _, _, obj_id = self.samples[index][1]
        pos_idx = random.choice(self.id_to_indices[obj_id])
        positive = self._get_single_item(pos_idx)

        neg_id = random.choice([x for x in self.ids if x != obj_id])
        neg_idx = random.choice(self.id_to_indices[neg_id])
        negative = self._get_single_item(neg_idx)

        return anchor, positive, negative

    def _get_single_item(self, idx):
        """处理单个样本的加载"""
        img_path, (x, y, w, h, obj_id) = self.samples[idx]

        img = Image.open(img_path).convert('RGB')

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(x + w, self.imWidth)
        y2 = min(y + h, self.imHeight)

        if x2 <= x1 or y2 <= y1:

            crop = img
        else:
            crop = img.crop((x1, y1, x2, y2))

        # 数据增强
        if self.transform:
            crop = self.transform(crop)

        return crop


class PKSampler(Sampler):

    def __init__(self, dataset, P=8, K=16):
        self.dataset = dataset
        self.P = P
        self.K = K
        self.valid_ids = [oid for oid in self.dataset.ids
                          if len(self.dataset.id_to_indices[oid]) >= K]
        print(f"Valid IDs for PK sampling: {len(self.valid_ids)}")

    def __iter__(self):
        np.random.shuffle(self.valid_ids)

        for i in range(0, len(self.valid_ids), self.P):
            batch_ids = self.valid_ids[i:i + self.P]
            if len(batch_ids) < self.P:
                continue

            batch = []
            for oid in batch_ids:
                indices = self.dataset.id_to_indices[oid]
                selected = np.random.choice(indices, self.K, replace=False)
                batch.extend(selected.tolist())

            yield batch

    def __len__(self):
        return len(self.valid_ids) // self.P


def collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives)
    )

def build_transform(is_train=True):
    return timm.data.create_transform(
        input_size=224,
        is_training=is_train,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5',
        interpolation='bicubic',
        re_prob=0.25,
        mean=(0.526, 0.576, 0.668),
        std=(0.154, 0.140, 0.132)
    )

