# Ultralytics YOLO 🚀, AGPL-3.0 license
from audioop import error
from collections import deque
import numpy as np
import sys



from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH
import torch
import torch
import torch.nn as nn
from timm import create_model
from timm.data import create_transform
from PIL import Image
import timm
import line_profiler
from line_profiler import LineProfiler
import torch.nn.functional as F
import cv2

class ReIDModel(nn.Module):
    def __init__(self, cfg):
        super(ReIDModel, self).__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            self.cfg.model_type,
            pretrained=False,
            num_classes=0
        )
        self.projector = nn.Linear(
            self.backbone.num_features,
            self.cfg.feature_dim
        )

        # 加载权重
        self.load_weights()

    def load_weights(self):
        """
        加载联合权重
        """
        if not self.cfg.model_path:
            print("未指定权重路径，请检查配置文件！")
            return

        try:
            weights_path = self.cfg.model_path
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state_dict, strict=True)

            print("权重加载成功！")
        except RuntimeError as e:
            print(f"加载权重失败：\n{e}")

    def forward(self, x):
        features = self.backbone(x)
        features = self.projector(features)
        return features


class ReID:
    def __init__(self,cfg):
        self.cfg = cfg
        self.mean = (0.526, 0.576, 0.668)
        self.std = (0.154, 0.140, 0.132)
        self.mean_tensor = torch.tensor(self.mean).view(3, 1, 1).to("cuda")
        self.std_tensor = torch.tensor(self.std).view(3, 1, 1).to("cuda")
        self.model = self.build_model()
        self.preprocess = self.get_preprocess_transform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_preprocess_transform(self):
        """定义图像预处理流程"""
        return create_transform(
    input_size=224,
    mean=(0.526, 0.576, 0.668),
    std=(0.154, 0.140, 0.132),
    is_training=False,  # 使用非训练模式（不应用数据增强）
    crop_pct=1.0,       # 不进行裁剪，保持原始尺寸
    interpolation="bicubic",  # 使用双三次插值调整大小
)

    def build_model(self):
        """构建模型并加载权重"""
        model = ReIDModel(self.cfg)
        model.eval()  # 设置为评估模式
        model.to("cuda")  # 移动模型到 GPU
        return model

    # @profile
    def inference(self, img, dets):
        """
        提取目标区域的特征（GPU 版）
        """
        if img is None:
            raise ValueError("输入图像为空！")

        # 将原始图像转为GPU tensor（提前执行通道转换）
        # img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).div(255).pin_memory()
        # img_tensor = img_tensor.to("cuda", non_blocking=True)
        img_tensor = torch.from_numpy(img).pin_memory().to("cuda", non_blocking=True).float().permute(2, 0, 1).div(255)
        # 批量ROI提取和预处理
        roi_tensors = []
        for det in dets:
            x, y, w, l = det[:4]
            x1 = x - w / 2
            y1 = y - l / 2
            x2 = x + w / 2
            y2 = y + l / 2

            # 确保目标框在图像范围内
            x1 = int(max(0, min(x1, img.shape[1])))
            y1 = int(max(0, min(y1, img.shape[0])))
            x2 = int(max(0, min(x2, img.shape[1])))
            y2 = int(max(0, min(y2, img.shape[0])))

            # GPU直接切片
            roi = img_tensor[:, y1:y2, x1:x2]

            # GPU预处理（双三次插值）
            if roi.size(-1) == 0 or roi.size(-2) == 0:
                raise ValueError("目标框为空")
            roi_resized = F.interpolate(roi.unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False)
            roi_normalized = (roi_resized - self.mean_tensor) / self.std_tensor
            roi_tensors.append(roi_normalized.squeeze(0))

        # 批量堆叠（已经是GPU tensor）
        batch_rois = torch.stack(roi_tensors)

        # 特征提取
        with torch.no_grad():
            features = self.model(batch_rois)  # 批量提取特征
            features = F.normalize(features, p=2, dim=1)  # 批量归一化

        # 转换为 numpy 数组
        features = features.detach().to('cpu', non_blocking=True).numpy()

        return features



class BOTrack(STrack):
    """
    An extended version of the STrack class for YOLOv8, adding object tracking features.

    This class extends the STrack class to include additional functionalities for object tracking, such as feature
    smoothing, Kalman filter prediction, and reactivation of tracks.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of BOTrack.
        smooth_feat (np.ndarray): Smoothed feature vector.
        curr_feat (np.ndarray): Current feature vector.
        features (deque): A deque to store feature vectors with a maximum length defined by `feat_history`.
        alpha (float): Smoothing factor for the exponential moving average of features.
        mean (np.ndarray): The mean state of the Kalman filter.
        covariance (np.ndarray): The covariance matrix of the Kalman filter.

    Methods:
        update_features(feat): Update features vector and smooth it using exponential moving average.
        predict(): Predicts the mean and covariance using Kalman filter.
        re_activate(new_track, frame_id, new_id): Reactivates a track with updated features and optionally new ID.
        update(new_track, frame_id): Update the YOLOv8 instance with new track and frame ID.
        tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        multi_predict(stracks): Predicts the mean and covariance of multiple object tracks using shared Kalman filter.
        convert_coords(tlwh): Converts tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh(tlwh): Convert bounding box to xywh format `(center x, center y, width, height)`.

    Examples:
        Create a BOTrack instance and update its features
        # >>> bo_track = BOTrack(tlwh=[100, 50, 80, 40], score=0.9, cls=1, feat=np.random.rand(128))
        # >>> bo_track.predict()
        # >>> new_track = BOTrack(tlwh=[110, 60, 80, 40], score=0.85, cls=1, feat=np.random.rand(128))
        # >>> bo_track.update(new_track, frame_id=2)
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
        """
        Initialize a BOTrack object with temporal parameters, such as feature history, alpha, and current features.

        Args:
            tlwh (np.ndarray): Bounding box coordinates in tlwh format (top left x, top left y, width, height).
            score (float): Confidence score of the detection.
            cls (int): Class ID of the detected object.
            feat (np.ndarray | None): Feature vector associated with the detection.
            feat_history (int): Maximum length of the feature history deque.

        Examples:
            Initialize a BOTrack object with bounding box, score, class ID, and feature vector
            >>> tlwh = np.array([100, 50, 80, 120])
            >>> score = 0.9
            >>> cls = 1
            >>> feat = np.random.rand(128)
            >>> bo_track = BOTrack(tlwh, score, cls, feat)
        """
        super().__init__(tlwh, score, cls)

        self.smooth_feat = None
        self.curr_feat = None
        if (feat is not None) or (not np.any(np.isnan(feat))):
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        """Update the feature vector and apply exponential moving average smoothing."""
        feat /= (np.linalg.norm(feat)+1e-10)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        """Predicts the object's future state using the Kalman filter to update its mean and covariance."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a track with updated features and optionally assigns a new ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track, frame_id):
        """Updates the YOLOv8 instance with new track information and the current frame ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self):
        """Returns the current bounding box position in `(top left x, top left y, width, height)` format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        """Predicts the mean and covariance for multiple object tracks using a shared Kalman filter."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh):
        """Converts tlwh bounding box coordinates to xywh format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box from tlwh (top-left-width-height) to xywh (center-x-center-y-width-height) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class DMSORT(BYTETracker):
    """
    An extended version of the BYTETracker class for YOLOv8, designed for object tracking with ReID and GMC algorithm.
    """
    def __init__(self, args, frame_rate=30):
        """
        Initialize YOLOv8 object with ReID module and GMC algorithm.
        """
        super().__init__(args, frame_rate)
        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.count=0
        self.encoder = None  # 初始化为 None
        if args.with_reid:
            self.encoder = ReID(args)
        self.gmc = GMC(method=args.gmc_method)

    def get_kalmanfilter(self):
        """Returns an instance of KalmanFilterXYWH for predicting and updating object states in the tracking process."""
        return KalmanFilterXYWH()

    def init_track(self, dets, scores, cls, img=None):
        """Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features."""
        if len(dets) == 0:
            return []
        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder.inference(img, dets)
            return [BOTrack(xyxy, s, c, f) for (xyxy, s, c, f) in zip(dets, scores, cls, features_keep)]  # detections
        else:
            return [BOTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)]  # detections

    def get_dists(self, tracks, detections):
        """Calculates distances between tracks and detections using IoU and optionally ReID embeddings."""
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh*1.5

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists = emb_dists * 800
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[dists_mask] = 1.0

            dists = np.multiply(dists, emb_dists)


        return dists

    def multi_predict(self, tracks):
        """Predicts the mean and covariance of multiple object tracks using a shared Kalman filter."""
        BOTrack.multi_predict(tracks)

    def reset(self):
        super().reset()
        self.gmc.reset_params()
