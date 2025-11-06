# Maritime Multi-Object Tracking System

This project is developed based on the BoT-SORT algorithm, combined with the designed RCDN detection method to achieve efficient multi-object tracking on the Singapore Maritime Dataset. The system supports independent training of detection models and ReID models, and introduces COFF strategy to optimize multi-modal feature fusion.
## Maritime Multi-Object Tracking System

First prepare the model weights, place the trained detection model weights (such as best.pt) in the project weights directory.

Then configure the paths, modify the path parameters in `get_dmsort_mot.py`:

# Detection model configuration (modify according to actual path)
model_cfg = "./ultralytics/cfg/models/11/RCDN.yaml"
pretrained_weights = "./weights/best.pt"

# Data input and output paths
base_path = "./data/test" # Test data directory
output_root = "./runs/mot/results" # Tracking result save directory

Finally run the tracking system, execute the command: `get_dmsort_mot.py`

## Model Training

### Detection Model Training

First prepare the dataset, download the Singapore Maritime Dataset and convert it to YOLO format, then modify the dataset path in data.yml.
Then start training, execute the command: `python train.py`

### ReID Model Training

First configure the dataset path, and edit the feature_asso/configs/litae.yaml file:
root_dir: "./reid_dataset" # ReID format dataset path
Then start training, enter the feature_asso directory and execute the command: `python train.py`

## Evaluation
The primary evaluation metric for multi-object tracking performance is the **Weighted Average** of the per-class scores. This approach accounts for class imbalance in the dataset, ensuring that the overall performance score is representative of the model's effectiveness across all maritime object categories.

## COFF Strategy Explanation

During the model fusion stage, the COFF strategy balances the contribution of detection and ReID models by dynamically adjusting the Scaling Factor. The core principles are as follows:
Parameter selection basis: The Scaling Factor should be determined by the distance distribution of positive/negative samples output by the ReID model. For example, if the positive sample distances from the ReID model are concentrated around 0.01 and the negative sample distances are concentrated around 0.1, then the Scaling Factor should be set to around 10 (to align their orders of magnitude).
Avoid fixed values: Fixed Scaling Factor may cause one modal feature to dominate, weakening the contribution of the other modal. It is recommended to adjust this parameter through the validation set.
