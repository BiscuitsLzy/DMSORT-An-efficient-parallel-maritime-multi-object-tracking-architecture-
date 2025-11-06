import os
from time import process_time_ns

#from pywin.framework.interact import valueFormatOutputError

from ultralytics import YOLO
import cv2
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def process_video(video_path, output_txt_path,model):

    model_cfg = r"basepath\DMSORT\ultralytics\cfg\models\11\RCDN.yaml"
    pretrained_weights = r"basepath\DMSORT\runs\DMSORT\best.pt"
    model = YOLO(model_cfg).load(pretrained_weights)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    # 创建输出文件
    with open(output_txt_path, 'w') as f:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model.track(
                frame,
                persist=True,
                tracker="dmsort.yaml",
                conf=0.1,
                iou=0.5,
                verbose=False
            )


            if results[0].boxes.id is not None:
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()

                for idx, (box, track_id) in enumerate(zip(results[0].boxes.xywh.cpu(), results[0].boxes.id.cpu())):
                    x, y, w, h = box.numpy()
                    conf = confidences[idx].item()
                    class_id = int(class_ids[idx])

                    line = f"{frame_idx+1},{int(track_id)},{x - w / 2:.2f},{y - h / 2:.2f},{w:.2f},{h:.2f},{conf:.2f},{class_id},1\n"
                    f.write(line)
            frame_idx += 1

        cap.release()


def generate_mot_results():
    base_path = r"Videos_basepath"
    output_root = r"your_output_path"

    os.makedirs(output_root, exist_ok=True)

    model_cfg = r"basepath\DMSORT\ultralytics\cfg\models\11\RCDN.yaml"
    pretrained_weights = r"basepath\DMSORT\runs\DMSORT\best.pt"
    model = YOLO(model_cfg).load(pretrained_weights)


    for dir_name in os.listdir(base_path):
        dir_path = os.path.join(base_path, dir_name)

        if os.path.isdir(dir_path):
            video_path = os.path.join(r"basepath\Videos", f"{dir_name}.avi")

            output_txt = os.path.join(output_root, f"{dir_name}.txt")

            if os.path.exists(video_path):
                print(f"Processing {dir_name}...")
                process_video(video_path, output_txt,model)
            else:
                print(f"Video not found: {video_path}")


if __name__ == "__main__":
    generate_mot_results()

