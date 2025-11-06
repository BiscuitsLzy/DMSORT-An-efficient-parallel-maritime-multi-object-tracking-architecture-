import warnings, os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    model = YOLO(r'basepath\DMSORT\ultralytics\cfg\models\11\RCDN.yaml')
    model.train(data='data.yml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=4,
                patience=0,
                project='runs/train',
                name='exp',
                )
