

from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.inference import BatchedModel

import cv2
import torch
from tqdm import tqdm
from pathlib import Path
import warnings
import time
import torch._dynamo
torch._dynamo.config.suppress_errors = True



warnings.simplefilter('ignore')

def main():
    MODEL_CFG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    MODEL_CKT = "./weights/groundingdino_swint_ogc.pth"
    TEXT_PROMPT = "floor . object . stain ."
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    model = BatchedModel(MODEL_CFG, MODEL_CKT, 'cuda', 'float16', False)
    dataset = "home_stains"
    data_dir = Path(f"/home/bzhou/dataset/277e/{dataset}")
    images = sorted(Path(data_dir, 'pngs').glob('*.png'))

    for img in tqdm(images):
        img_name = img.name
        save_name = Path(data_dir, 'bbox', img_name)
        image_source, image = load_image(img)
        image = image.to('cuda').half()[None, ...]
        
        tic = time.time()
        with torch.no_grad():
            bbox_batch, conf_batch, class_id_batch  = model(
                image_batch=image,
                text_prompts=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                nms_threshold=NMS_THRESHOLD
            )

        dino_time = time.time() - tic
        print(f"detection time: {dino_time}")



main()


