

from groundingdino.util.inference import load_model, load_image, predict, annotate

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
    model = load_model(MODEL_CFG, MODEL_CKT)

    dataset = "home_stains"
    data_dir = Path(f"/home/bzhou/dataset/277e/{dataset}")
    images = sorted(Path(data_dir, 'pngs').glob('*.png'))

    model = model.half()
    # model.compile()
    for img in tqdm(images):
        img_name = img.name
        save_name = Path(data_dir, 'bbox', img_name)
        image_source, image = load_image(img)
        image = image.half()

        tic = time.time()
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        dino_time = time.time() - tic

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        print(f"detection time: {dino_time}")
        cv2.imwrite(str(save_name.resolve()), annotated_frame)

main()


