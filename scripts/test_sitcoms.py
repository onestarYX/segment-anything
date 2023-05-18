from pathlib import Path
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
from tqdm import tqdm
import torchvision
import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True)

    sam = sam_model_registry["vit_h"](checkpoint="ckpt/sam_vit_h_4b8939.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)

    for img_file in tqdm(input_folder.iterdir()):
        input_img = torchvision.io.read_image(str(img_file))
        input_img = input_img.permute(1, 2, 0)
        input_img = input_img.numpy()
        masks = mask_generator.generate(input_img)
        mask_img = np.zeros(input_img.shape)
        for mask in masks:
            rand_color = np.random.rand(3)
            seg = mask['segmentation']
            mask_img[seg] = rand_color
        mask_overlay = (input_img.astype('float32') / 255.0) * 0.5 + mask_img * 0.5
        mask_overlay = (mask_overlay * 255).astype('uint8')
        cv2.imwrite(str(output_folder / img_file.name), mask_overlay)