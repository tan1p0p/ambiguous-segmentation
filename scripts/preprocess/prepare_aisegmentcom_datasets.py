import glob
import re
from os import path

import cv2
import numpy as np
from tqdm import tqdm


def main():
    alpha_dir = '/home/hassaku/research/ambiguous-segmentation/data/aimattingcom_dataset/alpha/'
    photo_dir = '/home/hassaku/research/ambiguous-segmentation/data/aimattingcom_dataset/photo/'
    trimap_dir = '/home/hassaku/research/ambiguous-segmentation/data/aimattingcom_dataset/trimap/'

    alpha_paths = glob.glob('/home/hassaku/dataset/aisegmentcom-matting-human-datasets/matting_human_half/matting/*/*/*.png')
    photo_paths = glob.glob('/home/hassaku/dataset/aisegmentcom-matting-human-datasets/matting_human_half/clip_img/*/*/*.jpg')

    alpha_basenames = [path.splitext(path.basename(i))[0] for i in alpha_paths]
    photo_basenames = [path.splitext(path.basename(i))[0] for i in photo_paths]
    valid_basenames = list(set(alpha_basenames) & set(photo_basenames))

    for i, basename in tqdm(enumerate(valid_basenames)):
        alpha_path = [i for i in alpha_paths if re.search(basename, i)][0]
        photo_path = [i for i in photo_paths if re.search(basename, i)][0]

        # 透過pngのアルファチャンネルのみ取得し、BGRの3chのグレイスケール画像に変換
        alpha_img = cv2.cvtColor(
            cv2.imread(alpha_path, cv2.IMREAD_UNCHANGED)[:, :, 3], 
            cv2.COLOR_GRAY2BGR)
        photo_img = cv2.imread(photo_path)

        # trimapを作成
        kernel = np.ones((3, 3), np.uint8)
        edge = cv2.Canny(alpha_img, 100, 200)
        translucent = np.where((10 < alpha_img) & (alpha_img < 245), 255, 0)
        comb_edge = np.maximum(cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR), translucent).astype('uint8')

        ambiguous = cv2.dilate(comb_edge, kernel, iterations=20)

        trimap_img = alpha_img.copy()
        trimap_img[ambiguous == 255] = 127

        cv2.imwrite(path.join(alpha_dir, f'{i:05}.png'), alpha_img)
        cv2.imwrite(path.join(photo_dir, f'{i:05}.png'), photo_img)
        cv2.imwrite(path.join(trimap_dir, f'{i:05}.png'), trimap_img)

if __name__ == "__main__":
    main()
