import os
import cv2
import torch

from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim

def read_image(img_path):
    # cv读图像并转换成torch
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float() / 255.0
    return image


# 遍历图像，评估每张图像rl比fastgs好多少，并找到最好那张

scenes = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden", "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]

# scene = "stump"
for scene in scenes:
    rlgs_path = "output/rl/" + scene + "/test/ours_30000"
    fastgs_path = "output/fastgs/" + scene + "_big" + "/test/ours_30000"
    # fastgs_path = "../Perceptual-GS/eval/" + scene + "/test/ours_30000"

    print("current scene: ", scene)

    pre_best_diff = 0.
    for img_name in os.listdir(os.path.join(rlgs_path, "renders")):
        gt_img_path = os.path.join(rlgs_path, "gt", img_name)
        rlgs_img_path = os.path.join(rlgs_path, "renders", img_name)
        fastgs_img_path = os.path.join(fastgs_path, "renders", img_name)

        gt_image = read_image(gt_img_path)
        image_rl = read_image(rlgs_img_path)
        image_fastgs = read_image(fastgs_img_path)

        Ll1 = l1_loss(image_rl, gt_image)
        ssim_value = fast_ssim(image_rl.unsqueeze(0), gt_image.unsqueeze(0))
        loss_rl = 0.8 * Ll1 + 0.2 * (1.0 - ssim_value)

        Ll1 = l1_loss(image_fastgs, gt_image)
        ssim_value = fast_ssim(image_fastgs.unsqueeze(0), gt_image.unsqueeze(0))
        loss_fastgs = 0.8 * Ll1 + 0.2 * (1.0 - ssim_value)

        diff = loss_fastgs - loss_rl
        # print(f"{img_name}: {diff}")

        if diff > pre_best_diff:
            pre_best_diff = diff
            best_img_name = img_name

    print("best image:", best_img_name, pre_best_diff.item())
