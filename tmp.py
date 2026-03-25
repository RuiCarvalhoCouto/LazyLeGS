import os
import json


scene_names = {
    "mipnerf360": ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"],
    "tanks_and_temples": ["truck", "train"],
    "deep_blending": ["drjohnson", "playroom"]
}
output_path = "output/rl2"

for dataset in scene_names.keys():
    print(dataset)
    SSIM, PSNR, LPIP, NGS = 0., 0., 0., 0.
    for scene in scene_names[dataset]:
        result_path = os.path.join(output_path, scene, "results.json")
        time_and_count_path = os.path.join(output_path, scene, "time_and_count.txt")

        if not os.path.exists(result_path):
            continue

        with open(result_path, "r") as f:
            results = json.load(f)["ours_30000"]
            SSIM += results["SSIM"]
            PSNR += results["PSNR"]
            LPIP += results["LPIPS"]

        with open(time_and_count_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Gaussian number"):
                    NGS += float(line.split(":")[1].strip())
                if line.startswith("Training time"):
                    training_time = float(line.split(":")[1].strip().split(" ")[0])

    SSIM /= len(scene_names[dataset])
    PSNR /= len(scene_names[dataset])
    LPIP /= len(scene_names[dataset])
    NGS /= len(scene_names[dataset])
    NGS /= 1e6
    training_time /= 60.

    print(f"SSIM: {SSIM:.4f}, PSNR: {PSNR:.4f}, LPIP: {LPIP:.4f}, NGS: {NGS:.4f}, Training time: {training_time:.4f}")
