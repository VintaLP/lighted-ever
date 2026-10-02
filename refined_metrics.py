from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    
    renders_dir = Path(renders_dir)
    gt_dir = Path(gt_dir)
    
    # Recursively search for all .png files (handles cam1/ and cam2/)
    for render_file in renders_dir.rglob('*.png'):
        # Get the relative path, e.g., "cam1/00000.png"
        rel_path = render_file.relative_to(renders_dir)
        gt_file = gt_dir / rel_path
        
        if not gt_file.exists():
            print(f"[WARNING] Missing GT image for {rel_path} in {gt_dir}")
            continue
            
        render = Image.open(render_file)
        gt = Image.open(gt_file)
        
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        
        # Use .as_posix() to ensure clean forward slashes in JSON
        image_names.append(rel_path.as_posix())
        
    return renders, gts, image_names

def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            if not test_dir.exists():
                print(f"[WARNING] No 'test' folder found in {scene_dir}.")
                continue

            for method in os.listdir(test_dir): # e.g., "ours_30000"
                print("\nMethod:", method)
                method_dir = test_dir / method
                
                render_dirs_dict = {}
                for item in os.listdir(method_dir):
                    # Ignore GT folders and non-directory items
                    if item.startswith("gt") or not os.path.isdir(method_dir / item):
                        continue
                    
                    item_dir = method_dir / item
                    
                    # Match the correct GT folder based on the mode name
                    gt_folder_name = "gt_lighted" # Fallback
                    if "unlit" in item:
                        gt_folder_name = "gt_unlit"
                    elif "normal" in item:
                        gt_folder_name = "gt_normals"
                    elif "brightness" in item:
                        gt_folder_name = "gt_brightness"
                        
                    gt_dir = method_dir / gt_folder_name
                    
                    # Look for the timestamp subfolder created by render.py
                    sub_folders = sorted([f for f in os.listdir(item_dir) if os.path.isdir(item_dir / f)])
                    if sub_folders:
                        latest_sub = sub_folders[-1] 
                        render_dirs_dict[f"{method}_{item}"] = (item_dir / latest_sub, gt_dir)
                    else:
                        render_dirs_dict[f"{method}_{item}"] = (item_dir, gt_dir)

                for sub_method, (renders_dir, gt_dir) in render_dirs_dict.items():
                    if not gt_dir.exists():
                        print(f"Skipping {sub_method}: GT folder '{gt_dir.name}' does not exist.")
                        continue
                        
                    print(f"--- Evaluating: {sub_method} (against {gt_dir.name}) ---")

                    full_dict[scene_dir][sub_method] = {}
                    per_view_dict[scene_dir][sub_method] = {}
                    full_dict_polytopeonly[scene_dir][sub_method] = {}
                    per_view_dict_polytopeonly[scene_dir][sub_method] = {}

                    renders, gts, image_names = readImages(renders_dir, gt_dir)
                    
                    if len(renders) == 0:
                        print(f"[WARNING] No images found for {sub_method}.")
                        continue

                    ssims = []
                    psnrs = []
                    lpipss = []
                    lpipsa = []
                    ms_ssims = []
                    Dssims = []
                    
                    for idx in tqdm(range(len(renders)), desc=f"Metrics ({sub_method})"):
                        ssims.append(ssim(renders[idx], gts[idx]))
                        psnrs.append(psnr(renders[idx], gts[idx]))
                        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                        ms_ssims.append(ms_ssim(renders[idx], gts[idx], data_range=1, size_average=True))
                        lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
                        Dssims.append((1-ms_ssims[-1])/2)

                    print("Scene: ", scene_dir,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    print("Scene: ", scene_dir,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    print("Scene: ", scene_dir,  "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                    print("Scene: ", scene_dir,  "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
                    print("Scene: ", scene_dir,  "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
                    print("Scene: ", scene_dir,  "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

                    full_dict[scene_dir][sub_method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                                            "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                                                            "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                                                            "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                                                            "D-SSIM": torch.tensor(Dssims).mean().item()},
                                                        )
                    per_view_dict[scene_dir][sub_method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                                "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                                "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                                                                "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                                                                "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)},
                                                                }
                                                            )

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
                
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            raise e

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)