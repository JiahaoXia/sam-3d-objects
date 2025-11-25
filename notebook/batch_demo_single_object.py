import os
import sys
import json
from pathlib import Path
import argparse
from inference import Inference, load_image
import time
import imageio



def load_last_done(PROGRESS_FILE) -> int:
    """Return last successfully processed index, or -1 if none."""
    if not os.path.exists(PROGRESS_FILE):
        return -1
    with open(PROGRESS_FILE, "r") as f:
        data = json.load(f)
    return int(data.get("last_index", -1))


def save_last_done(idx: int, PROGRESS_FILE) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_index": int(idx)}, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch process images to generate 3D model.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/original",
        help="Directory containing input images."
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/masks",
        help="Directory containing input masks."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/single_object",
        help="Directory to save output LAZ files."
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="/home/user/XJH/model_zoos/sam-3d-objects/hf-download/checkpoints/pipeline.yaml",
        help="Path to the configuration file."  
    )

    # model
    parser.add_argument(
        "--bake_texture_mode",
        type=str,
        default="fast",
        help="Texture baking mode."
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.995,
        help="Simplification ratio for the mesh."
    )
    parser.add_argument(
        "--texture_size",
        type=int,
        default=512,
        help="Texture size for the output model."
    )

    # parallel processing
    parser.add_argument(
        "--total_processes",
        type=int,
        default=4,
        help="Total number of parallel processes."
    )
    parser.add_argument(
        "--process_index",
        type=int,
        default=0,
        help="Index of this process (0 to total_processes-1)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_dir = args.image_dir
    mask_dir = args.mask_dir
    output_dir = args.output_dir
    config_path = args.cfg_path

    simplify = args.simplify
    texture_size = args.texture_size
    bake_texture_mode = args.bake_texture_mode

    # progress file
    total_processes = args.total_processes
    process_index = args.process_index

    output_dir = f"{output_dir}_bake-{bake_texture_mode}_simp-{simplify}_texsize-{texture_size}"
    os.makedirs(output_dir, exist_ok=True)
    PROGRESS_FILE = os.path.join(output_dir, f"progress_process_{process_index}_of_{total_processes}.json")

    inference = Inference(config_path, compile=False)

    mask_filenames = sorted(os.listdir(mask_dir))
    # distribute work among multiple processes
    mask_filenames = [f for i, f in enumerate(mask_filenames) if i % total_processes == process_index]

    t0 = time.time()
    last_done = load_last_done(PROGRESS_FILE)
    n = len(mask_filenames)

    print(f"Found {n} masks, last_done index = {last_done}")

    # If everything is already done, just exit normally
    if last_done >= n - 1:
        print("All images already processed.")
        return
    
    for idx, mask_filename in enumerate(mask_filenames):
        if idx <= last_done:
            # Skip already processed items
            continue

        try:
            MASK_PATH = os.path.join(mask_dir, mask_filename)
            IMAGE_NAME = os.path.splitext(mask_filename)[0]
            IMAGE_PATH = os.path.join(image_dir, IMAGE_NAME + ".jpg")

            if not os.path.exists(MASK_PATH):
                print(f"[WARNING] Mask not found for {IMAGE_NAME}, skipping.")
                save_last_done(idx, PROGRESS_FILE)
                continue

            image = load_image(IMAGE_PATH)
            mask = imageio.imread(MASK_PATH)
            mask = (mask > 0).astype("uint8")  # binary mask

            # run model
            output = inference(image, mask, seed=42, simplify=simplify, texture_size=texture_size, bake_texture_mode=bake_texture_mode)

            mesh = output["glb"] # trimesh object
            mesh.export(f"{output_dir}/{IMAGE_NAME}.glb", file_type='glb')  # save mesh w/ vertex colors
        except RuntimeError as e:
            msg = str(e)
            if "Cuda error: 710" in msg or "cudaErrorIllegalAddress" in msg:
                print(f"[ERROR] CUDA 710 hit at index {idx}, restarting script...")

                # At this point, last_done still points to the last successfully
                # completed item (idx-1), because we only update it after success.
                # So we don't need to touch progress.json here.

                # Re-exec the current script in the same process (hard restart).
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                # Some other error: re-raise so you notice it
                raise

        # If we got here, this item finished successfully
        save_last_done(idx, PROGRESS_FILE)

    print(f"All images processed successfully in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()