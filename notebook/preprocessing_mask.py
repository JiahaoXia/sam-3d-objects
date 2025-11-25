import argparse
import os
import shutil
from pathlib import Path
import cv2
import pandas as pd
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def process_masks(mask_dir: str, output_dir: str):
    """
    Finds the largest mask (by non-zero pixels) for each unique ID 
    and copies it to the output directory.
    """
    # 1. Setup Directories
    mask_path = Path(mask_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not mask_path.is_dir():
        logging.error(f"Mask directory not found: {mask_dir}")
        return

    logging.info(f"Processing masks from: {mask_path}")

    # 2. Collect Data
    result_data = []
    # Use glob for cleaner file discovery (handles .png, .jpg, etc. robustly)
    mask_files = mask_path.glob('*.[jp][pn]g')

    for file_path in mask_files:
        try:
            # Read image as grayscale
            mask = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                logging.warning(f"Skipping: Could not read image at {file_path}")
                continue

            # Extract ID: Assumes ID is before the first '---'
            # Use Pathlib's stem for filename without extension
            file_id = file_path.stem.split('---')[0]
            non_zero_count = cv2.countNonZero(mask)

            result_data.append({
                'id': file_id,
                'path': file_path,
                'count': non_zero_count
            })
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")

    if not result_data:
        logging.info("No valid mask files found.")
        return

    # 3. Group and Find Max (Using Pandas for efficiency)
    df = pd.DataFrame(result_data)
    
    # Group by ID and find the row with the maximum 'count'
    idx = df.groupby(['id'])['count'].idxmax()
    largest_masks = df.loc[idx]

    logging.info(f"Found {len(largest_masks)} unique IDs.")

    # 4. Copy Files
    for _, row in largest_masks.iterrows():
        original_filename = row['path'].name
        
        # Determine new filename: Remove everything after "_mask_0" and add .png
        # The logic below handles cases where "_mask_0" is not present, 
        # but assumes standard naming conventions.
        base_name_parts = original_filename.split("_mask_0")
        new_filename = base_name_parts[0] + ".png"
        
        target_file = output_path / new_filename
        
        logging.info(f"Copying {original_filename} (Count: {row['count']}) -> {target_file.name}")
        shutil.copy(row['path'], target_file)

# --- Argument Parsing (Your original structure) ---

def parse_args():
    parser = argparse.ArgumentParser(description="Find and copy the largest mask for each ID.")
    parser.add_argument('--mask_dir', required=True, help="Directory containing mask images.")
    parser.add_argument('--output_dir', required=True, help="Directory to save the resulting largest masks.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_masks(args.mask_dir, args.output_dir)