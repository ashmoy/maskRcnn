import os
import shutil
import argparse

def extract_nii_files(input_dir, output_dir):
    """
    Traverse through the input directory, find all `1_NIFTI` folders,
    and copy the `.nii.gz` files to the output directory.

    Parameters:
    - input_dir: str, path to the input directory.
    - output_dir: str, path to the output directory.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Traverse the input directory
    for root, dirs, files in os.walk(input_dir):
        if os.path.basename(root) == "1_NIFTI":
            for file in files:
                if file.endswith(".nii.gz"):
                    source_file = os.path.join(root, file)
                    destination_file = os.path.join(output_dir, file)
                    print(f"Copying {source_file} to {destination_file}")
                    shutil.copy2(source_file, destination_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and copy .nii.gz files from 1_NIFTI folders.")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")

    args = parser.parse_args()

    extract_nii_files(args.input_dir, args.output_dir)
