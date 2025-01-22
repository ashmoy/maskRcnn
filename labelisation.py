import os
import argparse
import numpy as np
from skimage.io import imread, imsave
from skimage.measure import label, regionprops

def assign_labels(seg_array, labeled_array, regions):
    """
    Assign labels based on tooth position:
    - Label 0: Background
    - Label 128: Tooth on the bottom/right
    - Label 255: Tooth on the top/left
    """
    # Get image height for reference
    image_height = seg_array.shape[0]

    if len(regions) == 2:
        # Two components, assign based on their vertical position (centroid)
        if regions[0].centroid[0] > regions[1].centroid[0]:
            # The first component is lower (label it 128), second is higher (label it 255)
            labeled_array[labeled_array == 1] = 128  # Right/bottom tooth gets label 128
            labeled_array[labeled_array == 2] = 255  # Left/top tooth gets label 255
        else:
            # The first component is higher (label it 255), second is lower (label it 128)
            labeled_array[labeled_array == 1] = 255  # Left/top tooth gets label 255
            labeled_array[labeled_array == 2] = 128  # Right/bottom tooth gets label 128

    elif len(regions) == 1:
        # Only one component, decide the label based on the vertical center of mass
        center_of_mass = regions[0].centroid[0]
        if center_of_mass > image_height / 2:
            # Majority of the component is on the bottom part (assign label 128)
            labeled_array[labeled_array == 1] = 128
        else:
            # Majority of the component is on the top part (assign label 255)
            labeled_array[labeled_array == 1] = 255

    return labeled_array

def process_image(input_image_path, output_image_path):
    # Read the PNG image as a binary mask (assuming white for the teeth, black for the background)
    seg_array = imread(input_image_path, as_gray=True)  # Read in grayscale
    seg_array = np.where(seg_array > 0, 1, 0)  # Binarize the image

    # Label connected components in the segmentation mask
    labeled_array, num_features = label(seg_array, return_num=True, connectivity=2)
    
    print(f"Processing {input_image_path}: Found {num_features} components.")
    
    if num_features > 0:
        # Get properties of the labeled regions (connected components)
        regions = regionprops(labeled_array)

        # Assign labels to the components: background as 0, teeth as 128 or 255
        labeled_array = assign_labels(seg_array, labeled_array, regions)

    else:
        print(f"No components found in {input_image_path}, skipping.")
        return

    # Ensure background is labeled as 0
    labeled_array[seg_array == 0] = 0

    # Convert the labeled array to 8-bit format for saving as PNG
    labeled_array = labeled_array.astype(np.uint8)

    # Save the updated image with distinct labels
    imsave(output_image_path, labeled_array)
    print(f"Saved updated image to {output_image_path}")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through all PNG files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            
            # Process each image
            process_image(input_image_path, output_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process segmentation images (PNG) and assign distinct labels to connected components.")
    parser.add_argument("--input_folder", required=True, help="Path to the folder containing the input segmentation images (PNG).")
    parser.add_argument("--output_folder", required=True, help="Path to the folder to save the processed images.")
    
    args = parser.parse_args()
    
    process_folder(args.input_folder, args.output_folder)
