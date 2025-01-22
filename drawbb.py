import os
import torch
import torchvision
import nibabel as nib
import numpy as np
import cv2
from torchvision.ops import masks_to_boxes

# Function to process NIfTI slices and draw solid red bounding boxes
def process_nifti_with_bboxes(scan_folder, model, device, output_folder):
    scan_files = sorted([f for f in os.listdir(scan_folder) if f.endswith('.nii.gz') or f.endswith('.nii')])

    for scan_file in scan_files:
        scan_path = os.path.join(scan_folder, scan_file)
        print(f"Processing scan: {scan_path}")
        scan_img = nib.load(scan_path)
        scan_data = scan_img.get_fdata()

        # Normalize the scan data to [0, 1] for visualization purposes
        scan_data = (scan_data - np.min(scan_data)) / (np.max(scan_data) - np.min(scan_data))

        # Initialize a copy for bounding boxes
        output_data = scan_data.copy()

        for i in range(scan_data.shape[2]):  # Iterate over slices
            slice_img = scan_data[:, :, i]
            slice_tensor = torch.tensor(slice_img, dtype=torch.float32).unsqueeze(0).to(device)

            # Convert the slice to 3 channels by repeating the single channel 3 times (for Mask-RCNN)
            slice_tensor_rgb = slice_tensor.repeat(3, 1, 1)  # Convert to [3, H, W]

            # Apply the model and get the prediction
            model.eval()
            with torch.no_grad():
                prediction = model([slice_tensor_rgb])  # Pass the 3D tensor directly without extra dimension

            pred = prediction[0]
            scores = pred["scores"].cpu().numpy()
            print(f"Slice {i}: Number of bounding boxes: {len(scores)}",'scores:',scores)

            if len(scores) > 0:
                # Filter out bounding boxes with scores less than 0.7
                high_score_indices = np.where(scores > 0.15)[0]

                # Add bounding boxes only for those with a score greater than 0.7
                boxes = pred['boxes'].cpu().numpy()[high_score_indices] if len(high_score_indices) > 0 else []
                labels = pred['labels'].cpu().numpy()[high_score_indices] if len(high_score_indices) > 0 else []

                for box, label in zip(boxes, labels):
                    x_min, y_min, x_max, y_max = map(int, box)

                    # Draw a solid red bounding box on the slice
                    red_value = 1  # Maximum red intensity
                    output_data[y_min:y_min+2, x_min:x_max, i] = red_value  # Top boundary
                    output_data[y_max-2:y_max, x_min:x_max, i] = red_value  # Bottom boundary
                    output_data[y_min:y_max, x_min:x_min+2, i] = red_value  # Left boundary
                    output_data[y_min:y_max, x_max-2:x_max, i] = red_value  # Right boundary


                     # Convert the slice to a color image for label drawing
                    slice_color_img = cv2.cvtColor((slice_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

                    # Add the label text near the top-left corner of the bounding box
                    text_position = (x_min, y_min - 10 if y_min - 10 > 0 else y_min + 10)  # Adjust position to avoid being out of bounds
                    label_text = f'Class: {label}'
                    cv2.putText(slice_color_img, label_text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)


                    # Optionally: print label and coordinates
                    print(f"Slice {i}: Label: {label}, Box: [{x_min}, {y_min}, {x_max}, {y_max}], Score: {scores[high_score_indices]}")

        # Save the NIfTI slice with bounding boxes applied
        output_nii = nib.Nifti1Image(output_data, affine=scan_img.affine)
        output_filename = scan_file.replace('.nii.gz', '_bboxes.nii.gz')
        output_path = os.path.join(output_folder, output_filename)
        nib.save(output_nii, output_path)
        print(f"Saved NIfTI with bounding boxes: {output_path}")

# Function to load the model and apply bounding boxes
def load_model_and_apply_bboxes(scan_folder, model_weights, output_folder, num_classes, device):
    # Load Mask-RCNN model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

    # Load the saved weights with a focus on matching only the common layers
    state_dict = torch.load(model_weights, map_location=device)

    # Ignore the mask predictor layer mismatches and reload the rest of the model
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Reinitialize the mask predictor layers with the correct shape
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 1024
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    model.to(device)

    # Process the scans and apply bounding boxes
    process_nifti_with_bboxes(scan_folder, model, device, output_folder)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply Solid Red Bounding Boxes to NIfTI scans.")
    parser.add_argument('--scans_folder', type=str, required=True, help='Folder containing the scan .nii.gz files.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the output NIfTI files.')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes for the model.')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model and apply solid red bounding boxes to the input NIfTI scans
    load_model_and_apply_bboxes(
        args.scans_folder,
        args.weights,
        args.output_folder,
        args.num_classes,
        device
    )

