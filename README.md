
# CLI-C Module

## Overview

**CLI-C** (Classification and Localization of Impacted Canines) is a 3D Slicer extension designed for automated segmentation of dental Cone-Beam Computed Tomography (CBCT) images using a deep learning Mask R-CNN model. It streamlines dental segmentation workflows for intra-osseous teeth, facilitating accurate identification and visualization of impacted canines for orthodontic and dental surgical planning.

---

## Key Features

- **Automatic Segmentation**: Precisely segments and classify impacted teeth, determininng the intra-osseous localization including buccal, bicortical, and palatal regions from CBCT scans.
- **Mask R-CNN Model Integration**: Utilizes advanced deep learning architectures to provide accurate and robust segmentation.
- **Progress Monitoring**: Real-time progress bar and log updates during segmentation tasks.
- **User-Friendly Interface**: Simple GUI integrated seamlessly within 3D Slicer.
- **Batch Processing**: Supports segmentation of single files or batch processing of entire directories of CBCT scans.

---

## Installation

### Prerequisites

- [3D Slicer 5.6+](https://www.slicer.org/)
- Python packages: `torch`, `torchvision`, `nibabel`, `numpy`, `scipy`, `requests`

The module checks and automatically installs these dependencies upon first usage.

### Setup


 **Load the Module in 3D Slicer**

   - Open 3D Slicer.
   - Navigate to `Module → Slicer Automated Dental Tools → CLIC`.

---

## Usage

### Step 1: Load CBCT Data

- Use the file dialog to select your CBCT image or directory containing multiple CBCT images.
- Available sample data for testing: [MN138.nii](https://github.com/ashmoy/maskRcnn/releases/tag/test_files/assets/MN138.nii) , [MG_test_scan.nii.gz](https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.1/MG_test_scan.nii.gz)

![image](https://github.com/user-attachments/assets/3f0e5bf4-4d0e-400f-b779-f16aa7d7f9d3)

### Step 2: Download/Select Model

- Click "Download Model" if using for the first time, or specify your existing model directory.

![image](https://github.com/user-attachments/assets/2082c8ff-20bd-4d72-828e-59b8606fb01b)


### Step 3: Configure your output

- Click "Choose save folder" if you want to a specific output folder or clic on save in input folder if you want the outputs in the same folder than the input
- write the suffix you want for th outputs

![image](https://github.com/user-attachments/assets/1a2b5908-63ce-4d56-9bea-447fd0446c7e)


### Step 3: Run Segmentation

- Click **Predict** to begin segmentation, if it s the first time you run a pop up will maybe ask you to install some dependency, clic on 'yes', the process will take a little time to start.
- Observe the progress bar and log output for real-time feedback.

![image](https://github.com/user-attachments/assets/be74b94e-ff85-4f04-ad33-db2dfd673546)


### Step 4: Visualize Results

- Segmentations automatically load into Slicer's viewer with clear labeling.
- A color-coded legend (Buccal, Bicortical, Palatal) appears in slice views.
- Be carefull, if you have scans with differents field of view in your input folder and you try to navigate through the scans when th process is end the views can turn black, in this case you have to clic on the button reset field of view.

![image](https://github.com/user-attachments/assets/d5ebffe7-08c9-4e06-bbbf-5b391d5caa8e)


![image](https://github.com/user-attachments/assets/eb032449-9653-43d6-a83b-b9019e3fe39a)




## Contributing



We welcome contributions! Please open issues or pull requests for enhancements and bug fixes.

---

## Acknowledgments

@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}

Thanks to the 3D Slicer community and open-source developers contributing to medical imaging software.

