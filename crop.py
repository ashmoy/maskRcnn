import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

def crop_to_match(scan_data, seg_data):
    """
    Recadre le scan et/ou la segmentation pour qu'ils aient les mêmes dimensions.
    - Si le scan est plus grand, il est recadré.
    - Si la segmentation est plus grande, elle est recadrée.
    """
    scan_shape = scan_data.shape
    seg_shape = seg_data.shape

    # Calculer les dimensions minimales communes
    min_x = min(scan_shape[0], seg_shape[0])
    min_y = min(scan_shape[1], seg_shape[1])
    min_z = min(scan_shape[2], seg_shape[2])

    # Recadrer le scan
    scan_data_cropped = scan_data[:min_x, :min_y, :min_z]
    
    # Recadrer la segmentation
    seg_data_cropped = seg_data[:min_x, :min_y, :min_z]

    return scan_data_cropped, seg_data_cropped

def process_scans_and_segmentations(scans_folder, segmentations_folder, output_scans_folder, output_segmentations_folder):
    """
    Parcourt les dossiers de scans et de segmentations, recadre chaque scan et segmentation
    pour qu'ils aient la même taille et sauvegarde les résultats dans des dossiers de sortie.
    """
    os.makedirs(output_scans_folder, exist_ok=True)
    os.makedirs(output_segmentations_folder, exist_ok=True)

    scan_files = sorted([f for f in os.listdir(scans_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])
    seg_files = sorted([f for f in os.listdir(segmentations_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])

    assert len(scan_files) == len(seg_files), "Le nombre de scans et de segmentations ne correspond pas."

    with tqdm(total=len(scan_files), desc="Traitement des scans et segmentations") as pbar:
        for scan_file, seg_file in zip(scan_files, seg_files):
            assert scan_file == seg_file, f"Les fichiers {scan_file} et {seg_file} ne correspondent pas."

            scan_path = os.path.join(scans_folder, scan_file)
            seg_path = os.path.join(segmentations_folder, seg_file)

            try:
                # Charger les données du scan et de la segmentation
                scan_img = nib.load(scan_path)
                seg_img = nib.load(seg_path)

                scan_data = scan_img.get_fdata()
                seg_data = seg_img.get_fdata()

                # Recadrer pour correspondre
                scan_cropped, seg_cropped = crop_to_match(scan_data, seg_data)

                # Créer les nouvelles images NIfTI
                scan_cropped_img = nib.Nifti1Image(scan_cropped, scan_img.affine, scan_img.header)
                seg_cropped_img = nib.Nifti1Image(seg_cropped, seg_img.affine, seg_img.header)

                # Sauvegarder les fichiers recadrés en .nii.gz
                scan_output_path = os.path.join(output_scans_folder, os.path.splitext(scan_file)[0] + '.nii.gz')
                seg_output_path = os.path.join(output_segmentations_folder, os.path.splitext(seg_file)[0] + '.nii.gz')

                nib.save(scan_cropped_img, scan_output_path)
                nib.save(seg_cropped_img, seg_output_path)

            except Exception as e:
                print(f"Erreur lors du traitement de {scan_file} et {seg_file}: {e}")

            pbar.update(1)


if __name__ == "__main__":
    scans_folder = "original_scan"
    segmentations_folder = "seg_with_classes"
    output_scans_folder = "original_scans_crop"
    output_segmentations_folder ="seg_crops"
    if os.path.exists(scans_folder) and os.path.exists(segmentations_folder):
        process_scans_and_segmentations(scans_folder, segmentations_folder, output_scans_folder, output_segmentations_folder)
    else:
        print("Un ou plusieurs dossiers spécifiés n'existent pas.")
