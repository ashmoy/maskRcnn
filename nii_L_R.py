import os
import nibabel as nib
import torch
import argparse

def label_left_right(segmentation_path, output_folder):
    # Charger le fichier NIfTI
    nii_img = nib.load(segmentation_path)
    img_data = nii_img.get_fdata()

    # Convertir en tenseur PyTorch pour manipuler plus facilement
    img_tensor = torch.tensor(img_data)

    # Diviser en deux moitiés le long de l'axe X (gauche-droite)
    midline = img_tensor.shape[0] // 2

    # Créer des masques pour la gauche et la droite
    left_mask = torch.zeros_like(img_tensor)
    right_mask = torch.zeros_like(img_tensor)

    # Tout ce qui est à gauche de la ligne médiane est gauche
    left_mask[:midline, :, :] = img_tensor[:midline, :, :]

    # Tout ce qui est à droite de la ligne médiane est droite
    right_mask[midline:, :, :] = img_tensor[midline:, :, :]

    # Vous pouvez maintenant assigner des labels en fonction de la position
    # Par exemple : 1 pour gauche, 2 pour droite
    left_mask[left_mask > 0] = 1  # Label pour la gauche
    right_mask[right_mask > 0] = 2  # Label pour la droite

    # Fusionner les deux masques
    labeled_img = left_mask + right_mask

    # Créer le chemin de sortie
    output_filename = os.path.basename(segmentation_path).replace("_SEG.nii.gz", "_labeled.nii")
    output_path = os.path.join(output_folder, output_filename)

    # Sauvegarder le fichier segmenté
    labeled_nii = nib.Nifti1Image(labeled_img.numpy(), nii_img.affine)
    nib.save(labeled_nii, output_path)

    print(f"Labeled segmentation saved to: {output_path}")

def process_folder(input_folder, output_folder):
    # Vérifier si le dossier de sortie existe, sinon le créer
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parcourir tous les fichiers NIfTI dans le dossier d'entrée
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii.gz"):
            segmentation_path = os.path.join(input_folder, filename)
            label_left_right(segmentation_path, output_folder)

if __name__ == "__main__":
    # Parser les arguments
    parser = argparse.ArgumentParser(description="Label left and right segmentations in NIfTI files.")
    parser.add_argument("--input_folder", type=str, help="Folder containing the NIfTI files.")
    parser.add_argument("--output_folder", type=str, help="Folder where the labeled NIfTI files will be saved.")

    args = parser.parse_args()

    # Traiter le dossier d'entrée
    process_folder(args.input_folder, args.output_folder)
