import os
import SimpleITK as sitk
import argparse


def convert_gipl_to_nii(input_folder, output_folder):
    """
    Convertit tous les fichiers .gipl.gz dans un dossier en .nii.gz.

    Args:
        input_folder (str): Chemin vers le dossier contenant les fichiers .gipl.gz.
        output_folder (str): Chemin vers le dossier pour sauvegarder les fichiers .nii.gz.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parcourir tous les fichiers dans le dossier d'entrée
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.gipl.gz'):
            input_path = os.path.join(input_folder, file_name)
            output_file_name = os.path.splitext(os.path.splitext(file_name)[0])[0] + '.nii.gz'
            output_path = os.path.join(output_folder, output_file_name)

            # Charger le fichier .gipl.gz
            image = sitk.ReadImage(input_path)

            # Sauvegarder au format .nii.gz
            sitk.WriteImage(image, output_path)

            print(f"Converted {file_name} to {output_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .gipl.gz files to .nii.gz.")
    parser.add_argument("--input_folder", required=True, type=str, help="Path to the folder containing .gipl.gz files.")
    parser.add_argument("--output_folder", required=True, type=str, help="Path to the folder for saving .nii.gz files.")
    args = parser.parse_args()

    convert_gipl_to_nii(args.input_folder, args.output_folder)
