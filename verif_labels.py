import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import argparse

def check_labels_in_masks(folder):
    """
    Vérifie les labels uniques présents dans chaque fichier .nii d'un dossier donné.

    Args:
        folder (str): Chemin du dossier contenant les fichiers .nii.

    Returns:
        dict: Un dictionnaire avec le nom de chaque fichier comme clé et les labels uniques comme valeur.
    """
    # Récupérer tous les fichiers .nii dans le dossier
    nii_files = [f for f in os.listdir(folder) if f.endswith('.nii')]

    if not nii_files:
        print("Aucun fichier .nii trouvé dans le dossier.")
        return {}

    labels_dict = {}

    print("\nAnalyse des fichiers :")
    # Utilisation de tqdm pour afficher la barre de progression
    for file in tqdm(nii_files, desc="Traitement des fichiers", unit="fichier"):
        file_path = os.path.join(folder, file)
        try:
            # Charger le fichier .nii
            mask_data = nib.load(file_path).get_fdata()

            # Trouver les labels uniques
            unique_labels = np.unique(mask_data)
            print(f"Labels uniques dans {file}: {unique_labels}")

        except Exception as e:
            print(f"Erreur lors de l'analyse de {file}: {e}")

    return labels_dict

def main():
    parser = argparse.ArgumentParser(description="Analyse les labels uniques dans les fichiers .nii d'un dossier.")
    parser.add_argument('--folder', type=str, required=True, help='Chemin du dossier contenant les fichiers .nii.')
    args = parser.parse_args()

    folder = args.folder

    if not os.path.exists(folder):
        print(f"Le dossier spécifié n'existe pas : {folder}")
        return

    labels_dict = check_labels_in_masks(folder)

if __name__ == "__main__":
    main()