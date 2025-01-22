import os
import shutil
import argparse

def strip_extension(filename):
    """Enlever toutes les extensions .nii ou .nii.gz pour obtenir le nom de base"""
    if filename.endswith(".nii.gz"):
        return filename[:-7]  # Supprimer ".nii.gz"
    elif filename.endswith(".nii"):
        return filename[:-4]  # Supprimer ".nii"
    else:
        return os.path.splitext(filename)[0]  # Autre extension, traitement par défaut

def copy_matching_files(folder_input_1, folder_input_2, folder_output_1, folder_output_2):
    # Créer les dossiers de sortie s'ils n'existent pas
    os.makedirs(folder_output_1, exist_ok=True)
    os.makedirs(folder_output_2, exist_ok=True)

    # Lister tous les fichiers dans les deux dossiers d'entrée
    files_input_1 = {strip_extension(f): f for f in os.listdir(folder_input_1)}
    files_input_2 = {strip_extension(f): f for f in os.listdir(folder_input_2)}

    # Trouver les fichiers avec le même nom de base (sans extension) dans les deux dossiers
    matching_files = set(files_input_1.keys()).intersection(files_input_2.keys())

    # Copier les fichiers dans les dossiers de sortie correspondants
    for base_name in matching_files:
        # Copier depuis folder_input_1 vers folder_output_1
        src_file_1 = os.path.join(folder_input_1, files_input_1[base_name])
        dest_file_1 = os.path.join(folder_output_1, files_input_1[base_name])
        shutil.copy(src_file_1, dest_file_1)
        print(f"Copié: {src_file_1} -> {dest_file_1}")

        # Copier depuis folder_input_2 vers folder_output_2
        src_file_2 = os.path.join(folder_input_2, files_input_2[base_name])
        dest_file_2 = os.path.join(folder_output_2, files_input_2[base_name])
        shutil.copy(src_file_2, dest_file_2)
        print(f"Copié: {src_file_2} -> {dest_file_2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copier les fichiers correspondants de deux dossiers vers deux dossiers de sortie.")
    parser.add_argument("--folder_input_1", required=True, help="Chemin du premier dossier d'entrée.")
    parser.add_argument("--folder_input_2", required=True, help="Chemin du deuxième dossier d'entrée.")
    parser.add_argument("--folder_output_1", required=True, help="Chemin du premier dossier de sortie.")
    parser.add_argument("--folder_output_2", required=True, help="Chemin du deuxième dossier de sortie.")

    args = parser.parse_args()
    
    # Appeler la fonction pour copier les fichiers correspondants
    copy_matching_files(args.folder_input_1, args.folder_input_2, args.folder_output_1, args.folder_output_2)
