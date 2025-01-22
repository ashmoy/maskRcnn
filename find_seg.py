import os
import shutil
import argparse

def copy_seg_files(input_folder, output_folder):
    # Parcourir tous les sous-dossiers du dossier d'entrée
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Vérifier si le fichier se termine par "SEG.nii.gz"
            if file.endswith("scan.nii.gz"):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(output_folder, file)
                
                # Copier le fichier uniquement s'il n'existe pas déjà dans le dossier de sortie
                if not os.path.exists(destination_file):
                    shutil.copy2(source_file, destination_file)
                    print(f"Copié: {source_file} vers {destination_file}")
                else:
                    print(f"Fichier existe déjà: {destination_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copier les fichiers SEG.nii.gz dans le dossier de sortie s'ils n'existent pas déjà")
    parser.add_argument("--input_folder", help="Chemin vers le dossier d'entrée")
    parser.add_argument("--output_folder", help="Chemin vers le dossier de sortie")
    
    args = parser.parse_args()
    
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # Lancer la fonction pour copier les fichiers
    copy_seg_files(args.input_folder, args.output_folder)
