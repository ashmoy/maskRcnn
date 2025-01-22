
import os
import argparse
from PIL import Image
import numpy as np

def check_non_zero_components(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            file_path = os.path.join(input_folder, filename)
            try:
                img = Image.open(file_path)
                img_array = np.array(img)

                if np.any(img_array != 0):
                    print('.')
                else:
                    print(f'{filename}: Ne contient aucun composant non nul.')
            except Exception as e:
                print(f'Erreur avec le fichier {filename}: {e}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vérification des fichiers PNG pour des composants non nuls.")
    parser.add_argument("--input_folder", type=str, help="Chemin vers le dossier contenant les fichiers PNG.")
    args = parser.parse_args()

    check_non_zero_components(args.input_folder)















