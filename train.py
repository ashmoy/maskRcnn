#######################################
# Importation des librairies
#######################################
import os
import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
import utils
from engine import train_one_epoch, evaluate
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import argparse

#######################################
# Fonction pour supprimer les boîtes non valides
#######################################
def remove_invalid_boxes(boxes, labels, masks, image_name, slide_idx):
    valid_boxes = []
    valid_labels = []
    valid_masks = []

    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        width = x1 - x0
        height = y1 - y0
        if width > 0 and height > 0:
            valid_boxes.append(box)
            valid_labels.append(labels[i])
            valid_masks.append(masks[i])

    if len(valid_boxes) == 0:
        return None, None, None

    return torch.stack(valid_boxes), torch.stack(valid_labels), torch.stack(valid_masks)

#######################################
# Fonction pour retirer le canal alpha
#######################################
def remove_alpha_channel(image):
    if image.shape[0] == 4:
        image = image[:3, :, :]  # Garde uniquement les 3 premiers canaux (RGB)
    return image

#######################################
# Classe pour le Dataset PennFudan
#######################################
class CanineDataset(torch.utils.data.Dataset):
    def __init__(self, scans_folder, masks_folder, transforms):
        self.scans_folder = scans_folder
        self.masks_folder = masks_folder
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(scans_folder)))
        self.masks = list(sorted(os.listdir(masks_folder)))
        assert len(self.imgs) == len(self.masks), "Le nombre d'images et de masques ne correspond pas !"

    def __getitem__(self, idx):
        img_path = os.path.join(self.scans_folder, self.imgs[idx])
        mask_path = os.path.join(self.masks_folder, self.masks[idx])
        img = read_image(img_path)
        img = remove_alpha_channel(img)
        mask = read_image(mask_path)
        mask = remove_alpha_channel(mask)

        mask[mask == 255] = 3
        mask[mask == 191] = 2
        mask[mask == 127] = 1
        mask[mask == 0] = 0

        obj_ids = torch.unique(mask)
        if len(obj_ids) == 1 and obj_ids[0] == 0:
            return None

        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        height, width = mask.shape[-2:]
        masks = torch.zeros((num_objs, height, width), dtype=torch.uint8)
        for i, obj_id in enumerate(obj_ids):
            masks[i] = (mask == obj_id).to(dtype=torch.uint8)

        boxes = masks_to_boxes(masks)
        labels = obj_ids.to(dtype=torch.int64)

        boxes, labels, masks = remove_invalid_boxes(boxes, labels, masks, self.imgs[idx], idx)
        if boxes is None:
            return []

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        img = tv_tensors.Image(img)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)


        # print('image',img)
        # print('target',target)

        return img, target

    def __len__(self):
        return len(self.imgs)

#######################################
# Fonction pour obtenir le modèle de segmentation d'instance
#######################################
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

#######################################
# Fonction pour définir les transformations
#######################################
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)




###########################################
# Fonction principale pour l'entraînement #
###########################################


def train_model(scans_folder, masks_folder, num_epochs,lr, device, output_folder):
    dataset = CanineDataset(scans_folder, masks_folder, get_transform(train=True))
    dataset_test = CanineDataset(scans_folder, masks_folder, get_transform(train=False))
    dataset_val = CanineDataset(scans_folder, masks_folder, get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    total_size = len(dataset)
    train_split = int(0.7 * total_size)
    val_split = int(0.85 * total_size)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    # Subsets pour entraînement, validation, et test
    dataset = torch.utils.data.Subset(dataset, train_indices)
    dataset_val = torch.utils.data.Subset(dataset_val, val_indices)
    dataset_test = torch.utils.data.Subset(dataset_test, test_indices)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn
    )

    model = get_model_instance_segmentation(num_classes=4)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr= lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=5)
            # Adjust learning rate
        #lr_scheduler.step()
        evaluate(model, data_loader_val, device=device)

    # Générer la matrice de confusion après l'entraînement
    generate_confusion_matrix(data_loader_test, model, device, 4, get_transform(train=False), output_folder)
    #print(f"Entraînement terminé après {num_epochs} époques. Les résultats sont enregistrés dans {output_folder}.")
        # Save the model after training
    model_save_path = os.path.join(output_folder, 'trained_model.pth')
    torch.save(model.state_dict(), model_save_path)

    return model_save_path





#######################################
# Fonction pour afficher les bounding boxes et générer la matrice de confusion
#######################################
def process_image(image_path, model, eval_transform, device, output_folder):
    # Chargement de l'image et modèle en mode évaluation
    image = read_image(image_path)

    model.eval()

    with torch.no_grad():
        x = eval_transform(image)
        # Convertir RGBA -> RGB et déplacer sur le bon appareil
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    # Normalisation de l'image pour affichage
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    # Tri des boîtes englobantes par score et sélection des deux meilleures
    scores = pred["scores"].cpu().numpy()
    sorted_indices = scores.argsort()[::-1]  # Trier les indices en fonction des scores décroissants

    # Filtrer les boîtes ayant des scores > 0.9
    filtered_indices = [idx for idx in sorted_indices if scores[idx] > 0.9]

    # Prendre au maximum les deux meilleures boîtes avec des scores > 0.9
    top_indices = filtered_indices[:2]

    if len(top_indices) > 0:
        top_boxes = pred["boxes"][top_indices].long()  # Prend les boîtes correspondantes
        top_labels = pred["labels"][top_indices]  # Prend les labels correspondants
        top_scores = scores[top_indices]  # Prend les scores correspondants

        # Ajout des boîtes englobantes et des labels
        pred_labels = [f"label: {label.item()}, score: {score:.3f}" for label, score in zip(top_labels, top_scores)]
        output_image = draw_bounding_boxes(image, top_boxes, pred_labels, colors="red")

        # Ajout des masques de segmentation (seulement les masques des deux meilleures boîtes)
        top_masks = pred["masks"][top_indices]
        masks = (top_masks > 0.8).squeeze(1)
        output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

        # Sauvegarder l'image avec bounding boxes et masques
        output_filename = os.path.basename(image_path).replace(".png", "_labeled.png")
        output_path = os.path.join(output_folder, output_filename)

        plt.figure(figsize=(12, 12))
        plt.imshow(output_image.permute(1, 2, 0))  # S'assurer du bon format [H, W, C]
        plt.axis('off')  # Masquer les axes pour une meilleure visualisation
        plt.savefig(output_path)  # Sauvegarde l'image dans le dossier de sortie
        plt.close()

        print(f"Image sauvegardée: {output_path}")
    else:
        print(f"Aucune boîte avec un score supérieur à 0.9 trouvée pour {image_path}")


#######################################
# Fonction pour traiter un dossier entier
#######################################



def process_folder(input_folder, model, eval_transform, device, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(input_folder):
        if image_file.endswith(".png"):
            image_path = os.path.join(input_folder, image_file)
            process_image(image_path, model, eval_transform, device, output_folder)



#######################################
# Fonction pour charger le modèle sauvegardé
#######################################
def load_trained_model(model_path, device, num_classes=4):
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path))  # Charger les poids sauvegardés
    model.to(device)  # Envoyer le modèle sur le bon appareil (CPU/GPU)
    model.eval()  # Mettre le modèle en mode évaluation
    return model



#################################################
# Fonction pour générer la matrice de confusion #
#################################################



def generate_confusion_matrix(data_loader_test, model, device, num_classes, eval_transform, output_folder):
    all_true_labels = []
    all_pred_labels = []

    for images, targets in data_loader_test:
        images = list(img.to(device) for img in images)

        with torch.no_grad():
            x = eval_transform(images[0])
            x = x[:3, ...].to(device)
            predictions = model([x, ])
            pred = predictions[0]

        true_labels = [target['labels'].cpu().numpy() for target in targets]

        scores = pred["scores"].cpu().numpy()
        sorted_indices = scores.argsort()[::-1].copy()

        for i, true in enumerate(true_labels):
            if len(true) == 1:
                top_label = pred["labels"][sorted_indices[0]].item()
                all_true_labels.append(true[0])
                all_pred_labels.append(top_label)
            elif len(true) == 2:
                top_labels = pred["labels"][sorted_indices[:2]].cpu().numpy().copy()
                all_true_labels.extend(true)
                all_pred_labels.extend(top_labels)

    valid_true_labels = [label for label in all_true_labels if label <= num_classes]
    valid_pred_labels = [label for label in all_pred_labels if label <= num_classes]

    if len(valid_true_labels) != len(valid_pred_labels):
        print(f"Error: Mismatch in the number of true and predicted labels.")
    else:
        conf_matrix = confusion_matrix(valid_true_labels, valid_pred_labels)
        conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print("Confusion Matrix:")
        print(conf_matrix_percentage)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_percentage, annot=True, cmap="Blues", fmt=".2f",
                    xticklabels=np.arange(num_classes),
                    yticklabels=np.arange(num_classes))
        plt.title("Confusion Matrix (Percentage)")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")

        output_path = os.path.join(output_folder, "confusion_matrix.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Confusion matrix saved to {output_path}")



#####################################################
# Utilisation d'argparse pour définir les arguments #
#####################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model with dataset and generating results.")
    parser.add_argument('--scans_folder', type=str, required=True, help='Dossier contenant les scans')
    parser.add_argument('--masks_folder', type=str, required=True, help='Dossier contenant les masques')
    parser.add_argument('--epochs', type=int, default=10, help='Nombre d\'époques pour l\'entraînement')
    parser.add_argument('--output_folder', type=str, required=True, help='Dossier où sauvegarder les résultats')
    parser.add_argument('--lr', type=float, default=0.005, help='Taux d\'apprentissage pour l\'optimiseur')
    parser.add_argument('--model_path', type=str, help='Chemin vers le modèle sauvegardé', default=None)

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Vérifier si un modèle sauvegardé est fourni, sinon entraîner un modèle
    if args.model_path:
        model = load_trained_model(args.model_path, device)
    else:
        # Appel de la fonction d'entraînement
        train_model(args.scans_folder, args.masks_folder, args.epochs, args.lr, device, args.output_folder)
        model_path = os.path.join(args.output_folder, 'trained_model.pth')
        model = load_trained_model(model_path, device)


    # Traitement des images avec bounding boxes et labels
    process_folder(args.scans_folder, model, get_transform(train=False), device, args.output_folder)






