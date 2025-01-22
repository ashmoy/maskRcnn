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





def remove_invalid_boxes(boxes, labels, masks,image_name, slide_idx):
    """Supprime les boîtes englobantes avec une hauteur ou une largeur nulle."""
    valid_boxes = []
    valid_labels = []
    valid_masks = []

    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        width = x1 - x0
        height = y1 - y0
        # print(f"Boîte {i}: largeur={width}, hauteur={height}")
        if width > 0 and height > 0:
            valid_boxes.append(box)
            #print(f"Boîte valide trouvée : {box}")
            valid_labels.append(labels[i])
            valid_masks.append(masks[i])
            #print('taille boites valide',len(valid_boxes))



            # print(f"Boîte non valide trouvée et supprimée : {box}")
            # print(f"Aucune boîte valide après filtrage pour l'image '{image_name}', slide {slide_idx}.")

    if len(valid_boxes) == 0:
        #print("Aucune boîte valide après filtrage.")
        return None, None, None

    return torch.stack(valid_boxes), torch.stack(valid_labels), torch.stack(valid_masks)




def remove_alpha_channel(image):
    """Fonction pour enlever le canal alpha s'il est présent."""
    if image.shape[0] == 4:  # Vérifie si l'image a 4 canaux (RGBA)
        image = image[:3, :, :]  # Garde uniquement les 3 premiers canaux (RGB)
    return image



class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "scans2"))))
        #print(os.listdir(os.path.join(root, "scans")))
        self.masks = list(sorted(os.listdir(os.path.join(root, "seg_classe2"))))
         # Vérification du nombre de fichiers dans les deux dossiers
        assert len(self.imgs) == len(self.masks), "Le nombre d'images et de masques ne correspond pas !"


    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "scans2", self.imgs[idx])
        mask_path = os.path.join(self.root, "seg_classe2", self.masks[idx])
        img = read_image(img_path)
        img = remove_alpha_channel(img)
        mask = read_image(mask_path)
        mask = remove_alpha_channel(mask)


            # Assumons que 255 doit être 3, 1 reste 1, et 2 reste 2
        mask[mask == 255] = 3
        mask[mask == 191] = 2
        mask[mask == 127] = 1
        mask[mask == 0] = 0  # Le fond reste à 0
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)

        obj_ids = torch.unique(mask)
        # print(f"Objets détectés : {obj_ids}")

        # Vérifier s'il n'y a que le fond (tensor([0]))
        if len(obj_ids) == 1 and obj_ids[0] == 0:
            # print(f"Aucun objet trouvé pour l'image {self.imgs[idx]}, elle sera ignorée.")
            return None



        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # print(f"Objets détectés après suppression du fond : {obj_ids}")
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
       # Create binary masks, ensuring they all have the same size as the original image
        height, width = mask.shape[-2:]  # Get height and width of the original image
        masks = torch.zeros((num_objs, height, width), dtype=torch.uint8)  # Create an empty mask tensor

        for i, obj_id in enumerate(obj_ids):
            masks[i] = (mask == obj_id).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # print(f"Masks shape: {masks.shape}")
        # print(f"Boxes: {boxes}")

        #print(boxes)

        # Les labels sont basés sur les valeurs de `obj_ids`
        labels = obj_ids.to(dtype=torch.int64)

        boxes, labels, masks = remove_invalid_boxes(boxes, labels, masks,self.imgs[idx], idx)


        if boxes is None:
                return []  # Ignorer cette image si aucune boîte n'est valide


        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        #print('-----------labels--------------',labels)
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)














def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model











def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)








model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
dataset = PennFudanDataset('data3', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
     dataset,
     batch_size=2,
     shuffle=True,
     collate_fn=utils.collate_fn
 )

 # For Training
images, targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]

output = model(images, targets)  # Returns losses and detections
#print('train',output)

#  For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)  # Returns predictions
#print(predictions[0])







# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 4
# use our dataset and defined transformations
dataset = PennFudanDataset('data3', get_transform(train=True))
dataset_test = PennFudanDataset('data3', get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
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

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 2 epochs
num_epochs =10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)



# Ensure the model is in evaluation mode
model.eval()



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

def process_folder(input_folder, model, eval_transform, device, output_folder):
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parcourir tous les fichiers images dans le dossier d'entrée
    for image_file in os.listdir(input_folder):
        if image_file.endswith(".png"):  # Assurez-vous que seuls les fichiers PNG sont traités
            image_path = os.path.join(input_folder, image_file)
            process_image(image_path, model, eval_transform, device, output_folder)






if __name__ == "__main__":
    # Dossier d'entrée et de sortie à passer directement
    input_folder = "data3/scans"  # Remplacez par le chemin vers votre dossier d'entrée
    output_folder = "output_enzo"  # Remplacez par le chemin vers votre dossier de sortie

    # Charger le modèle et les transformations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_transform = get_transform(train=False)  # Utilisez votre fonction de transformation

    # Traiter le dossier d'entrée
    process_folder(input_folder, model, eval_transform, device, output_folder)






# Initialize empty lists to store true and predicted labels
all_true_labels = []
all_pred_labels = []

for images, targets in data_loader_test:
    images = list(img.to(device) for img in images)  # Move images to device

    # Get predictions from the model
    with torch.no_grad():
        x = eval_transform(images[0])  # Apply the evaluation transform
        x = x[:3, ...].to(device)  # Convert RGBA -> RGB and move to device
        predictions = model([x, ])
        pred = predictions[0]

    # Extract true labels from targets (on the CPU)
    true_labels = [target['labels'].cpu().numpy() for target in targets]

    # Sort predicted scores to get the highest scoring labels
    scores = pred["scores"].cpu().numpy()
    sorted_indices = scores.argsort()[::-1].copy()  # Sort scores in descending order and make a copy

    # For each true label, find the corresponding predicted label
    for i, true in enumerate(true_labels):
        if len(true) == 1:
            # If there's only one true label, take the label with the highest score
            top_label = pred["labels"][sorted_indices[0]].item()  # Take top 1 label
            all_true_labels.append(true[0])
            all_pred_labels.append(top_label)

        elif len(true) == 2:
            # If there are two true labels, take the top 2 predicted labels
            top_labels = pred["labels"][sorted_indices[:2]].cpu().numpy().copy()  # Take top 2 labels and make a copy
            all_true_labels.extend(true)  # Add both true labels
            all_pred_labels.extend(top_labels)  # Add the corresponding predicted labels

# Ensure all labels are within the range of valid classes
valid_true_labels = [label for label in all_true_labels if label <= num_classes]
valid_pred_labels = [label for label in all_pred_labels if label <= num_classes]

# Check if the lengths match before generating the confusion matrix
if len(valid_true_labels) != len(valid_pred_labels):
    print(f"Error: Mismatch in the number of true ({len(valid_true_labels)}) and predicted ({len(valid_pred_labels)}) labels.")
else:
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(valid_true_labels, valid_pred_labels)
    conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    print("Confusion Matrix:")
    print(conf_matrix_percentage)



    # Define the output folder and create it if it doesn't exist
    output_folder = "confusion_matrix_output"
    os.makedirs(output_folder, exist_ok=True)

    # Save the confusion matrix plot as a PNG image
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_percentage, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=np.arange(num_classes),
                yticklabels=np.arange(num_classes))
    plt.title("Confusion Matrix (Percentage)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # Save the plot
    output_path = os.path.join(output_folder, "confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Confusion matrix saved to {output_path}")
