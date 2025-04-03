import os
import slicer
import qt
import webbrowser
import ctk
from slicer.ScriptedLoadableModule import *
import torch
import subprocess
import numpy as np
import nibabel as nib
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import scipy.ndimage
import time
from CondaSetUp import CondaSetUpCall, CondaSetUpCallWsl
import threading
from slicer.util import VTKObservationMixin
import logging
import requests
import glob
import zipfile
import queue
import subprocess
import sys

def PathFromNode(node):
    storageNode = node.GetStorageNode()
    if storageNode is not None:
        filepath = storageNode.GetFullNameFromFileName()
    else:
        filepath = None
    return filepath

# -- Classe pour les signaux de mise à jour --
class ProgressUpdater(qt.QObject):
    progressChanged = qt.Signal(int)
    logChanged = qt.Signal(str)

class CLIC(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CLIC"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = ["CondaSetUp"]
        self.parent.contributors = ["Your Name"]
        self.parent.helpText = """This module performs segmentation using a Mask R-CNN model."""
        self.parent.acknowledgementText = "Thanks to the community for support."

class CLICWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.CBCT_as_input = True   # True: CBCT image, False: surface IOS
        self.folder_as_input = False
        self.MRMLNode_scan = None
        self.input_path = None
        self.model_folder = None
        self.output_folder = None
        self.goup_output_files = False
        self.scan_count = 0
        self.currentSegNode = None  # Pour suivre le volume affiché
        self.ui_queue = queue.Queue()
    def UpdateSaveType(self, checked):
        self.goup_output_files = checked
        self.ui.SearchSaveFolder.setHidden(checked)
        self.ui.SaveFolderLineEdit.setHidden(checked)
        self.ui.PredictFolderLabel.setHidden(checked)

    def setup(self):
        self.conda_wsl = CondaSetUpCallWsl()
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/CLIC.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Création de la logique métier
        self.logic = CLICLogic()

        # Instanciation de l'updater pour les signaux
        self.progressUpdater = ProgressUpdater()

        # La progressBar "PredScanProgressBar" sera utilisée pour le téléchargement,
        # et "progressBar" pour le traitement des slices.
        self.progressUpdater.progressChanged.connect(lambda value: self.ui.progressBar.setValue(value))
        self.progressUpdater.logChanged.connect(lambda text: self.ui.logTextEdit.append(text))

        self.ui.SavePredictCheckBox.toggled.connect(self.UpdateSaveType)
        self.ui.DownloadModelPushButton.clicked.connect(self.onModelDownloadButton)
        self.ui.SearchScanFolder.clicked.connect(self.onSearchScanButton)
        self.ui.SearchModelFolder.clicked.connect(self.onSearchModelButton)
        self.ui.SearchSaveFolder.clicked.connect(self.onSearchSaveButton)
        self.ui.PredictionButton.clicked.connect(self.onPredictButton)
        self.ui.CancelButton.clicked.connect(self.onCancel)

        # Initialisation de la visibilité des éléments d'interface
        self.ui.progressBar.setVisible(False)
        self.ui.PredScanProgressBar.setVisible(False)  # Masquer la barre de progression au démarrage
        self.RunningUI(False)

        self.initializeParameterNode()

    def SwitchInputExtension(self, index):
        if index == 1:
            self.ui.ScanPathLabel.setText('DICOM Folder')
        else:
            self.ui.ScanPathLabel.setText('Scan File/Folder')

    def onNodeChanged(self):
        self.MRMLNode_scan = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox.currentNodeID)
        if self.MRMLNode_scan:
            self.input_path = PathFromNode(self.MRMLNode_scan)
            self.scan_count = 1
            self.ui.PrePredInfo.setText("Number of scans to process: 1")
            # Charger le scan dans Slicer (si ce n'est pas déjà fait)
            self.load_nii_in_slicer(self.input_path)
            return True
        return False

    def onModelDownloadButton(self):
        model_url = "https://github.com/ashmoy/maskRcnn/releases/download/model/final_model.pth"
        default_model_folder = os.path.join(os.path.expanduser("~"), "Documents", "CLIC_Models")
        os.makedirs(default_model_folder, exist_ok=True)
        model_path = os.path.join(default_model_folder, "final_model.pth")
        try:
            self.ui.PredScanLabel.setText("Downloading model...")
            self.ui.PredScanProgressBar.setVisible(True)  # Afficher la barre de progression
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            chunk_size = 1024
            with open(model_path, 'wb') as file:
                for data in response.iter_content(chunk_size=chunk_size):
                    file.write(data)
                    downloaded_size += len(data)
                    progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                    self.ui.PredScanProgressBar.setValue(int(progress))
            self.ui.lineEditModelPath.setText(default_model_folder)
            self.model_folder = default_model_folder
            self.ui.PredScanLabel.setText("Model downloaded successfully!")
        except Exception as e:
            qt.QMessageBox.warning(self.parent, 'Error', f"Failed to download model: {str(e)}")
            self.ui.PredScanLabel.setText("Error during download.")
        finally:
            self.ui.PredScanProgressBar.setVisible(False)  # Masquer la barre après téléchargement

    def onSearchScanButton(self):
        scan_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
        if scan_folder:
            self.input_path = scan_folder
            self.ui.lineEditScanPath.setText(self.input_path)

    def onSearchModelButton(self):
        model_path = qt.QFileDialog.getExistingDirectory(self.parent, "Select Model Folder", "")
        if model_path:
            self.model_folder = model_path
            self.ui.lineEditModelPath.setText(model_path)

    def onSearchSaveButton(self):
        save_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select Save Folder", "")
        if save_folder:
            self.output_folder = save_folder
            self.ui.SaveFolderLineEdit.setText(save_folder)


    def check_dependencies(self):
        # Ajouter un message dans les logs pour indiquer le début de la vérification
        self.ui_queue.put(("log", "Checking required libraries..."))

        required_libraries = ["torch", "nibabel", "numpy", "scipy", "requests"]
        missing_libraries = []

        # Vérifier les bibliothèques requises
        for lib in required_libraries:
            try:
                __import__(lib)
            except ImportError:
                missing_libraries.append(lib)

        # Si des bibliothèques sont manquantes, proposer de les installer
        if missing_libraries:
            self.ui_queue.put(("log", f"Missing libraries detected: {', '.join(missing_libraries)}"))
            reply = qt.QMessageBox.question(
                self.parent,
                "Missing Dependencies",
                f"The following libraries are missing: {', '.join(missing_libraries)}.\n"
                "Do you want to install them automatically?",
                qt.QMessageBox.Yes | qt.QMessageBox.No
            )

            if reply == qt.QMessageBox.Yes:
                for lib in missing_libraries:
                    try:
                        self.ui_queue.put(("log", f"Installing library: {lib}..."))
                        if lib == "torch":
                            # Installation spécifique pour PyTorch avec CUDA 11.8
                            subprocess.check_call([
                                sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
                                "--index-url", "https://download.pytorch.org/whl/cu118"
                            ])
                        else:
                            # Installation standard pour les autres bibliothèques
                            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                    except Exception as e:
                        self.ui_queue.put(("log", f"Failed to install {lib}: {str(e)}"))
                        qt.QMessageBox.critical(
                            self.parent,
                            "Installation Error",
                            f"Failed to install {lib}. Error: {str(e)}"
                        )
                        return False

                # Réessayer d'importer les bibliothèques après installation
                for lib in missing_libraries:
                    try:
                        __import__(lib)
                    except ImportError:
                        self.ui_queue.put(("log", f"Failed to load {lib} after installation."))
                        qt.QMessageBox.critical(
                            self.parent,
                            "Dependency Error",
                            f"Failed to load {lib} after installation. Please install it manually."
                        )
                        return False
            else:
                self.ui_queue.put(("log", "User chose not to install missing libraries."))
                qt.QMessageBox.critical(
                    self.parent,
                    "Missing Dependencies",
                    "Please install the missing libraries manually before proceeding."
                )
                return False

        # Ajouter un message dans les logs pour indiquer la fin de la vérification
        self.ui_queue.put(("log", "All required libraries are installed and ready to use."))
        return True

  
    def onPredictButton(self):
            # Vérifier les dépendances
        if not self.check_dependencies():
          return
        if not self.input_path:
            qt.QMessageBox.warning(self.parent, 'Warning', 'Please select an input file/folder')
            return
        if not self.model_folder:
            qt.QMessageBox.warning(self.parent, 'Warning', 'Please select a model folder')
            return
        param = {
            "input_path": self.input_path,
            "model_folder": self.model_folder,
            "output_dir": self.output_folder,
        }

        # Définition des callbacks pour progression, log et affichage.
        def update_progress(progress):
            self.ui_queue.put(("progress", progress))

        def update_log(message):
           self.ui_queue.put(("log", message))

        def display_segmentation(seg_file):
            self.ui_queue.put(("segmentation", seg_file))

        try:
            nii_files = glob.glob(os.path.join(self.input_path, "*.nii"))
            if nii_files:
                self.load_nii_in_slicer(nii_files[0])
            self.RunningUI(True)
            self.processThread = threading.Thread(
                target=self.logic.process,
                args=(param, update_progress, update_log, display_segmentation)
            )
            self.processThread.start()

            start_time = time.time()
            while self.processThread.is_alive():
                slicer.app.processEvents()
                current_time = time.time()
                self.ui.TimerLabel.setText(f"Time elapsed: {current_time - start_time:.2f}s")
                self.process_ui_queue()
                time.sleep(0.1)

            self.ui_queue.put(("log", "Segmentation completed successfully!"))
        except Exception as e:
            self.ui_queue.put(("log", f"An error occurred during segmentation: {str(e)}"))
        finally:
            self.RunningUI(False)


    def process_ui_queue(self):
      """Traite les messages dans la file d'attente pour mettre à jour l'interface utilisateur."""
      while not self.ui_queue.empty():
          action, data = self.ui_queue.get()
          if action == "progress":
              self.ui.progressBar.setValue(int(data))
          elif action == "log":
              self.ui.logTextEdit.append(data)
          elif action == "segmentation":
              self.load_segmentation(data)


    def load_segmentation(self, seg_file):
        if self.currentSegNode:
            slicer.mrmlScene.RemoveNode(self.currentSegNode)
            self.currentSegNode = None
        qt.QTimer.singleShot(0, lambda: self._load_segmentation_in_main_thread(seg_file))
    def create_legend_node(self, volume_node):
        """Crée et retourne un nœud de légende visible immédiatement"""
        # Création du nœud
        legend_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", "SegLegend")
        
        # Récupération des couleurs depuis le volume
        display_node = volume_node.GetDisplayNode()
        if not display_node:
            display_node = volume_node.CreateDefaultDisplayNode()
        
        # Construction HTML manuelle (compatible Python 3.6+)
        html_parts = [
            '<div style="',
            'background-color: rgba(0,0,0,0.7);',
            'padding: 10px;',
            'border-radius: 5px;',
            'font-family: Arial;',
            '">',
            '<h4 style="margin:5px 0;color:white;">LÉGENDE</h4>'
        ]
        
        for label, name in sorted(self.CLASS_NAMES.items()):
            color = display_node.GetColor(label)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(color[0]*255),
                int(color[1]*255),
                int(color[2]*255))
            html_parts.append(
                '<p style="margin:2px 0;">'
                '<span style="color:{};font-size:20px;vertical-align:middle;">■</span> '
                '<span style="color:white;font-size:16px;vertical-align:middle;">{}</span>'
                '</p>'.format(hex_color, name))
        
        html_parts.append('</div>')
        legend_node.SetText(''.join(html_parts))
        
        # Configuration d'affichage obligatoire
        legend_display = legend_node.CreateDefaultDisplayNode()
        legend_display.SetHorizontalAlignment(1)  # 1 = droite
        legend_display.SetVerticalAlignment(0)    # 0 = haut
        legend_display.SetBackgroundOpacity(0.7)
        legend_display.SetFontSize(16)
        legend_display.SetVisibility(True)
        
        return legend_node
    def _load_segmentation_in_main_thread(self, seg_file):
        try:
            # Chargement du volume
            loadedNodes = slicer.util.loadVolume(seg_file)
            if not loadedNodes or "volumeNode" not in loadedNodes:
                raise ValueError("Échec du chargement du volume")
                
            self.currentSegNode = loadedNodes["volumeNode"]
            
            # Suppression de l'ancienne légende
            old_legend = slicer.mrmlScene.GetFirstNodeByName("SegLegend")
            if old_legend:
                slicer.mrmlScene.RemoveNode(old_legend)
            
            # Création et affichage de la légende
            self.logic.create_legend_node(self.currentSegNode)
            
            # Configuration de la vue
            if hasattr(self, 'MRMLNode_scan') and self.MRMLNode_scan:
                slicer.util.setSliceViewerLayers(
                    background=self.MRMLNode_scan,
                    label=self.currentSegNode
                )
            
            # Rafraîchissement forcé
            slicer.util.forceRenderAllViews()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            qt.QMessageBox.critical(self.parent, 
                                "Erreur de chargement", 
                                f"Erreur lors de l'affichage :\n{str(e)}")


    def onCancel(self):
        if self.logic:
            self.logic.cancelRequested = True
            self.progressUpdater.logChanged.emit("Cancellation requested.")
        if hasattr(self, "processThread") and self.processThread.is_alive():
            self.processThread.join(1)
            self.ui.PredScanLabel.setText("Prediction canceled.")

    def RunningUI(self, running):
        self.ui.PredictionButton.setEnabled(not running)
        self.ui.CancelButton.setEnabled(running)
        self.ui.progressBar.setVisible(running)

    def initializeParameterNode(self):
        pass

    def load_nii_in_slicer(self, nii_file):
        if not os.path.exists(nii_file):
            raise FileNotFoundError(f"NIfTI file not found: {nii_file}")
        volume_node = slicer.util.loadVolume(nii_file)
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        slicer.util.setSliceViewerLayers(background=volume_node)

class CLICLogic(ScriptedLoadableModuleLogic):
    CLASS_NAMES = {1: "buccal", 2: "bicortical", 3: "palatal"}

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.seg_files = []  # Pour stocker les segmentations sauvegardées
        self.cancelRequested = False

    def get_model_instance_segmentation(self, num_classes):
        model = maskrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        return model

    def load_model(self, model_path, num_classes, device):
        model = self.get_model_instance_segmentation(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    def process_nii_file(self, model, nii_path, device, progress_callback=None, log_callback=None, score_threshold=0.7):
        nib_vol = nib.load(nii_path)
        vol_data = nib_vol.get_fdata(dtype=np.float32)
        H, W, Z = vol_data.shape
        if log_callback:
            log_callback("Processing slices for file: " + nii_path)
        all_detections = []
        slice_counts = {"buccal": {"left": 0, "right": 0},
                        "bicortical": {"left": 0, "right": 0},
                        "palatal": {"left": 0, "right": 0}}
        for z in range(Z):
            if self.cancelRequested:
                if log_callback:
                    log_callback("Processing cancelled during slice processing at slice %d." % z)
                break
            slice_2d = vol_data[..., z]
            slice_norm = self.normalize_slice(slice_2d)
            slice_tensor = torch.from_numpy(slice_norm).unsqueeze(0).repeat(3, 1, 1).float().to(device)
            with torch.no_grad():
                preds = model([slice_tensor])[0]
            scores = preds["scores"]
            masks = preds.get("masks", None)
            labels = preds["labels"]
            keep = scores >= score_threshold
            scores = scores[keep]
            labels = labels[keep]
            if masks is not None:
                masks = masks[keep]
            if masks is None or len(scores) == 0:
                if progress_callback:
                    progress = ((z + 1) / Z) * 100
                    progress_callback(progress)
                continue
            labels_np = labels.cpu().numpy()
            masks_np = (masks > 0.5).squeeze(1).cpu().numpy()
            for i in range(len(masks_np)):
                l = int(labels_np[i])
                mk = masks_np[i]
                com = scipy.ndimage.center_of_mass(mk)
                side = "left" if com[0] < (H/2) else "right"
                slice_counts[self.CLASS_NAMES[l]][side] += 1
                all_detections.append({
                    "label": l,
                    "slice_z": z,
                    "mask_2d": mk
                })
            if progress_callback:
                progress = ((z + 1) / Z) * 100
                progress_callback(progress)
        return vol_data, nib_vol, all_detections, slice_counts

    def normalize_slice(self, slice_2d):
        mn, mx = slice_2d.min(), slice_2d.max()
        if (mx - mn) > 1e-8:
            return (slice_2d - mn) / (mx - mn)
        else:
            return np.zeros_like(slice_2d, dtype=np.float32)

    def save_nii(self, vol_np, nib_ref, out_path):
        nii_img = nib.Nifti1Image(vol_np.astype(np.int16), nib_ref.affine, nib_ref.header)
        nib.save(nii_img, out_path)

    def process(self, parameters, progress_callback=None, log_callback=None, display_callback=None):
        try:
            self.cancelRequested = False
            input_path = parameters["input_path"]
            model_folder = parameters["model_folder"]
            output_dir = parameters["output_dir"]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_files = glob.glob(os.path.join(model_folder, "*.pth"))
            if not model_files:
                raise FileNotFoundError("No .pth files found in the specified model folder.")
            model_path = model_files[0]
            num_classes = 4
            model = self.load_model(model_path, num_classes, device)
            if os.path.isfile(input_path) and (input_path.endswith(".nii") or input_path.endswith(".nii.gz")):
                file_list = [input_path]
            elif os.path.isdir(input_path):
                file_list = glob.glob(os.path.join(input_path, "*.nii")) + glob.glob(os.path.join(input_path, "*.nii.gz"))
            else:
                raise ValueError("Invalid input path. Please provide a valid .nii/.nii.gz file or folder.")
            seg_files = []
            for nii_file in file_list:
                if self.cancelRequested:
                    if log_callback:
                        log_callback("Processing cancelled before file: " + nii_file)
                    break
                if log_callback:
                    log_callback("Processing file: " + nii_file)
                vol_data, nib_ref, detections, counts = self.process_nii_file(model, nii_file, device, progress_callback, log_callback)
                if log_callback:
                    log_callback("Finished processing file: " + nii_file)
                    log_callback("Label to class mapping:")
                    for label, name in self.CLASS_NAMES.items():
                        log_callback(f"  Label {label}: {name}")
                    log_callback("Counts:")
                    for cls in counts:
                        log_callback(f"  {cls}: Left = {counts[cls]['left']}, Right = {counts[cls]['right']}")

                seg_data = np.zeros(vol_data.shape, dtype=np.int16)
                for det in detections:
                    z = det["slice_z"]
                    mask = det["mask_2d"]
                    label = det["label"]
                    seg_data[..., z][mask] = label

                base_name = os.path.basename(nii_file).replace(".nii.gz", "").replace(".nii", "")
                scan_output_folder = os.path.join(output_dir, base_name)
                os.makedirs(scan_output_folder, exist_ok=True)
                self._embed_visual_legend(seg_data)

                output_seg_path = os.path.join(scan_output_folder, f"{base_name}_seg.nii.gz")
                output_summary_path = os.path.join(scan_output_folder, f"{base_name}_summary.txt")

                # Sauvegarde avec métadonnées dans le header NIfTI
                nii_img = nib.Nifti1Image(seg_data.astype(np.int16), nib_ref.affine, nib_ref.header)
                nii_img.header['descrip'] = str(self.CLASS_NAMES)  # Ajout du mapping dans les métadonnées
                nib.save(nii_img, output_seg_path)

                # Sauvegarde du fichier texte avec la légende
                with open(output_summary_path, "w") as summary_file:
                    summary_file.write("Label to Class Mapping:\n")
                    for label, name in self.CLASS_NAMES.items():
                        summary_file.write(f"Label {label}: {name}\n")
                    summary_file.write("\nCounts:\n")
                    for cls in counts:
                        summary_file.write(f"{cls}: Left = {counts[cls]['left']}, Right = {counts[cls]['right']}\n")

                seg_files.append(output_seg_path)
                if seg_files and display_callback:
                    display_callback(seg_files[-1])

            if log_callback:
                log_callback("Processing completed successfully.")
            self.seg_files = seg_files
        except Exception as e:
            if log_callback:
                log_callback(f"Error during processing: {str(e)}")
            raise
    def _embed_visual_legend(self, volume_data):
        """Ajoute une légende horizontale en bas (gauche ➝ droite) correctement orientée."""
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        h, w, d = volume_data.shape
        legend_height = 60

        print(f"[DEBUG] volume shape = (h={h}, w={w}, d={d})")

        # Police lisible
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        except:
            font = ImageFont.load_default()

        for z in range(d):
            print(f"[DEBUG] Processing slice z={z}")

            # Créer image PIL horizontale
            legend_img = Image.new("L", (w, legend_height), color=0)
            draw = ImageDraw.Draw(legend_img)

            spacing = 200
            box_size = 25
            start_x = 30
            y = 15

            for i, (label, name) in enumerate(self.CLASS_NAMES.items()):
                x = start_x + i * spacing
                draw.rectangle([x, y, x + box_size, y + box_size], fill=int(label))
                draw.text((x + box_size + 10, y), name, font=font, fill=255)

            # Convertir l’image et la retourner verticalement
            legend_array = np.array(legend_img)
            legend_array_flipped = np.flipud(legend_array)  # ✅ flip vertical

            print(f"[DEBUG] legend_array_flipped shape: {legend_array_flipped.shape}")

            # Injection en bas de l'image
            y_start = h - legend_height
            y_end = h

            mask = legend_array_flipped > 0
            sub_volume = volume_data[y_start:y_end, :, z]
            sub_volume[mask] = legend_array_flipped[mask]
            volume_data[y_start:y_end, :, z] = sub_volume

            # Debug position
            indices = np.argwhere(mask)
            if indices.size > 0:
                mapped_y = y_start + indices[0][0]
                mapped_x = indices[0][1]
                print(f"[DEBUG] Will write first non-zero pixel to volume_data[{mapped_y}, {mapped_x}, {z}]")

