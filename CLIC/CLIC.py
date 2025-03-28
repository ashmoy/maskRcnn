import os
import slicer
import qt
import webbrowser
import ctk
from slicer.ScriptedLoadableModule import *
from CLICLib.logic import process_nii_file, load_model, save_nii  # Vos fonctions spécifiques
from CLICLib.ui_CLIC import Ui_CLIC  # Assurez-vous que le nom du fichier correspond
import torch
import subprocess
import time
from CondaSetUp import CondaSetUpCall,CondaSetUpCallWsl
import threading
from slicer.util import VTKObservationMixin
import logging
import requests
import glob
import zipfile

def PathFromNode(node):
  storageNode=node.GetStorageNode()
  if storageNode is not None:
    filepath=storageNode.GetFullNameFromFileName()
  else:
    filepath=None
  return filepath


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
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        self.CBCT_as_input = True  # True: CBCT image, False: surface IOS
        self.folder_as_input = False  # If use a folder as input
        self.MRMLNode_scan = None  # MRML node of the selected scan
        self.input_path = None  # Path to the folder containing the scans
        self.model_folder = None  # Path to the folder containing the models
        self.output_folder = None  # If save the output in a folder
        self.goup_output_files = False
        self.scan_count = 0  # Number of scans in the input folder

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        self.conda_wsl = CondaSetUpCallWsl()
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/CLIC.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = CLICLogic()

        # Connexions des signaux

        self.ui.ExtensioncomboBox.currentIndexChanged.connect(self.SwitchInputExtension)
        self.SwitchInputExtension(0)


        # self.ui.InputTypeComboBox.currentIndexChanged.connect(self.SwitchInput)
        # self.SwitchInput(0)
        
        self.ui.DownloadModelPushButton.clicked.connect(self.onModelDownloadButton)
        self.ui.SearchScanFolder.clicked.connect(self.onSearchScanButton)
        self.ui.SearchModelFolder.clicked.connect(self.onSearchModelButton)
        self.ui.SearchSaveFolder.clicked.connect(self.onSearchSaveButton)
        self.ui.PredictionButton.clicked.connect(self.onPredictButton)
        self.ui.CancelButton.clicked.connect(self.onCancel)
        self.ui.SavePredictCheckBox.toggled.connect(self.UpdateSaveType)

        # Initialisez les états de l'interface utilisateur
        self.ui.SearchSaveFolder.setHidden(False)
        self.ui.SaveFolderLineEdit.setHidden(False)
        self.ui.PredictFolderLabel.setHidden(False)

        # Désactivez l'interface utilisateur pendant le traitement
        self.RunningUI(False)

        # Initialisez le nœud de paramètres
        self.initializeParameterNode()

    def SwitchInputType(self, index):
        if index == 1:
            self.CBCT_as_input = False
            self.ui.MRMLNodeComboBox.nodeTypes = ['vtkMRMLModelNode']
        else:
            self.CBCT_as_input = True
            self.ui.MRMLNodeComboBox.nodeTypes = ['vtkMRMLVolumeNode']

    def SwitchInputExtension(self, index):
        if index == 1:  # DICOM Files
            self.ui.ScanPathLabel.setText('DICOM Folder')
        else:
            self.ui.ScanPathLabel.setText('Scan File/Folder')

    # def SwitchInput(self, index):
    #     """
    #     Switch between file and folder input types.
    #     :param index: 0 for file input, 1 for folder input
    #     """
    #     if index == 1:  # Folder as input
    #         self.folder_as_input = True
    #         self.input_path = None

    #         # Show folder-related widgets
    #         self.ui.ScanPathLabel.setVisible(True)
    #         self.ui.lineEditScanPath.setVisible(True)
    #         self.ui.SearchScanFolder.setVisible(True)

    #         # Hide node-related widgets
    #         self.ui.SelectNodeLabel.setVisible(False)
    #         self.ui.MRMLNodeComboBox.setVisible(False)

    #     else:  # File as input
    #         self.folder_as_input = False

    #         # Hide folder-related widgets
    #         self.ui.ScanPathLabel.setVisible(False)
    #         self.ui.lineEditScanPath.setVisible(False)
    #         self.ui.SearchScanFolder.setVisible(False)

    #         # Show node-related widgets
    #         self.ui.SelectNodeLabel.setVisible(True)
    #         self.ui.MRMLNodeComboBox.setVisible(True)

    #     # Reset the input path or node selection
    #     self.onNodeChanged()

    def onNodeChanged(self):
        self.MRMLNode_scan = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox.currentNodeID)
        if self.MRMLNode_scan:
            self.input_path = PathFromNode(self.MRMLNode_scan)
            self.scan_count = 1
            self.ui.PrePredInfo.setText("Number of scans to process: 1")
            return True
        return False

    def onTestDownloadButton(self):
        webbrowser.open("https://example.com/test-scan")  # Remplacez par votre lien de téléchargement

    def onModelDownloadButton(self):
        """
        Télécharge directement le fichier .pth contenant le modèle dans le dossier spécifié.
        """
        # URL du fichier .pth contenant le modèle
        model_url = "https://github.com/ashmoy/maskRcnn/releases/download/model/final_model.pth"
        
        # Dossier de destination pour les modèles
        default_model_folder = os.path.join(os.path.expanduser("~"), "Documents", "CLIC_Models")
        os.makedirs(default_model_folder, exist_ok=True)  # Crée le dossier s'il n'existe pas

        # Chemin du fichier .pth téléchargé
        model_path = os.path.join(default_model_folder, "final_model.pth")

        try:
            # Étape 1 : Télécharger le fichier .pth
            self.ui.PredScanLabel.setText("Downloading model...")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()  # Vérifie si le téléchargement a réussi

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            chunk_size = 1024  # Taille du bloc de téléchargement

            with open(model_path, 'wb') as file:
                for data in response.iter_content(chunk_size=chunk_size):
                    file.write(data)
                    downloaded_size += len(data)
                    progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                    self.ui.PredScanProgressBar.setValue(int(progress))  # Met à jour la barre de progression

            # Mettre à jour l'interface utilisateur
            self.ui.lineEditModelPath.setText(default_model_folder)
            self.model_folder = default_model_folder
            self.ui.PredScanLabel.setText("Model downloaded successfully!")
            qt.QMessageBox.information(self.parent, 'Success', 'Model downloaded successfully!')

        except Exception as e:
            # Gérer les erreurs
            qt.QMessageBox.warning(self.parent, 'Error', f"Failed to download model: {str(e)}")
            self.ui.PredScanLabel.setText("Error during download.")

    def onSearchScanButton(self):
        """
        Open a file dialog to select a scan file or folder.
        """

        # Open a folder dialog
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

    def onPredictButton(self):
        if not self.input_path:
            qt.QMessageBox.warning(self.parent, 'Warning', 'Please select an input file/folder')
            return
        if not self.model_folder:
            qt.QMessageBox.warning(self.parent, 'Warning', 'Please select a model folder')
            return

        param = {}
        param["input_path"] = self.input_path
        param["model_folder"] = self.model_folder
        param["output_dir"] = self.output_folder
        param["save_in_folder"] = self.goup_output_files
            # Activer l'interface utilisateur pour indiquer que le traitement est en cours
        self.RunningUI(True)
        process = threading.Thread(target=self.logic.process, args=(param,))
        process.start()
        start_time = time.time()
        while process.is_alive():
            slicer.app.processEvents()
            current_time = time.time()
            self.ui.TimerLabel.setText(f"Time elapsed: {current_time - start_time:.2f}s")

    def onCancel(self):
        if hasattr(self, "process") and self.process.is_alive():
            self.process.terminate()
            self.ui.PredScanLabel.setText("Prediction canceled.")

    def RunningUI(self, running):
        self.ui.PredictionButton.setEnabled(not running)
        self.ui.CancelButton.setEnabled(running)
        self.ui.progressBar.setVisible(running)

    def initializeParameterNode(self):
        pass

    def onSceneStartClose(self, caller, event):
        pass

    def onSceneEndClose(self, caller, event):
        pass

    def UpdateSaveType(self, checked):
        self.goup_output_files = checked
        self.ui.SearchSaveFolder.setHidden(checked)
        self.ui.SaveFolderLineEdit.setHidden( checked)
        self.ui.PredictFolderLabel.setHidden( checked)

class CLICLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    def process(self, parameters):
        logging.info('Processing started')
        try:
            # Récupérer les paramètres
            input_path = parameters["input_path"]
            model_folder = parameters["model_folder"]
            output_dir = parameters["output_dir"]
            save_in_folder = parameters["save_in_folder"]

            # Rechercher les fichiers .pth dans le dossier spécifié
            model_files = glob.glob(os.path.join(model_folder, "*.pth"))
            if not model_files:
                raise FileNotFoundError("No .pth files found in the specified model folder.")

            # Charger le premier fichier modèle trouvé
            model_path = model_files[0]
            logging.info(f"Loading model from: {model_path}")

            # Charger le modèle
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_classes = 4  # Nombre de classes (y compris le background)
            model = load_model(model_path, num_classes, device)

            # Traiter tous les fichiers .nii ou .nii.gz dans le dossier d'entrée
            if os.path.isfile(input_path) and (input_path.endswith(".nii") or input_path.endswith(".nii.gz")):
                file_list = [input_path]
            elif os.path.isdir(input_path):
                file_list = glob.glob(os.path.join(input_path, "*.nii")) + glob.glob(os.path.join(input_path, "*.nii.gz"))
            else:
                raise ValueError("Invalid input path. Please provide a valid .nii/.nii.gz file or folder.")

            # Traiter chaque fichier
            total_files = len(file_list)
            for i, nii_file in enumerate(file_list):
                logging.info(f"Processing file {i + 1}/{total_files}: {nii_file}")
                vol_data, nib_ref, detections = process_nii_file(model, nii_file, device)

                # Sauvegarder les résultats
                if save_in_folder:
                    output_path = os.path.join(output_dir, os.path.basename(nii_file).replace(".nii", "_seg.nii").replace(".nii.gz", "_seg.nii.gz"))
                else:
                    output_path = os.path.join(os.path.dirname(nii_file), os.path.basename(nii_file).replace(".nii", "_seg.nii").replace(".nii.gz", "_seg.nii.gz"))

                save_nii(vol_data, nib_ref, output_path)
                logging.info(f"Segmentation completed for {nii_file}. Output saved to {output_path}")

            logging.info('All files processed successfully.')
            qt.QMessageBox.information(None, 'Success', f'Segmentation completed for all {total_files} files.')

        except Exception as e:
            logging.error(f"Error during processing: {str(e)}")
            qt.QMessageBox.warning(None, 'Error', f"An error occurred during segmentation: {str(e)}")