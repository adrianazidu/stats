import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets
from PIL import Image
import cv2
from ultralytics import YOLO
import easyocr
from pathlib import Path

#datasets used
#https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder for car recognition
#https://www.kaggle.com/datasets/fareselmenshawii/license-plate-dataset for YOLO car plates recognition
#kaggle notebook for training
#https://www.kaggle.com/code/adrianazidu/resnet-yolo-ocr-training

# ==============================================================================
# 0. CONFIGURATION & AUTO-DOWNLOAD MODEL SETUP
# ==============================================================================

# Detect whether the script is running inside a Kaggle Notebook

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if IS_KAGGLE:
    # Read-only input datasets attached in the Kaggle sidebar
    CARS_DATA_DIR = Path("/kaggle/input/datasets/jutrera/stanford-car-dataset-by-classes-folder/car_data/car_data")
    PLATES_YAML = prepare_plates_yaml()
    OCR_DATA_DIR = Path("/kaggle/input/your-ocr-dataset")
    
    # Writable output directory so you can download trained weights afterward
    WEIGHTS_DIR = Path("/kaggle/working/saved_models")
else:
    # Local machine paths
    CARS_DATA_DIR = Path("./data/cardata/cartypes/car_data/car_data")
    PLATES_YAML = Path("./data/plates/dataset.yaml")
    OCR_DATA_DIR = Path("./data/ocr")
    WEIGHTS_DIR = Path("./saved_models")

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Character vocabulary for License Plate OCR (Index 0 is reserved for CTC Blank)
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(ALPHABET)}
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(ALPHABET)}
NUM_CLASSES_OCR = len(ALPHABET) + 1  # +1 for CTC blank token

# Initialize once at startup (downloads lightweight PP-OCR weights automatically)
ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

RESNET_WEIGHTS = "resnet_car_make.pth"
YOLO_WEIGHTS = "yolo_plate_best.pt"

"""
.pt stands for PyTorch (preferred by Ultralytics/YOLOv8 and the official PyTorch documentation, mainly because .pth already existed in Python for site-packages path configuration files).
.pth stands for PyTorch History/checkpoint (a long-standing community convention in torchvision and research codebases).
"""
# ==============================================================================
# STAGE 1: CAR MAKE CLASSIFIER (RESNET50 FEATURE EXTRACTION), saves model into resnet_car_make.pth
# ==============================================================================
#convert the Path to string bwfore passing to the function
def train_car_classifier(data_dir=str(CARS_DATA_DIR), epochs=5, batch_size=32):
    print("\n--- [Stage 1] Training Car Make Classifier (ResNet50) ---")
    save_path = os.path.join(WEIGHTS_DIR, RESNET_WEIGHTS)

    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #create a pytorch dataset by scanning the train folder
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)

    #batching engine-32 images simultaneously, runs transforms, stacks them in a 4D tensor with a 1D tensor of lables
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #extract the class names
    class_names = train_dataset.classes

    #download resnet weights to ~/.cache/torch/hub/checkpoints if not present
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze all convolutional layers (Feature Extraction mode),prevent weights update, stop gradient tracking, intermediate tensors are not saved
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer for our specific car types
    #the last decision layer takes the numbers generated for each image and turns them into 1000 scores (categories)

    #read how many numbers come out of resnet body (except last layer)
    #for resnet 50 is 2048, and fro resnet 18 is 512 but we don't want to hardcode it in case we change the model
    in_features = model.fc.in_features

    #takes off the original Linear(2048 -> 1000) layer and replaces it with a brand-new, untrained layer Linear(2048 -> len(class_names))
    #so it outputs exactly the number of our categories
    #this is the only layer with requires_grad = True (the default on layer creation)
    model.fc = nn.Linear(in_features, len(class_names))

    #move entire model to CPU/GPU (including both body (resnet intiial) and our added layer(linear)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # Optimize ONLY the newly added head parameters
    #model.fc.parameters() contains ONLY the new layer's weight matrix and bias vector,the old ResNet fc layer was completely overwritten and discarded
    #inside the model we now have model.parameters() = all frozen convolutional layers (conv1, layer1–layer4, etc.) + the new model.fc layer
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    print("start training")

    model.train() #entry in train mode
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        running_loss = 0.0
        for images, labels in train_loader:

            #move images and labels to the target hardware device (GPU/CPU) so the model can process them
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images) #predict
            loss = criterion(outputs, labels) #confront output with target
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

    # Save weights alongside class names so inference knows index -> brand mapping
    torch.save({"model_state": model.state_dict(), "classes": class_names}, save_path)

    print(f"Saved ResNet car classifier to {save_path}")
    return save_path


# ==============================================================================
# STAGE 2: LICENSE PLATE DETECTOR (YOLOv8), scan the image find the bounding box (x1 y1 x2 y2) of the license plate and crops it out
# ==============================================================================
def train_plate_detector(data_yaml=str(PLATES_YAML), epochs=20):
    print("\n--- [Stage 2] Training License Plate Detector (YOLOv8) ---")

    #ultanalytics checks if this weights path exists, if ot's missing it automatically dowbloads the base COCO pretrained yolov8n.pt weights from github
    base_weights = os.path.join(WEIGHTS_DIR, "yolov8n.pt")
    trained_save_path = os.path.join(WEIGHTS_DIR, YOLO_WEIGHTS)

    # Ultralytics automatically downloads 'yolov8n.pt' if it does not exist locally
    model = YOLO(base_weights)

    # Train on your custom license plate bounding-box dataset
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        project="runs_plate",
        name="exp",
        exist_ok=True
    )

    # Copy the best checkpoint into our unified weights folder
    best_pt = os.path.join("runs_plate", "exp", "weights", "best.pt")
    if os.path.exists(best_pt):
        os.replace(best_pt, trained_save_path)
    print(f"Saved YOLO plate detector to {trained_save_path}")
    return trained_save_path


# ==============================================================================
# STAGE 3: Takes the small cropped plate rectangle and reads the exact letters and numbers off it.
# ==============================================================================
def read_plate_paddleocr(cropped_plate_pil) -> str:
    plate_np = np.array(cropped_plate_pil)
    results = paddle_ocr.ocr(plate_np, cls=True)
    
    if not results or not results[0]:
        return "Unreadable"
    
    # Extract detected text strings in reading order
    texts = [line[1][0] for line in results[0]]
    return " ".join(texts).strip()


# ==============================================================================
# 4. END-TO-END INFERENCE ON A NEW IMAGE
# ==============================================================================
def predict_on_new_image(image_path:str, ocr_engine:PaddleOCR):
    """Runs car brand classification, plate detection, and PaddleOCR reading on a single image."""
    print(f"\n--- Running End-to-End Prediction on: {image_path} ---")
    raw_image = Image.open(image_path).convert("RGB")

    # 1. Predict Car Make (ResNet-50)
    checkpoint = torch.load(WEIGHTS_DIR / RESNET_WEIGHTS, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(class_names))
    resnet.load_state_dict(checkpoint["model_state_dict"])
    resnet = resnet.to(DEVICE).eval()

    cls_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensor = cls_transform(raw_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(resnet(input_tensor), dim=1)
        conf, pred_idx = torch.max(probs, 1)
        car_brand = class_names[pred_idx.item()]

    # 2. Detect & Crop License Plate (YOLOv8)
    yolo_weights = WEIGHTS_DIR / YOLO_WEIGHTS
    yolo_model = YOLO(str(yolo_weights))
    results = yolo_model.predict(source=raw_image, conf=0.25, verbose=False)

    plate_text = "No plate detected"
    boxes = results[0].boxes
    if len(boxes) > 0:
        # Take highest-confidence bounding box
        x1, y1, x2, y2 = map(int, boxes.xyxy[0].tolist())
        cropped_plate = raw_image.crop((x1, y1, x2, y2))

        # 3. Read Characters from Cropped Plate (PaddleOCR)
        plate_np = np.array(cropped_plate)
        ocr_result = ocr_engine.predict(plate_np)

        # Extract recognized text strings (PaddleOCR v3+ result format)
        extracted_lines = []
        for res in ocr_result:
            if "rec_texts" in res and res["rec_texts"]:
                extracted_lines.extend(res["rec_texts"])

        plate_text = " ".join(extracted_lines).strip() if extracted_lines else "Unreadable"

    print(f"Result -> Car Brand: {car_brand} ({conf.item():.1%}) | Plate: {plate_text}")
    return {"car_brand": car_brand, "confidence": conf.item(), "plate_number": plate_text}

def save_pipeline():
    import mlflow

    mlflow.set_experiment("license_plate_pipeline")

    with mlflow.start_run(run_name="yolov8n_plate_v1.0.0"):
        # 1. Log hyperparameters directly from Ultralytics' args.yaml
        mlflow.log_artifact("runs/v1.0.0/args.yaml", artifact_path="config")
        
        # 2. Log evaluation metrics & plots
        mlflow.log_artifact("runs/v1.0.0/results.csv", artifact_path="metrics")
        mlflow.log_artifact("runs/v1.0.0/confusion_matrix.png", artifact_path="plots")
        
        # 3. Log and register the versioned model weight file
        mlflow.log_artifact("weights/yolo_plate_best.pt", artifact_path="weights")
        mlflow.set_tags({
            "model_stage": "plate_detector",
            "base_architecture": "yolov8n",
            "dataset": "fareselmenshawii/license-plate-dataset",
            "status": "staging",
        })

if __name__ == "__main__":
    # Install requirements first:
    # uv pip install torch torchvision ultralytics opencv-python pillow

    # 1. Train the 2 models on kaggle/colab
    #w_resnet = train_car_classifier()
    #w_yolo   = train_plate_detector()
    save_pipeline()



    import sys
    sys.exit()

    #  Run End-to-End Inference on a New Image (no OCR training needed)
    test_img = Path("./test_car.jpg")
    resnet_ready = (WEIGHTS_DIR / "car_classifier_resnet50.pth").exists()
    yolo_ready = (WEIGHTS_DIR / "plate_yolo/weights/best.pt").exists()

    if test_img.exists() and resnet_ready and yolo_ready:
        # Initialize PaddleOCR once (downloads lightweight PP-OCR weights automatically)
        paddle_ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, lang="en")
        predict_on_new_image(str(test_img), ocr_engine=paddle_ocr)