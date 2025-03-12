from pathlib import Path
import os 

''' Main Application Constants. '''

APPLICATION_PATH = Path(__file__).resolve().parent

''' DIRECTORY PATH CONSTANTS. '''

ASSETS_DIR = './assets/'
ICONS_DIR_PATH = os.path.join(APPLICATION_PATH, ASSETS_DIR)
CAPTURE_DIR = './captures/'
CAPTURES_DIR_PATH = os.path.join(APPLICATION_PATH, CAPTURE_DIR)

''' MODELS FOR INFERENCE. '''

# Model Dir Path.
MODELS_PATH = 'detection_models/'
# Available models.
# Lowest performance, better if application becomes more constrained or cant leverage hardware acceleration.
YOLO_V11S = os.path.join(MODELS_PATH, 'yolo11s.pt')
# Self trained model to detect license plates. 
PLATE_TRAINED_YOLO_V8 = os.path.join(MODELS_PATH, 'plate_detection.pt')
# Set path to run inference on.
DETECTION_MODEL_PATH = os.path.join(APPLICATION_PATH, YOLO_V11S)
# Path for self trained plate detection model. 
PLATE_DETECTION_MODEL_PATH = os.path.join(APPLICATION_PATH, PLATE_TRAINED_YOLO_V8)

# One size fits all confidence threshold before adjustment. 
BASE_YOLO_CONFIDENCE_THRESHOLD = 0.85
PLATE_YOLO_CONFIDENCE_THRESHOLD = 0.60

CLASSES_OF_INTEREST = {
    'car' : {
        'width' : 1.821, 'height' : 1.534
    },
    'motorcycle' : {
        'width' : 0.995, 'height' : 2.190
    },
    'bus' : {
        'width' :  2.560, 'height' : 4.200
    },
    'truck' : {
        'width' :  2.400, 'height' : 2.590
    },
    'License_Plate' : {
        'width': 0.52, 'height': 0.11
    }
}