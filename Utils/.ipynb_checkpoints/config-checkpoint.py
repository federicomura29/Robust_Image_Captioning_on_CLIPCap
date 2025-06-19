import os
import torch

# -----------------------------------------
# configuration
# -----------------------------------------

EVAL = True  # Set to True for evaluation mode


# -----------------------------------------
# Paths
# -----------------------------------------

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
DATA_VAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/coco/val2014")
DATA_ANN_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/coco/annotations")
DATA_EVAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval_images")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Outputs")
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models")
PRETRAINED_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pretrained_models/model_wieghts.pt")

# -----------------------------------------
# Parameters
# -----------------------------------------

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
