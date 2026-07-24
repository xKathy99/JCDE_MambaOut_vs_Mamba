import os

# 1. DATASET PATHS (CHANGE THESE TO YOUR LOCAL PATHS)
DATA_ROOT = "..path/to/datasets"  

# ACDC dataset
ACDC_TRAIN_IMAGES = os.path.join(DATA_ROOT, "acdc/training/images")
ACDC_TRAIN_MASKS  = os.path.join(DATA_ROOT, "acdc/training/masks")
ACDC_TEST_IMAGES  = os.path.join(DATA_ROOT, "acdc/testing/images")
ACDC_TEST_MASKS   = os.path.join(DATA_ROOT, "acdc/testing/masks")

# M&Ms dataset
MMS_TEST_IMAGES   = os.path.join(DATA_ROOT, "mms/testing/images")
MMS_TEST_MASKS    = os.path.join(DATA_ROOT, "mms/testing/masks")

# M&Ms-2 dataset
MMS2_TEST_IMAGES  = os.path.join(DATA_ROOT, "mm2/testing/images")
MMS2_TEST_MASKS   = os.path.join(DATA_ROOT, "mm2/testing/masks")

# 2. TRAINING HYPERPARAMS
IMG_SIZE        = 224
BATCH_SIZE      = 4
NUM_EPOCHS      = 80
LEARNING_RATE   = 0.001
UNFREEZE_EPOCH  = 12    
PATIENCE        = 5     
NUM_CLASSES     = 4     

#  Focal Loss CLASS WEIGHTS
ALPHA = [0.1, 0.3, 0.3, 0.3]

# 3. OUTPUT DIRS (weights, logs, etc.)
WEIGHTS_DIR       = "../checkpoints"
BEST_WEIGHTS_DIR  = "../checkpoints/best"

# Model selection
MODEL_ARCH = "pretrainedmambaout"  
# options:
# pretrainedmamba, hybridmamba, hybridmambaout
# puremamba, puremambaout
