"""Architecture, training and data constants shared by fine-tuning, DAPT and checkpoints.

Moved out of ``finetuning.py`` (Phase 6). ``finetuning.py`` re-exports every name.
"""

DIM = 128
NUM_LAYERS = 6
NUM_HEADS = 4
MLP_DIM = 512
MAX_LEN_SPATIAL = 500
NUM_GRADIENTS = 10
DELTA = 0.5
MAX_LEN_TEMP = 5000
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
TIME_STEPS = 200
ROIs = 450
