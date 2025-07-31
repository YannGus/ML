import torch

class Config:
    DATASET: str = 'FASHION_MNIST'
    BATCH_SIZE: int = 1024
    NUM_WORKERS: int = 0

    USE_SUBSET: bool = True
    TRAIN_SUBSET_SIZE: int = 20000
    VAL_SUBSET_SIZE: int = 4000

    INPUT_SIZE: int = 28 * 28
    HIDDEN_SIZE: int = 256
    NUM_CLASSES: int = 10

    TRADITIONAL_EPOCHS: int = 100
    PRETRAINING_EPOCHS: int = 10
    JUMP_EPOCHS: int = 1
    FINE_TUNING_EPOCHS: int = 5

    INITIAL_LR: float = 0.05
    JUMP_LR_MULTIPLIER: float = 2.0
    FINE_TUNE_LR: float = 0.001

    PREDICTOR_HIDDEN_SIZE: int = 32
    PREDICTOR_EPOCHS: int = 20
    PREDICTOR_LR: float = 0.001

    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    RESULTS_DIR: str = 'outputs'
    MODELS_DIR: str = 'outputs/models'
    PLOTS_DIR: str = 'outputs/plots'

    NUM_RUNS: int = 3
    RANDOM_SEED: int = 42
    REUSE_TRADITIONAL_MODEL: bool = True
    TRADITIONAL_MODEL_PATH: str = 'outputs/models/traditional_model.pth'
