from .poolers import AveragePool, GatedPool, SoftAttentionPool, AttentionPool
from .networks import Classifier, Regressor
from .dataset import (
    EmbeddingDataset,
    collate_classification_stack,
    collate_regression,
    get_class_weights,
    get_pos_weights,
)
from .train import train_classifier, train_classifier_stack
from .inference import get_predictions, get_predictions_stack
from .plots import (
    plot_train_curves,
    plot_confusion_matrix,
    plot_patch_similarity,
    plot_feature_map,
    view_object,
)
