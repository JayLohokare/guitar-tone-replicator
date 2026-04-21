"""Core package for Tone Replicator."""
try:
    from .model import ToneNet, LSTMToneNet, create_model
    from .dataset import ToneDataset, ReferenceDIProvider
    from .trainer import ToneTrainer, ESR, MRSTFTLoss
except ImportError:
    from model import ToneNet, LSTMToneNet, create_model
    from dataset import ToneDataset, ReferenceDIProvider
    from trainer import ToneTrainer, ESR, MRSTFTLoss

__all__ = [
    "ToneNet", "LSTMToneNet", "create_model",
    "ToneDataset", "ReferenceDIProvider",
    "ToneTrainer", "ESR", "MRSTFTLoss",
]