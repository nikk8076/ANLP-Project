"""
Configuration settings for Legal-BERT training and risk discovery
"""
from dataclasses import dataclass
from typing import Dict, Any
import torch

@dataclass
class LegalBertConfig:
    """Configuration for Legal-BERT model and training"""
    
    # Model parameters
    bert_model_name: str = "bert-base-uncased"
    num_risk_categories: int = 7  # Will be dynamically determined by risk discovery
    max_sequence_length: int = 512
    dropout_rate: float = 0.1
    
    # Training parameters
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Multi-task loss weights
    task_weights: Dict[str, float] = None
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    data_path: str = "dataset/CUAD_v1/CUAD_v1.json"
    model_save_path: str = "models/legal_bert"
    checkpoint_dir: str = "checkpoints"
    
    # Risk discovery parameters
    risk_discovery_clusters: int = 7
    tfidf_max_features: int = 10000
    tfidf_ngram_range: tuple = (1, 3)
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                'classification': 1.0,
                'severity': 0.5,
                'importance': 0.5
            }

# Global configuration instance
config = LegalBertConfig()