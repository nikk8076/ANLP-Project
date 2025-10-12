"""
Legal-BERT Model Architecture - Fully Learning-Based
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Any, Optional

class FullyLearningBasedLegalBERT(nn.Module):
    """
    Legal-BERT model that learns from discovered risk patterns.
    NO hardcoded risk categories!
    """
    
    def __init__(self, config, num_discovered_risks: int = 7):
        super().__init__()
        self.config = config
        self.num_discovered_risks = num_discovered_risks
        
        # Load BERT model
        try:
            self.bert = AutoModel.from_pretrained(config.bert_model_name)
            # Configure BERT dropout
            self.bert.config.hidden_dropout_prob = config.dropout_rate
            self.bert.config.attention_probs_dropout_prob = config.dropout_rate
        except:
            # Fallback for testing without transformers
            print("Warning: Using mock BERT model (transformers not available)")
            self.bert = None
        
        # Multi-task heads
        hidden_size = 768  # BERT-base hidden size
        
        # Risk classification head (for discovered risk patterns)
        self.risk_classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size // 2, num_discovered_risks)
        )
        
        # Severity regression head (0-10 scale)
        self.severity_regressor = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # Output between 0-1, will be scaled to 0-10
        )
        
        # Importance regression head (0-10 scale)
        self.importance_regressor = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # Output between 0-1, will be scaled to 0-10
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        
        if self.bert is not None:
            # Real BERT forward pass
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
        else:
            # Mock output for testing
            batch_size = input_ids.size(0)
            pooled_output = torch.randn(batch_size, 768)
            if input_ids.is_cuda:
                pooled_output = pooled_output.cuda()
        
        # Multi-task predictions
        risk_logits = self.risk_classifier(pooled_output)
        severity_score = self.severity_regressor(pooled_output).squeeze(-1) * 10  # Scale to 0-10
        importance_score = self.importance_regressor(pooled_output).squeeze(-1) * 10  # Scale to 0-10
        
        # Apply temperature scaling to classification logits
        calibrated_logits = risk_logits / self.temperature
        
        return {
            'risk_logits': risk_logits,
            'calibrated_logits': calibrated_logits,
            'severity_score': severity_score,
            'importance_score': importance_score,
            'pooled_output': pooled_output
        }
    
    def predict_risk_pattern(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
        """Make predictions and return interpretable results"""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            # Get predictions
            risk_probs = torch.softmax(outputs['calibrated_logits'], dim=-1)
            predicted_risk = torch.argmax(risk_probs, dim=-1)
            confidence = torch.max(risk_probs, dim=-1)[0]
            
            return {
                'predicted_risk_id': predicted_risk.cpu().numpy(),
                'risk_probabilities': risk_probs.cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'severity_score': outputs['severity_score'].cpu().numpy(),
                'importance_score': outputs['importance_score'].cpu().numpy()
            }

class LegalBertTokenizer:
    """Tokenizer wrapper for Legal-BERT"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            print("Warning: Using mock tokenizer (transformers not available)")
            self.tokenizer = None
    
    def tokenize_clauses(self, clauses: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Tokenize legal clauses for model input"""
        
        if self.tokenizer is None:
            # Mock tokenization for testing
            batch_size = len(clauses)
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, max_length)),
                'attention_mask': torch.ones(batch_size, max_length)
            }
        
        # Real tokenization
        encoded = self.tokenizer(
            clauses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def decode_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs back to text"""
        if self.tokenizer is None:
            return ["Mock decoded text"] * token_ids.size(0)
        
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)