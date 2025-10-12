"""
Legal-BERT Training Pipeline - Learning-Based Risk Classification
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any
import os
from sklearn.metrics import accuracy_score, classification_report
import json

from config import LegalBertConfig
from model import FullyLearningBasedLegalBERT, LegalBertTokenizer
from risk_discovery import UnsupervisedRiskDiscovery
from data_loader import CUADDataLoader

class LegalClauseDataset(Dataset):
    """Dataset for legal clauses with discovered risk labels"""
    
    def __init__(self, clauses: List[str], risk_labels: List[int], 
                severity_scores: List[float], importance_scores: List[float],
                tokenizer: LegalBertTokenizer, max_length: int = 512):
        self.clauses = clauses
        self.risk_labels = risk_labels
        self.severity_scores = severity_scores
        self.importance_scores = importance_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.clauses)
    
    def __getitem__(self, idx):
        clause = self.clauses[idx]
        
        # Tokenize
        encoded = self.tokenizer.tokenize_clauses([clause], self.max_length)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'risk_label': torch.tensor(self.risk_labels[idx], dtype=torch.long),
            'severity_score': torch.tensor(self.severity_scores[idx], dtype=torch.float),
            'importance_score': torch.tensor(self.importance_scores[idx], dtype=torch.float)
        }

class LegalBertTrainer:
    """
    Trainer for Legal-BERT with discovered risk patterns.
    NO hardcoded risk categories!
    """
    
    def __init__(self, config: LegalBertConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        self.risk_discovery = UnsupervisedRiskDiscovery(
            n_clusters=config.risk_discovery_clusters,
            random_state=42
        )
        self.tokenizer = LegalBertTokenizer(config.bert_model_name)
        
        # Will be initialized during training
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
    
    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load data and discover risk patterns"""
        print("Preparing data with unsupervised risk discovery...")
        
        # Load CUAD data
        data_loader = CUADDataLoader(data_path)
        df_clauses, contracts = data_loader.load_data()
        splits = data_loader.create_splits()
        
        # Get training clauses for risk discovery
        train_clauses = splits['train']['clause_text'].tolist()
        
        # Discover risk patterns from training data
        discovered_patterns = self.risk_discovery.discover_risk_patterns(train_clauses)
        
        # Create datasets for each split
        datasets = {}
        dataloaders = {}
        
        for split_name, split_data in splits.items():
            clauses = split_data['clause_text'].tolist()
            
            # Get discovered risk labels
            risk_labels = self.risk_discovery.get_risk_labels(clauses)
            
            # Generate synthetic severity and importance scores
            # (In practice, these could be learned from other signals)
            severity_scores = self._generate_synthetic_scores(clauses, 'severity')
            importance_scores = self._generate_synthetic_scores(clauses, 'importance')
            
            # Create dataset
            dataset = LegalClauseDataset(
                clauses=clauses,
                risk_labels=risk_labels,
                severity_scores=severity_scores,
                importance_scores=importance_scores,
                tokenizer=self.tokenizer,
                max_length=self.config.max_sequence_length
            )
            
            datasets[split_name] = dataset
            
            # Create dataloader
            shuffle = (split_name == 'train')
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=0  # Set to 0 to avoid multiprocessing issues
            )
            dataloaders[split_name] = dataloader
        
        print(f"Data preparation complete!")
        print(f"Discovered {len(discovered_patterns)} risk patterns")
        
        return dataloaders['train'], dataloaders['val'], dataloaders['test']
    
    def _generate_synthetic_scores(self, clauses: List[str], score_type: str) -> List[float]:
        """Generate synthetic severity/importance scores based on text features"""
        scores = []
        
        for clause in clauses:
            # Extract risk features
            features = self.risk_discovery.extract_risk_features(clause)
            
            if score_type == 'severity':
                # Base severity on risk intensity and liability terms
                score = (
                    features.get('risk_intensity', 0) * 10 +
                    features.get('liability_terms_density', 0) * 5 +
                    features.get('obligation_strength', 0) * 3
                )
            else:  # importance
                # Base importance on legal complexity and clause length
                score = (
                    features.get('legal_complexity', 0) * 8 +
                    min(features.get('clause_length', 0) / 100, 1) * 2 +
                    features.get('obligation_terms_complexity', 0) * 5
                )
            
            # Normalize to 0-10 scale and add some randomness
            normalized_score = min(max(score + np.random.normal(0, 0.5), 0), 10)
            scores.append(normalized_score)
        
        return scores
    
    def setup_training(self, train_loader: DataLoader):
        """Initialize model, optimizer, and scheduler"""
        num_discovered_risks = self.risk_discovery.n_clusters
        
        # Initialize model
        self.model = FullyLearningBasedLegalBERT(
            config=self.config,
            num_discovered_risks=num_discovered_risks
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        
        print(f"Model initialized with {num_discovered_risks} discovered risk categories")
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        
        # Classification loss (discovered risk patterns)
        classification_loss = self.classification_loss(
            outputs['risk_logits'],
            batch['risk_label']
        )
        
        # Severity regression loss
        severity_loss = self.regression_loss(
            outputs['severity_score'],
            batch['severity_score']
        )
        
        # Importance regression loss
        importance_loss = self.regression_loss(
            outputs['importance_score'],
            batch['importance_score']
        )
        
        # Weighted combination
        total_loss = (
            self.config.task_weights['classification'] * classification_loss +
            self.config.task_weights['severity'] * severity_loss +
            self.config.task_weights['importance'] * importance_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'severity_loss': severity_loss,
            'importance_loss': importance_loss
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        loss_components = {'classification': 0, 'severity': 0, 'importance': 0}
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            risk_labels = batch['risk_label'].to(self.device)
            severity_scores = batch['severity_score'].to(self.device)
            importance_scores = batch['importance_score'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            
            # Prepare batch for loss computation
            batch_for_loss = {
                'risk_label': risk_labels,
                'severity_score': severity_scores,
                'importance_score': importance_scores
            }
            
            # Compute loss
            losses = self.compute_loss(outputs, batch_for_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += losses['total_loss'].item()
            
            # Classification accuracy
            predictions = torch.argmax(outputs['risk_logits'], dim=-1)
            correct_predictions += (predictions == risk_labels).sum().item()
            total_samples += risk_labels.size(0)
            
            # Loss components
            loss_components['classification'] += losses['classification_loss'].item()
            loss_components['severity'] += losses['severity_loss'].item()
            loss_components['importance'] += losses['importance_loss'].item()
            
            # Progress logging
            if batch_idx % 50 == 0:
                print(f"    Batch {batch_idx}/{len(train_loader)}, Loss: {losses['total_loss'].item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        # Average loss components
        for key in loss_components:
            loss_components[key] /= len(train_loader)
        
        return avg_loss, accuracy, loss_components
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                risk_labels = batch['risk_label'].to(self.device)
                severity_scores = batch['severity_score'].to(self.device)
                importance_scores = batch['importance_score'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Prepare batch for loss computation
                batch_for_loss = {
                    'risk_label': risk_labels,
                    'severity_score': severity_scores,
                    'importance_score': importance_scores
                }
                
                # Compute loss
                losses = self.compute_loss(outputs, batch_for_loss)
                total_loss += losses['total_loss'].item()
                
                # Classification accuracy
                predictions = torch.argmax(outputs['risk_logits'], dim=-1)
                correct_predictions += (predictions == risk_labels).sum().item()
                total_samples += risk_labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Complete training pipeline"""
        print(f"🚀 Starting Legal-BERT training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        
        self.setup_training(train_loader)
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Train
            train_loss, train_acc, loss_components = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            # Log results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Loss Components:")
            print(f"Classification: {loss_components['classification']:.4f}")
            print(f"Severity: {loss_components['severity']:.4f}")
            print(f"Importance: {loss_components['importance']:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch)
        
        print(f"Training complete!")
        return self.training_history
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'config': self.config,
            'discovered_patterns': self.risk_discovery.discovered_patterns
        }
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'legal_bert_epoch_{epoch+1}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model
        num_discovered_risks = len(checkpoint['discovered_patterns'])
        self.model = FullyLearningBasedLegalBERT(
            config=checkpoint['config'],
            num_discovered_risks=num_discovered_risks
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore training state
        self.training_history = checkpoint['training_history']
        self.risk_discovery.discovered_patterns = checkpoint['discovered_patterns']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint['epoch']