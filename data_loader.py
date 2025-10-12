"""
Data loading and preprocessing for Legal-BERT training
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import re
from sklearn.model_selection import train_test_split

class CUADDataLoader:
    """
    CUAD dataset loader and preprocessor for learning-based risk classification
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df_clauses = None
        self.contracts = None
        self.splits = None
        
    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and parse CUAD dataset"""
        print(f"Loading CUAD dataset from {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            cuad_data = json.load(f)
        
        # Extract contract clauses
        clauses_data = []
        
        for item in cuad_data['data']:
            title = item['title']
            
            for paragraph in item['paragraphs']:
                context = paragraph['context']
                
                for qa in paragraph['qas']:
                    question = qa['question']
                    clause_category = question
                    
                    # Extract answers (clauses)
                    for answer in qa['answers']:
                        clause_text = answer['text']
                        start_pos = answer['answer_start']
                        
                        clauses_data.append({
                            'filename': title,
                            'clause_text': clause_text,
                            'category': clause_category,
                            'start_position': start_pos,
                            'contract_context': context
                        })
        
        self.df_clauses = pd.DataFrame(clauses_data)
        
        # Group by contract for analysis
        self.contracts = self.df_clauses.groupby('filename').agg({
            'clause_text': list,
            'category': list,
            'contract_context': 'first'
        }).reset_index()
        
        print(f"Loaded {len(self.df_clauses)} clauses from {len(self.contracts)} contracts")
        print(f"Found {self.df_clauses['category'].nunique()} unique clause categories")
        
        return self.df_clauses, self.contracts.set_index('filename').to_dict('index')
    
    def create_splits(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        """Create train/validation/test splits at contract level"""
        if self.contracts is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        unique_contracts = self.contracts['filename'].unique()
        
        # First split: train+val vs test
        train_val_contracts, test_contracts = train_test_split(
            unique_contracts,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        # Second split: train vs val
        train_contracts, val_contracts = train_test_split(
            train_val_contracts,
            test_size=val_size/(1-test_size),  # Adjust for remaining data
            random_state=random_state,
            shuffle=True
        )
        
        # Create clause-level splits
        train_clauses = self.df_clauses[self.df_clauses['filename'].isin(train_contracts)]
        val_clauses = self.df_clauses[self.df_clauses['filename'].isin(val_contracts)]
        test_clauses = self.df_clauses[self.df_clauses['filename'].isin(test_contracts)]
        
        self.splits = {
            'train': train_clauses,
            'val': val_clauses,
            'test': test_clauses
        }
        
        print(f"Data splits created:")
        print(f"Train: {len(train_clauses)} clauses from {len(train_contracts)} contracts")
        print(f"Val: {len(val_clauses)} clauses from {len(val_contracts)} contracts")
        print(f"Test: {len(test_clauses)} clauses from {len(test_contracts)} contracts")
        
        return self.splits
    
    def get_clause_texts(self, split: str = 'train') -> List[str]:
        """Get clause texts for a specific split"""
        if self.splits is None:
            raise ValueError("Splits must be created first using create_splits()")
        
        return self.splits[split]['clause_text'].tolist()
    
    def get_categories(self, split: str = 'train') -> List[str]:
        """Get categories for a specific split"""
        if self.splits is None:
            raise ValueError("Splits must be created first using create_splits()")
        
        return self.splits[split]['category'].tolist()
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess clause text"""
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal punctuation
        text = re.sub(r'[^\w\s.,;:()"-]', ' ', text)
        
        # Clean up spacing
        text = text.strip()
        
        return text