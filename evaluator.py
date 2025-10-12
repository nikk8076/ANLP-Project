"""
Evaluation and Analysis Tools for Legal-BERT
"""
import numpy as np
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class LegalBertEvaluator:
    """
    Comprehensive evaluation for Legal-BERT with discovered risk patterns
    """
    def __init__(self, model, tokenizer, risk_discovery):
        self.model = model
        self.tokenizer = tokenizer
        self.risk_discovery = risk_discovery
        self.evaluation_results = {}
    
    def evaluate_model(self, test_loader, save_results: bool = True) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        print("Starting comprehensive evaluation...")
        
        # Collect predictions
        all_predictions = []
        all_true_labels = []
        all_severity_preds = []
        all_severity_true = []
        all_importance_preds = []
        all_importance_true = []
        all_confidences = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                
                # Get predictions
                outputs = self.model.predict_risk_pattern(input_ids, attention_mask)
                
                # Store results
                all_predictions.extend(outputs['predicted_risk_id'])
                all_true_labels.extend(batch['risk_label'].numpy())
                all_severity_preds.extend(outputs['severity_score'])
                all_severity_true.extend(batch['severity_score'].numpy())
                all_importance_preds.extend(outputs['importance_score'])
                all_importance_true.extend(batch['importance_score'].numpy())
                all_confidences.extend(outputs['confidence'])
        
        # Calculate metrics
        results = {
            'classification_metrics': self._calculate_classification_metrics(
                all_true_labels, all_predictions, all_confidences
            ),
            'regression_metrics': self._calculate_regression_metrics(
                all_severity_true, all_severity_preds,
                all_importance_true, all_importance_preds
            ),
            'risk_pattern_analysis': self._analyze_risk_patterns(
                all_true_labels, all_predictions
            )
        }
        
        self.evaluation_results = results
        
        if save_results:
            self.save_evaluation_results(results)
        
        print("Evaluation complete!")
        return results
    
    def _calculate_classification_metrics(self, true_labels: List[int], 
                                        predictions: List[int], 
                                        confidences: List[float]) -> Dict[str, Any]:
        """Calculate classification metrics"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Confidence analysis
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std
        }
    
    def _calculate_regression_metrics(self, severity_true: List[float], severity_pred: List[float],
                                    importance_true: List[float], importance_pred: List[float]) -> Dict[str, Any]:
        """Calculate regression metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Severity metrics
        severity_mse = mean_squared_error(severity_true, severity_pred)
        severity_mae = mean_absolute_error(severity_true, severity_pred)
        severity_r2 = r2_score(severity_true, severity_pred)
        
        # Importance metrics
        importance_mse = mean_squared_error(importance_true, importance_pred)
        importance_mae = mean_absolute_error(importance_true, importance_pred)
        importance_r2 = r2_score(importance_true, importance_pred)
        
        return {
            'severity': {
                'mse': severity_mse,
                'mae': severity_mae,
                'r2_score': severity_r2
            },
            'importance': {
                'mse': importance_mse,
                'mae': importance_mae,
                'r2_score': importance_r2
            }
        }
    
    def _analyze_risk_patterns(self, true_labels: List[int], predictions: List[int]) -> Dict[str, Any]:
        """Analyze discovered risk patterns"""
        discovered_patterns = self.risk_discovery.discovered_patterns
        pattern_names = list(discovered_patterns.keys())
        
        # Pattern distribution
        true_distribution = defaultdict(int)
        pred_distribution = defaultdict(int)
        
        for label in true_labels:
            true_distribution[pattern_names[label]] += 1
        
        for pred in predictions:
            pred_distribution[pattern_names[pred]] += 1
        
        # Pattern-specific performance
        pattern_performance = {}
        for i, pattern_name in enumerate(pattern_names):
            pattern_true = [1 if label == i else 0 for label in true_labels]
            pattern_pred = [1 if pred == i else 0 for pred in predictions]
            
            if sum(pattern_true) > 0:  # Avoid division by zero
                precision = sum([1 for t, p in zip(pattern_true, pattern_pred) if t == 1 and p == 1]) / max(sum(pattern_pred), 1)
                recall = sum([1 for t, p in zip(pattern_true, pattern_pred) if t == 1 and p == 1]) / sum(pattern_true)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                pattern_performance[pattern_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': sum(pattern_true)
                }
        
        return {
            'true_distribution': dict(true_distribution),
            'predicted_distribution': dict(pred_distribution),
            'pattern_performance': pattern_performance,
            'discovered_patterns_info': discovered_patterns
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            raise ValueError("Must run evaluation first")
        
        results = self.evaluation_results
        
        report = []
        report.append("=" * 80)
        report.append("LEGAL-BERT EVALUATION REPORT")
        report.append("=" * 80)
        
        # Classification Performance
        report.append("\nRISK CLASSIFICATION PERFORMANCE")
        report.append("-" * 50)
        clf_metrics = results['classification_metrics']
        report.append(f"Accuracy: {clf_metrics['accuracy']:.4f}")
        report.append(f"Precision: {clf_metrics['precision']:.4f}")
        report.append(f"Recall: {clf_metrics['recall']:.4f}")
        report.append(f"F1-Score: {clf_metrics['f1_score']:.4f}")
        report.append(f"Average Confidence: {clf_metrics['avg_confidence']:.4f}")
        
        # Regression Performance
        report.append("\nREGRESSION PERFORMANCE")
        report.append("-" * 50)
        reg_metrics = results['regression_metrics']
        
        report.append("Severity Prediction:")
        report.append(f"  MSE: {reg_metrics['severity']['mse']:.4f}")
        report.append(f"  MAE: {reg_metrics['severity']['mae']:.4f}")
        report.append(f"  R²: {reg_metrics['severity']['r2_score']:.4f}")
        
        report.append("Importance Prediction:")
        report.append(f"  MSE: {reg_metrics['importance']['mse']:.4f}")
        report.append(f"  MAE: {reg_metrics['importance']['mae']:.4f}")
        report.append(f"  R²: {reg_metrics['importance']['r2_score']:.4f}")
        
        # Risk Pattern Analysis
        report.append("\nDISCOVERED RISK PATTERNS")
        report.append("-" * 50)
        pattern_analysis = results['risk_pattern_analysis']
        
        report.append("Pattern Distribution (True vs Predicted):")
        for pattern, count in pattern_analysis['true_distribution'].items():
            pred_count = pattern_analysis['predicted_distribution'].get(pattern, 0)
            report.append(f"  {pattern}: {count} → {pred_count}")
        
        report.append("\nPattern-Specific Performance:")
        for pattern, metrics in pattern_analysis['pattern_performance'].items():
            report.append(f"  {pattern}:")
            report.append(f"    Precision: {metrics['precision']:.4f}")
            report.append(f"    Recall: {metrics['recall']:.4f}")
            report.append(f"    F1-Score: {metrics['f1_score']:.4f}")
            report.append(f"    Support: {metrics['support']}")
        
        # Discovered Patterns Info
        report.append("\nDISCOVERED PATTERN DETAILS")
        report.append("-" * 50)
        for pattern_name, details in pattern_analysis['discovered_patterns_info'].items():
            report.append(f"\n{pattern_name}:")
            report.append(f"  Clauses: {details['clause_count']}")
            report.append(f"  Risk Intensity: {details['avg_risk_intensity']:.3f}")
            report.append(f"  Legal Complexity: {details['avg_legal_complexity']:.3f}")
            report.append(f"  Key Terms: {', '.join(details['key_terms'][:5])}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_for_json(results)
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save report
        report = self.generate_report()
        with open('evaluation_report.txt', 'w') as f:
            f.write(report)
        
        print("Evaluation results saved:")
        print("  - evaluation_results.json")
        print("  - evaluation_report.txt")
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

# Mock imports for environments without sklearn/matplotlib
try:
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError:
    print("Warning: Some evaluation dependencies not available. Using mock implementations.")
    
    # Mock torch
    class MockTensor:
        def __init__(self, data):
            self.data = data
        def numpy(self):
            return self.data
        def to(self, device):
            return self
    
    class MockModule:
        def eval(self):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    torch = type('torch', (), {
        'no_grad': lambda: type('context', (), {'__enter__': lambda self: None, '__exit__': lambda *args: None})()
    })()
    
    # Mock sklearn functions
    def accuracy_score(y_true, y_pred):
        return sum([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true)
    
    def precision_recall_fscore_support(y_true, y_pred, average=None):
        return 0.5, 0.5, 0.5, None
    
    def confusion_matrix(y_true, y_pred):
        return [[1, 0], [0, 1]]
    
    def mean_squared_error(y_true, y_pred):
        return sum([(t - p) ** 2 for t, p in zip(y_true, y_pred)]) / len(y_true)
    
    def mean_absolute_error(y_true, y_pred):
        return sum([abs(t - p) for t, p in zip(y_true, y_pred)]) / len(y_true)
    
    def r2_score(y_true, y_pred):
        return 0.5