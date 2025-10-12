"""
Utilities and helper functions for Legal-BERT project
"""
import os
import json
import re
from typing import Dict, List, Any, Tuple
import logging

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('legal_bert.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_directory_exists(path: str):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file"""
    ensure_directory_exists(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON: {filepath}")

def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded JSON: {filepath}")
    return data

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep legal punctuation
    text = re.sub(r'[^\w\s.,;:()"-]', ' ', text)
    
    # Clean up spacing
    text = text.strip()
    
    return text

def extract_contract_metadata(filename: str) -> Dict[str, str]:
    """Extract metadata from contract filename"""
    # CUAD filename pattern: COMPANY_DATE_FILING_EXHIBIT_AGREEMENT
    parts = filename.replace('.txt', '').split('_')
    
    metadata = {
        'company': parts[0] if len(parts) > 0 else 'Unknown',
        'date': parts[1] if len(parts) > 1 else 'Unknown',
        'filing_type': parts[2] if len(parts) > 2 else 'Unknown',
        'exhibit': parts[3] if len(parts) > 3 else 'Unknown',
        'agreement_type': '_'.join(parts[4:]) if len(parts) > 4 else 'Unknown'
    }
    
    return metadata

def format_risk_score(score: float) -> str:
    """Format risk score for display"""
    if score < 2:
        return f"LOW ({score:.2f})"
    elif score < 5:
        return f"MEDIUM ({score:.2f})"
    elif score < 8:
        return f"HIGH ({score:.2f})"
    else:
        return f"CRITICAL ({score:.2f})"

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values"""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
    
    import statistics
    
    return {
        'mean': statistics.mean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
        'median': statistics.median(values)
    }

def print_progress_bar(iteration: int, total: int, prefix: str = 'Progress', 
                      suffix: str = 'Complete', length: int = 50):
    """Print a progress bar"""
    percent = (100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='')
    if iteration == total:
        print()

def validate_config(config) -> List[str]:
    """Validate configuration settings"""
    errors = []
    
    # Check required fields
    required_fields = ['bert_model_name', 'data_path', 'batch_size', 'num_epochs']
    for field in required_fields:
        if not hasattr(config, field):
            errors.append(f"Missing required config field: {field}")
    
    # Check data path exists
    if hasattr(config, 'data_path') and not os.path.exists(config.data_path):
        errors.append(f"Data path does not exist: {config.data_path}")
    
    # Check positive values
    if hasattr(config, 'batch_size') and config.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if hasattr(config, 'num_epochs') and config.num_epochs <= 0:
        errors.append("Number of epochs must be positive")
    
    # Check learning rate range
    if hasattr(config, 'learning_rate') and (config.learning_rate <= 0 or config.learning_rate > 1):
        errors.append("Learning rate must be between 0 and 1")
    
    return errors

def create_model_summary(model, config) -> str:
    """Create a summary of the model architecture"""
    try:
        # Try to get parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except:
        total_params = "Unknown"
        trainable_params = "Unknown"
    
    summary = [
        "MODEL SUMMARY",
        "=" * 50,
        f"Architecture: Legal-BERT (Fully Learning-Based)",
        f"Base Model: {config.bert_model_name}",
        f"Risk Categories: {config.num_risk_categories} (discovered)",
        f"Max Sequence Length: {config.max_sequence_length}",
        f"Dropout Rate: {config.dropout_rate}",
        f"Total Parameters: {total_params}",
        f"Trainable Parameters: {trainable_params}",
        f"Device: {config.device}",
        "=" * 50
    ]
    
    return "\n".join(summary)

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available"""
    dependencies = {
        'torch': False,
        'transformers': False,
        'sklearn': False,
        'numpy': False,
        'pandas': False
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies

def print_dependency_status():
    """Print status of dependencies"""
    deps = check_dependencies()
    
    print("DEPENDENCY STATUS")
    print("-" * 30)
    
    for dep, available in deps.items():
        status = "Available" if available else "Missing"
        print(f"{dep:12} : {status}")
    
    missing = [dep for dep, available in deps.items() if not available]
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install torch transformers scikit-learn numpy pandas")
        print("For demo mode, dependencies are not required.")
    else:
        print("\nAll dependencies available!")

def get_sample_contract_text() -> str:
    """Get sample contract text for testing"""
    return """
    SERVICES AGREEMENT
    
    This Services Agreement ("Agreement") is entered into as of the Effective Date
    by and between Company A ("Provider") and Company B ("Client").
    
    1. SERVICES
    Provider shall provide the services described in Exhibit A ("Services") to Client
    in accordance with the terms and conditions set forth herein.
    
    2. PAYMENT TERMS
    Client shall pay Provider the fees specified in Exhibit B within thirty (30) days
    of receipt of each invoice. Late payments shall incur a penalty of 1.5% per month.
    
    3. INDEMNIFICATION
    Each party shall indemnify and hold harmless the other party from and against any
    third-party claims arising out of such party's breach of this Agreement.
    
    4. LIMITATION OF LIABILITY
    In no event shall either party's liability exceed the total amount paid under this
    Agreement in the twelve (12) months preceding the claim.
    
    5. TERMINATION
    Either party may terminate this Agreement upon thirty (30) days written notice
    to the other party. Upon termination, all confidential information shall be returned.
    
    6. GOVERNING LAW
    This Agreement shall be governed by and construed in accordance with the laws
    of the State of Delaware.
    """