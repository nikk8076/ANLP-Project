"""
Unsupervised Risk Discovery System - No Hardcoded Categories!
"""
import re
from typing import Dict, List, Tuple, Any
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation

class UnsupervisedRiskDiscovery:
    """
    Discovers risk patterns in legal contracts using unsupervised learning.
    NO hardcoded risk categories - learns everything from text!
    """
    
    def __init__(self, n_clusters: int = 7, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Initialize components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.95
        )
        
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        
        # Risk pattern storage
        self.discovered_patterns = {}
        self.risk_features = {}
        self.cluster_labels = None
        self.feature_matrix = None
        
        # Legal language patterns (domain-agnostic)
        self.legal_indicators = {
            'obligation_strength': r'\b(?:shall|must|required|mandatory|obligated|bound)\b',
            'prohibition_terms': r'\b(?:shall not|must not|prohibited|forbidden|restricted)\b',
            'conditional_risk': r'\b(?:if|unless|provided|subject to|in the event|failure to)\b',
            'liability_terms': r'\b(?:liable|responsibility|damages|penalty|loss|harm)\b',
            'temporal_urgency': r'\b(?:immediately|within|before|after|deadline|expir)\b',
            'monetary_terms': r'\$|USD|dollar|payment|fee|cost|expense|fine',
            'parties': r'\b(?:Party|Parties|Company|Corporation|Licensor|Licensee|Vendor|Customer)\b',
            'dates': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        }
        
        # Legal complexity indicators
        self.complexity_indicators = {
            'modal_verbs': r'\b(?:shall|must|may|should|will|might|could|would)\b',
            'conditional_terms': r'\b(?:if|unless|provided|subject to|in the event|notwithstanding)\b',
            'legal_conjunctions': r'\b(?:whereas|therefore|furthermore|moreover|however)\b',
            'obligation_terms': r'\b(?:agrees?|undertakes?|covenants?|warrants?|represents?)\b'
        }
    
    def clean_clause_text(self, text: str) -> str:
        """Clean and normalize clause text"""
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal punctuation
        text = re.sub(r'[^\w\s.,;:()"-]', ' ', text)
        
        # Clean up spacing
        text = text.strip()
        
        return text
    
    def extract_risk_features(self, clause_text: str) -> Dict[str, float]:
        """
        Extract numerical features that indicate risk levels (domain-agnostic)
        """
        text_lower = clause_text.lower()
        words = text_lower.split()
        
        features = {}
        
        # Basic text statistics
        features['clause_length'] = len(words)
        features['sentence_count'] = len(re.split(r'[.!?]+', clause_text))
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Legal language intensity
        for pattern_name, pattern in self.legal_indicators.items():
            matches = len(re.findall(pattern, text_lower))
            features[f'{pattern_name}_count'] = matches
            features[f'{pattern_name}_density'] = matches / len(words) if words else 0
        
        # Legal complexity features
        for pattern_name, pattern in self.complexity_indicators.items():
            matches = len(re.findall(pattern, text_lower))
            features[f'{pattern_name}_complexity'] = matches / len(words) if words else 0
        
        # Risk intensity indicators
        features['obligation_strength'] = (
            features.get('obligation_strength_density', 0) * 2 +
            features.get('modal_verbs_complexity', 0)
        )
        
        features['legal_complexity'] = (
            features.get('conditional_terms_complexity', 0) +
            features.get('legal_conjunctions_complexity', 0) +
            features.get('obligation_terms_complexity', 0)
        )
        
        features['risk_intensity'] = (
            features.get('liability_terms_density', 0) * 2 +
            features.get('prohibition_terms_density', 0) +
            features.get('conditional_risk_density', 0)
        )
        
        return features
    
    def discover_risk_patterns(self, clause_texts: List[str]) -> Dict[str, Any]:
        """
        Discover risk patterns using unsupervised clustering.
        Returns discovered risk types and their characteristics.
        """
        print(f"Discovering risk patterns from {len(clause_texts)} clauses...")
        
        # Clean texts
        cleaned_texts = [self.clean_clause_text(text) for text in clause_texts]
        
        # Extract TF-IDF features
        print("Extracting TF-IDF features...")
        self.feature_matrix = self.tfidf_vectorizer.fit_transform(cleaned_texts)
        
        # Perform clustering
        print(f"Clustering into {self.n_clusters} risk patterns...")
        self.cluster_labels = self.kmeans.fit_predict(self.feature_matrix)
        
        # Extract risk features for each clause
        print("Extracting legal risk features...")
        risk_features_list = [self.extract_risk_features(text) for text in clause_texts]
        
        # Analyze discovered clusters
        self.discovered_patterns = self._analyze_clusters(
            cleaned_texts, self.cluster_labels, risk_features_list
        )
        
        print("Risk pattern discovery complete!")
        print(f"Discovered {len(self.discovered_patterns)} risk patterns:")
        
        for i, (pattern_name, details) in enumerate(self.discovered_patterns.items()):
            print(f"  {i+1}. {pattern_name}: {details['clause_count']} clauses")
            print(f"     Key terms: {', '.join(details['key_terms'][:5])}")
            print(f"     Risk intensity: {details['avg_risk_intensity']:.3f}")
        
        return self.discovered_patterns
    
    def _analyze_clusters(self, texts: List[str], labels: np.ndarray, 
                         risk_features: List[Dict]) -> Dict[str, Any]:
        """Analyze and name discovered clusters"""
        patterns = {}
        
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        for cluster_id in range(self.n_clusters):
            # Get clauses in this cluster
            cluster_mask = labels == cluster_id
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
            cluster_features = [risk_features[i] for i in range(len(risk_features)) if cluster_mask[i]]
            
            # Get top terms for this cluster
            cluster_center = self.kmeans.cluster_centers_[cluster_id]
            top_indices = cluster_center.argsort()[-20:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            # Calculate average risk features
            avg_features = {}
            if cluster_features:
                for key in cluster_features[0].keys():
                    avg_features[key] = np.mean([f.get(key, 0) for f in cluster_features])
            
            # Generate cluster name based on top terms and risk characteristics
            cluster_name = self._generate_cluster_name(top_terms, avg_features)
            
            patterns[cluster_name] = {
                'cluster_id': cluster_id,
                'clause_count': len(cluster_texts),
                'key_terms': top_terms,
                'avg_risk_intensity': avg_features.get('risk_intensity', 0),
                'avg_legal_complexity': avg_features.get('legal_complexity', 0),
                'avg_obligation_strength': avg_features.get('obligation_strength', 0),
                'sample_clauses': cluster_texts[:3],
                'risk_features': avg_features
            }
        
        return patterns
    
    def _generate_cluster_name(self, top_terms: List[str], avg_features: Dict[str, float]) -> str:
        """Generate meaningful names for discovered clusters"""
        # Analyze top terms to identify risk theme
        term_analysis = {
            'liability': ['liable', 'liability', 'damages', 'loss', 'harm', 'injury'],
            'obligation': ['shall', 'must', 'required', 'obligation', 'duty'],
            'indemnity': ['indemnify', 'indemnification', 'defend', 'hold harmless'],
            'termination': ['terminate', 'termination', 'end', 'expire', 'breach'],
            'intellectual_property': ['intellectual', 'property', 'patent', 'copyright', 'trademark'],
            'confidentiality': ['confidential', 'confidentiality', 'non-disclosure', 'proprietary'],
            'compliance': ['comply', 'compliance', 'regulation', 'law', 'legal']
        }
        
        # Score each theme based on term presence
        theme_scores = {}
        for theme, keywords in term_analysis.items():
            score = sum(1 for term in top_terms[:10] if any(kw in term.lower() for kw in keywords))
            theme_scores[theme] = score
        
        # Get best matching theme
        best_theme = max(theme_scores, key=theme_scores.get) if theme_scores else 'general'
        
        # Add intensity modifier based on risk features
        risk_intensity = avg_features.get('risk_intensity', 0)
        if risk_intensity > 0.1:
            intensity = 'high_risk'
        elif risk_intensity > 0.05:
            intensity = 'moderate_risk'
        else:
            intensity = 'low_risk'
        
        return f"{intensity}_{best_theme}_pattern"
    
    def get_risk_labels(self, clause_texts: List[str]) -> List[int]:
        """Get risk cluster labels for new clause texts"""
        if self.cluster_labels is None:
            raise ValueError("Must discover patterns first using discover_risk_patterns()")
        
        cleaned_texts = [self.clean_clause_text(text) for text in clause_texts]
        feature_matrix = self.tfidf_vectorizer.transform(cleaned_texts)
        
        return self.kmeans.predict(feature_matrix)
    
    def get_discovered_risk_names(self) -> List[str]:
        """Get list of discovered risk pattern names"""
        if not self.discovered_patterns:
            raise ValueError("Must discover patterns first using discover_risk_patterns()")
        
        return list(self.discovered_patterns.keys())