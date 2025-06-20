#!/usr/bin/env python3
"""
Classifier Adapter
Adapter for creating and using custom classifiers for vault-specific tasks
"""

import os
import json
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac

logger = logging.getLogger(__name__)

# Check for ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Check for joblib (safer alternative to pickle)
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available. Install with: pip install joblib")
    # Create dummy sklearn classes
    class DummySklearn:
        def __init__(self, *args, **kwargs): pass
        def fit(self, *args, **kwargs): return self
        def predict(self, *args, **kwargs): return []
        def predict_proba(self, *args, **kwargs): return []
    TfidfVectorizer = MultinomialNB = SVC = RandomForestClassifier = DummySklearn
    LogisticRegression = Pipeline = train_test_split = cross_val_score = DummySklearn
    classification_report = confusion_matrix = DummySklearn
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")

@dataclass
class ClassificationExample:
    """Training example for classification"""
    text: str
    label: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ClassifierConfig:
    """Configuration for a classifier"""
    name: str
    classifier_type: str  # "nb", "svm", "rf", "lr"
    vectorizer_type: str  # "tfidf", "count"
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95

class ClassifierAdapter:
    """Adapter for custom classifiers with secure model persistence"""
    
    def __init__(self, model_dir: str = "./classifier_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.classifiers = {}
        self.training_data = {}
        
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file for integrity verification"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def _create_vectorizer(self, config: ClassifierConfig):
        """Create text vectorizer based on config"""
        if config.vectorizer_type == "tfidf":
            return TfidfVectorizer(
                max_features=config.max_features,
                ngram_range=config.ngram_range,
                min_df=config.min_df,
                max_df=config.max_df,
                lowercase=True,
                stop_words='english'
            )
        else:
            raise ValueError(f"Unknown vectorizer type: {config.vectorizer_type}")
            
    def _create_classifier(self, config: ClassifierConfig):
        """Create classifier based on config"""
        if config.classifier_type == "nb":
            return MultinomialNB(alpha=0.1)
        elif config.classifier_type == "svm":
            return SVC(kernel='linear', probability=True)
        elif config.classifier_type == "rf":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif config.classifier_type == "lr":
            return LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown classifier type: {config.classifier_type}")
            
    async def create_classifier(self, config: ClassifierConfig) -> bool:
        """Create a new classifier"""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available")
            return False
            
        try:
            # Create pipeline
            vectorizer = self._create_vectorizer(config)
            classifier = self._create_classifier(config)
            
            pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            
            self.classifiers[config.name] = {
                'pipeline': pipeline,
                'config': config,
                'trained': False,
                'labels': set(),
                'performance': {}
            }
            
            logger.info(f"Created classifier '{config.name}' with {config.classifier_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create classifier: {e}")
            return False
            
    async def add_training_examples(self, classifier_name: str, 
                                  examples: List[ClassificationExample]):
        """Add training examples for a classifier"""
        if classifier_name not in self.training_data:
            self.training_data[classifier_name] = []
            
        self.training_data[classifier_name].extend(examples)
        
        # Update labels
        if classifier_name in self.classifiers:
            labels = {ex.label for ex in examples}
            self.classifiers[classifier_name]['labels'].update(labels)
            
        logger.info(f"Added {len(examples)} training examples to '{classifier_name}'")
        
    async def train_classifier(self, classifier_name: str, 
                             test_size: float = 0.2,
                             cross_validate: bool = True) -> Dict[str, Any]:
        """Train a classifier with the provided examples"""
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}
            
        if classifier_name not in self.classifiers:
            return {"error": f"Classifier {classifier_name} not found"}
            
        if classifier_name not in self.training_data:
            return {"error": "No training data available"}
            
        try:
            # Get training data
            examples = self.training_data[classifier_name]
            if len(examples) < 10:
                return {"error": "Not enough training examples (need at least 10)"}
                
            X = [ex.text for ex in examples]
            y = [ex.label for ex in examples]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Get pipeline
            pipeline = self.classifiers[classifier_name]['pipeline']
            
            # Train
            logger.info(f"Training classifier '{classifier_name}' with {len(X_train)} examples...")
            
            # Run training in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, pipeline.fit, X_train, y_train)
            
            # Evaluate
            y_pred = await loop.run_in_executor(None, pipeline.predict, X_test)
            
            # Calculate metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Cross-validation if requested
            cv_scores = None
            if cross_validate and len(examples) >= 50:
                cv_scores = await loop.run_in_executor(
                    None,
                    lambda: cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro').tolist()
                )
                
            # Update classifier info
            self.classifiers[classifier_name]['trained'] = True
            self.classifiers[classifier_name]['performance'] = {
                'accuracy': report['accuracy'],
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'cv_scores': cv_scores,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'labels': list(set(y)),
                'classification_report': report
            }
            
            logger.info(f"Classifier '{classifier_name}' trained with accuracy: {report['accuracy']:.3f}")
            
            return self.classifiers[classifier_name]['performance']
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"error": str(e)}
            
    async def classify(self, classifier_name: str, text: str, 
                     return_proba: bool = False) -> Union[str, Dict[str, float]]:
        """Classify a single text"""
        if classifier_name not in self.classifiers:
            raise ValueError(f"Classifier {classifier_name} not found")
            
        if not self.classifiers[classifier_name]['trained']:
            raise ValueError(f"Classifier {classifier_name} not trained")
            
        pipeline = self.classifiers[classifier_name]['pipeline']
        
        try:
            if return_proba:
                # Get probabilities
                probas = pipeline.predict_proba([text])[0]
                classes = pipeline.classes_
                return {cls: float(prob) for cls, prob in zip(classes, probas)}
            else:
                # Get single prediction
                prediction = pipeline.predict([text])[0]
                return prediction
                
        except Exception as e:
            logger.error(f"Classification error: {e}")
            raise
            
    async def batch_classify(self, classifier_name: str, texts: List[str], 
                           return_proba: bool = False) -> List[Union[str, Dict[str, float]]]:
        """Classify multiple texts"""
        if classifier_name not in self.classifiers:
            raise ValueError(f"Classifier {classifier_name} not found")
            
        if not self.classifiers[classifier_name]['trained']:
            raise ValueError(f"Classifier {classifier_name} not trained")
            
        pipeline = self.classifiers[classifier_name]['pipeline']
        
        try:
            loop = asyncio.get_event_loop()
            
            if return_proba:
                probas = await loop.run_in_executor(None, pipeline.predict_proba, texts)
                classes = pipeline.classes_
                return [
                    {cls: float(prob) for cls, prob in zip(classes, proba)}
                    for proba in probas
                ]
            else:
                predictions = await loop.run_in_executor(None, pipeline.predict, texts)
                return predictions.tolist()
                
        except Exception as e:
            logger.error(f"Batch classification error: {e}")
            raise
            
    async def save_classifier(self, classifier_name: str):
        """Save classifier to disk securely"""
        if classifier_name not in self.classifiers:
            raise ValueError(f"Classifier {classifier_name} not found")
            
        classifier_data = self.classifiers[classifier_name]
        
        # Use joblib if available, with additional security measures
        if JOBLIB_AVAILABLE and SKLEARN_AVAILABLE:
            model_path = self.model_dir / f"{classifier_name}.joblib"
            metadata_path = self.model_dir / f"{classifier_name}_metadata.json"
            
            # Save model with joblib (safer than pickle)
            joblib.dump(classifier_data['pipeline'], model_path, compress=3)
            
            # Save metadata separately as JSON
            metadata = {
                'config': {
                    'name': classifier_data['config'].name,
                    'classifier_type': classifier_data['config'].classifier_type,
                    'vectorizer_type': classifier_data['config'].vectorizer_type,
                    'max_features': classifier_data['config'].max_features,
                    'ngram_range': classifier_data['config'].ngram_range,
                    'min_df': classifier_data['config'].min_df,
                    'max_df': classifier_data['config'].max_df
                },
                'labels': list(classifier_data['labels']),
                'performance': classifier_data['performance'],
                'trained': classifier_data['trained'],
                'saved_at': datetime.now().isoformat(),
                'model_hash': self._calculate_file_hash(model_path)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        else:
            # Fallback: Save only metadata if ML libraries not available
            metadata_path = self.model_dir / f"{classifier_name}_metadata.json"
            metadata = {
                'config': {
                    'name': classifier_data['config'].name,
                    'classifier_type': classifier_data['config'].classifier_type,
                    'vectorizer_type': classifier_data['config'].vectorizer_type,
                    'max_features': classifier_data['config'].max_features,
                    'ngram_range': classifier_data['config'].ngram_range,
                    'min_df': classifier_data['config'].min_df,
                    'max_df': classifier_data['config'].max_df
                },
                'labels': list(classifier_data['labels']),
                'performance': classifier_data.get('performance', {}),
                'trained': classifier_data.get('trained', False),
                'saved_at': datetime.now().isoformat(),
                'ml_libraries_available': False
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        # Save training data if exists
        if classifier_name in self.training_data:
            training_path = self.model_dir / f"{classifier_name}_training.json"
            training_data = [
                {
                    'text': ex.text,
                    'label': ex.label,
                    'metadata': ex.metadata
                }
                for ex in self.training_data[classifier_name]
            ]
            with open(training_path, 'w') as f:
                json.dump(training_data, f, indent=2)
                
        logger.info(f"Saved classifier '{classifier_name}' to {save_path}")
        
    async def load_classifier(self, classifier_name: str) -> bool:
        """Load classifier from disk securely"""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available for loading classifier")
            return False
            
        # Try joblib format first (newer, safer)
        joblib_path = self.model_dir / f"{classifier_name}.joblib"
        metadata_path = self.model_dir / f"{classifier_name}_metadata.json"
        
        if joblib_path.exists() and metadata_path.exists() and JOBLIB_AVAILABLE:
            try:
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Verify model integrity
                stored_hash = metadata.get('model_hash')
                if stored_hash:
                    current_hash = self._calculate_file_hash(joblib_path)
                    if current_hash != stored_hash:
                        logger.error(f"Model file integrity check failed for {classifier_name}")
                        return False
                
                # Load model
                pipeline = joblib.load(joblib_path)
                
                # Reconstruct classifier data
                config = ClassifierConfig(**metadata['config'])
                self.classifiers[classifier_name] = {
                    'pipeline': pipeline,
                    'config': config,
                    'labels': set(metadata['labels']),
                    'performance': metadata.get('performance', {}),
                    'trained': metadata.get('trained', False)
                }
                
                logger.info(f"Loaded classifier '{classifier_name}' from {joblib_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load classifier with joblib: {e}")
                return False
        
        # Check for legacy pickle format (log warning)
        legacy_path = self.model_dir / f"{classifier_name}.pkl"
        if legacy_path.exists():
            logger.warning(f"Found legacy pickle format for {classifier_name}. Please re-save the model for better security.")
            return False
        
        logger.error(f"Classifier files for '{classifier_name}' not found")
        return False
            
            # Load training data if exists
            training_path = self.model_dir / f"{classifier_name}_training.json"
            if training_path.exists():
                with open(training_path, 'r') as f:
                    training_data = json.load(f)
                    
                examples = [
                    ClassificationExample(
                        text=item['text'],
                        label=item['label'],
                        metadata=item.get('metadata')
                    )
                    for item in training_data
                ]
                self.training_data[classifier_name] = examples
                
            logger.info(f"Loaded classifier '{classifier_name}' from {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return False
            
    def get_classifier_info(self, classifier_name: str) -> Dict[str, Any]:
        """Get information about a classifier"""
        if classifier_name not in self.classifiers:
            return {"error": f"Classifier {classifier_name} not found"}
            
        classifier_data = self.classifiers[classifier_name]
        
        return {
            "name": classifier_name,
            "type": classifier_data['config'].classifier_type,
            "trained": classifier_data['trained'],
            "labels": list(classifier_data['labels']),
            "performance": classifier_data.get('performance', {}),
            "training_examples": len(self.training_data.get(classifier_name, []))
        }
        
    def list_classifiers(self) -> List[str]:
        """List all available classifiers"""
        return list(self.classifiers.keys())

# Example usage for vault-specific classifiers
async def create_vault_classifiers():
    """Create classifiers useful for vault management"""
    adapter = ClassifierAdapter()
    
    print("🤖 Creating Vault-Specific Classifiers")
    
    # 1. Document Type Classifier
    doc_type_config = ClassifierConfig(
        name="document_type",
        classifier_type="nb",
        vectorizer_type="tfidf",
        max_features=3000,
        ngram_range=(1, 2)
    )
    
    await adapter.create_classifier(doc_type_config)
    
    # Add training examples
    doc_type_examples = [
        # Meeting notes
        ClassificationExample("Discussed project timeline with team. Action items: review specs", "meeting"),
        ClassificationExample("1:1 with manager. Talked about career goals and performance", "meeting"),
        ClassificationExample("Sprint planning session. Estimated story points for backlog", "meeting"),
        
        # Daily notes
        ClassificationExample("Today I worked on the API integration. Fixed bug #123", "daily_note"),
        ClassificationExample("Morning: emails. Afternoon: coding. Evening: documentation", "daily_note"),
        
        # Project docs
        ClassificationExample("Project Alpha: Requirements and Architecture Overview", "project"),
        ClassificationExample("Technical specification for the payment processing system", "project"),
        
        # Personal notes
        ClassificationExample("Thoughts on productivity. Need to improve focus time", "personal"),
        ClassificationExample("Book notes: Deep Work by Cal Newport. Key takeaways", "personal"),
    ]
    
    await adapter.add_training_examples("document_type", doc_type_examples)
    
    # 2. Priority Classifier
    priority_config = ClassifierConfig(
        name="priority",
        classifier_type="lr",
        vectorizer_type="tfidf",
        max_features=2000
    )
    
    await adapter.create_classifier(priority_config)
    
    priority_examples = [
        ClassificationExample("URGENT: Server is down. Need immediate attention", "high"),
        ClassificationExample("Critical bug affecting all users. Must fix today", "high"),
        ClassificationExample("Review PR when you have time. No rush", "low"),
        ClassificationExample("Nice to have feature. Can wait until next sprint", "low"),
        ClassificationExample("Complete user story by end of sprint", "medium"),
    ]
    
    await adapter.add_training_examples("priority", priority_examples)
    
    # 3. Sentiment Classifier
    sentiment_config = ClassifierConfig(
        name="sentiment",
        classifier_type="svm",
        vectorizer_type="tfidf"
    )
    
    await adapter.create_classifier(sentiment_config)
    
    sentiment_examples = [
        ClassificationExample("Great progress today! Team did amazing work", "positive"),
        ClassificationExample("Frustrated with the bugs. Nothing seems to work", "negative"),
        ClassificationExample("Meeting was okay. Got some things done", "neutral"),
        ClassificationExample("Excellent presentation. Very impressed with results", "positive"),
        ClassificationExample("Disappointed with the delays. Behind schedule again", "negative"),
    ]
    
    await adapter.add_training_examples("sentiment", sentiment_examples)
    
    return adapter

async def test_classifier_adapter():
    """Test classifier functionality"""
    if not SKLEARN_AVAILABLE:
        print("⚠️ scikit-learn not available. Install with: pip install scikit-learn")
        return
        
    # Create classifiers
    adapter = await create_vault_classifiers()
    
    # Train classifiers
    print("\n📚 Training classifiers...")
    
    for classifier_name in ["document_type", "priority", "sentiment"]:
        print(f"\nTraining {classifier_name}...")
        result = await adapter.train_classifier(classifier_name, test_size=0.3)
        
        if "error" not in result:
            print(f"✅ Accuracy: {result['accuracy']:.3f}")
            print(f"   F1 Score: {result['macro_f1']:.3f}")
        else:
            print(f"❌ Error: {result['error']}")
            
    # Test classification
    print("\n🔍 Testing classification...")
    
    test_texts = [
        "Met with Sarah to discuss Q4 roadmap. We need to prioritize mobile features",
        "CRITICAL: Database connection pool exhausted. Users can't login",
        "Feeling great about the progress. Team morale is high!",
        "Today: fixed 3 bugs, reviewed 2 PRs, updated documentation"
    ]
    
    for text in test_texts:
        print(f"\nText: {text[:50]}...")
        
        # Classify with each classifier
        for classifier_name in ["document_type", "priority", "sentiment"]:
            try:
                # Get prediction
                prediction = await adapter.classify(classifier_name, text)
                
                # Get probabilities
                probas = await adapter.classify(classifier_name, text, return_proba=True)
                
                print(f"{classifier_name}: {prediction} (confidence: {probas[prediction]:.2f})")
            except Exception as e:
                print(f"{classifier_name}: Error - {e}")
                
    # Save classifiers
    print("\n💾 Saving classifiers...")
    for classifier_name in adapter.list_classifiers():
        await adapter.save_classifier(classifier_name)
        
    print("\n✅ Classifiers ready for use!")

if __name__ == "__main__":
    asyncio.run(test_classifier_adapter())