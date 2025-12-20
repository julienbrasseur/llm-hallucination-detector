"""
XGBoost probe for binary classification.
"""

import numpy as np
import xgboost as xgb
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report
)
from typing import Dict, Optional, Union, Tuple
import pickle


class XGBoostProbe:
    """
    XGBoost classifier.
    
    Supports:
    - Sparse matrix inputs (scipy.sparse)
    - Early stopping with validation set
    - Feature importance extraction
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, xgb_params: Optional[Dict] = None):
        """
        Initialize XGBoost probe.
        
        Args:
            xgb_params: XGBoost hyperparameters. If None, uses sensible defaults
                       optimized for sparse, high-dimensional data.
        """
        # Default parameters
        default_params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'tree_method': 'hist',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
        }
        
        self.params = {**default_params, **(xgb_params or {})}
        self.model = None
        self.feature_names = None
        self.eval_results = {}
    
    def fit(
        self,
        X_train: Union[np.ndarray, sparse.spmatrix],
        y_train: np.ndarray,
        X_val: Optional[Union[np.ndarray, sparse.spmatrix]] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 20,
        verbose: bool = True
    ):
        """
        Train XGBoost model on features.
        
        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training labels [n_samples]
            X_val: Optional validation features for early stopping
            y_val: Optional validation labels
            early_stopping_rounds: Stop if validation metric doesn't improve
            verbose: Print training progress
        """
        # Convert labels to numpy if needed
        y_train = np.array(y_train)
        if y_val is not None:
            y_val = np.array(y_val)
        
        # Create DMatrix (XGBoost's internal data structure)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Setup evaluation
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))
        
        # Train
        evals_result = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get('n_estimators', 200),
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
            evals_result=evals_result,
            verbose_eval=10 if verbose else False
        )
        
        self.eval_results = evals_result
        
        if verbose:
            print(f"\nTraining complete. Best iteration: {self.model.best_iteration}")
            if 'val' in evals_result:
                best_score = evals_result['val'][self.params['eval_metric']][self.model.best_iteration]
                print(f"Best validation {self.params['eval_metric']}: {best_score:.4f}")
    
    def predict_proba(
        self,
        X: Union[np.ndarray, sparse.spmatrix]
    ) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features [n_samples, n_features]
        
        Returns:
            Probabilities [n_samples, 2] for classes [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        dtest = xgb.DMatrix(X)
        probs_pos = self.model.predict(dtest)
        
        # Return probabilities for both classes
        probs = np.column_stack([1 - probs_pos, probs_pos])
        return probs
    
    def predict(
        self,
        X: Union[np.ndarray, sparse.spmatrix],
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features [n_samples, n_features]
            threshold: Decision threshold (default 0.5)
        
        Returns:
            Predicted labels [n_samples]
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)
    
    def evaluate(
        self,
        X: Union[np.ndarray, sparse.spmatrix],
        y: np.ndarray,
        threshold: float = 0.5,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model and return comprehensive metrics.
        
        Args:
            X: Features
            y: True labels
            threshold: Decision threshold
            verbose: Print detailed results
        
        Returns:
            Dictionary of metrics
        """
        y = np.array(y)
        
        # Predictions
        y_pred = self.predict(X, threshold=threshold)
        y_proba = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(y, y_proba)
        except ValueError:
            auc = 0.0  # In case of single-class predictions
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        if verbose:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"AUC:       {auc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y, y_pred, digits=4, zero_division=0))
            print("="*60)
        
        return metrics
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain',
        top_k: Optional[int] = None
    ) -> Dict[int, float]:
        """
        Get feature importance scores from trained model.
        
        Args:
            importance_type: Type of importance
                - 'gain': Average gain across splits using the feature
                - 'weight': Number of times feature appears in trees
                - 'cover': Average coverage across splits
            top_k: Return only top K most important features (None = all)
        
        Returns:
            Dictionary mapping feature index to importance score
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get importance scores
        importance = self.model.get_score(importance_type=importance_type)
        
        # Convert feature names (f0, f1, ...) to integers
        importance_dict = {
            int(k.replace('f', '')): v 
            for k, v in importance.items()
        }
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Return top-k if specified
        if top_k is not None:
            sorted_importance = dict(list(sorted_importance.items())[:top_k])
        
        return sorted_importance
    
    def save(self, filepath: str):
        """Save trained model to disk using pickle."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        save_dict = {
            'model': self.model,
            'params': self.params,
            'eval_results': self.eval_results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'XGBoostProbe':
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        probe = cls(xgb_params=save_dict['params'])
        probe.model = save_dict['model']
        probe.eval_results = save_dict['eval_results']
        
        print(f"Model loaded from {filepath}")
        return probe
    
    def __repr__(self):
        status = "trained" if self.model is not None else "untrained"
        return f"XGBoostProbe({status}, n_estimators={self.params.get('n_estimators', 'N/A')})"