"""
Machine learning module for the trading system.
Provides model training, feature selection, and prediction capabilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

# Expose key classes
from .model_trainer import ModelTrainer
from .feature_selector import FeatureSelector
from .predictors import SignalPredictor, StrengthPredictor
from .utils import prepare_data, evaluate_model, save_model

__all__ = [
    'ModelTrainer',
    'FeatureSelector',
    'SignalPredictor',
    'StrengthPredictor',
    'prepare_data',
    'evaluate_model',
    'save_model'
]