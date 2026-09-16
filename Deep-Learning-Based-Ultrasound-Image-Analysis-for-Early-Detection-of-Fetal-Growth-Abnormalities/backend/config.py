"""
Configuration Module
====================
Central configuration for the application.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
UPLOAD_FOLDER = PROJECT_ROOT / 'uploads'
TEMPLATE_FOLDER = PROJECT_ROOT / 'templates'
STATIC_FOLDER = PROJECT_ROOT / 'static'
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / 'rag' / 'knowledge_base.json'

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
STATIC_FOLDER.mkdir(exist_ok=True)

# Flask Configuration
FLASK_CONFIG = {
    'UPLOAD_FOLDER': str(UPLOAD_FOLDER),
    'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50MB
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif'},
    'JSON_SORT_KEYS': False,
    'DEBUG': os.getenv('FLASK_DEBUG', 'True') == 'True'
}

# Model Configuration
MODEL_CONFIG = {
    'model_name': 'unet_resnet18',
    'architecture': 'U-Net (ResNet18 encoder) + Classification Head',
    'num_classes': 3,
    'device': 'cpu',  # 'cpu' or 'cuda'
    'input_size': 224,
    'pretrained': True,
    'checkpoint_path': 'models/fgr_unet_model.pth',
}

# Class configurations — index order: normal(0), fgr(1), abnormal(2)
CLASS_LABELS = {
    0: "Normal Fetus",
    1: "Fetal Growth Restriction (FGR)",
    2: "Other Fetal Abnormalities"
}

RISK_LABELS = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk"
}

CLASS_DESCRIPTIONS = {
    0: "The fetus appears to be developing normally with appropriate growth parameters.",
    1: "Signs of restricted fetal growth detected. Abdominal circumference and/or femur length below expected percentile.",
    2: "Abnormal anatomical features detected. Requires immediate clinical evaluation."
}

# RAG Configuration
RAG_CONFIG = {
    'embedding_model': 'all-MiniLM-L6-v2',
    'knowledge_base_path': str(KNOWLEDGE_BASE_PATH),
    'top_k_retrieval': 3,
    'similarity_threshold': 0.5
}

# Report Configuration
REPORT_CONFIG = {
    'include_visualizations': True,
    'include_recommendations': True,
    'confidence_threshold': 50.0  # Percentage
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}

# Risk level mapping (for report styling)
RISK_LEVELS = {
    'low': {'threshold': 40, 'label': 'LOW RISK', 'color': '#27ae60'},
    'moderate': {'threshold': 60, 'label': 'MODERATE RISK', 'color': '#f39c12'},
    'high': {'threshold': 75, 'label': 'HIGH RISK', 'color': '#e74c3c'},
    'urgent': {'threshold': 100, 'label': 'URGENT', 'color': '#c0392b'}
}
