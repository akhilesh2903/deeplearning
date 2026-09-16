"""
Utility Functions and Helpers
==============================
Common utilities used across the application.
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations and validation."""
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    @staticmethod
    def is_allowed_file(filename):
        """Check if file extension is allowed."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in FileManager.ALLOWED_EXTENSIONS
    
    @staticmethod
    def is_file_size_valid(file):
        """Check if file size is within limit."""
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        return size <= FileManager.MAX_FILE_SIZE
    
    @staticmethod
    def get_file_hash(filepath, algorithm='md5'):
        """Calculate hash of file for integrity checking."""
        hash_obj = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    @staticmethod
    def cleanup_old_files(directory, hours=24):
        """Remove files older than specified hours."""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        removed = 0
        
        try:
            for file in Path(directory).glob('*'):
                if file.stat().st_mtime < cutoff_time:
                    file.unlink()
                    removed += 1
            logger.info(f"Cleaned up {removed} old files from {directory}")
        except Exception as e:
            logger.error(f"Error cleaning up files: {e}")
        
        return removed


class JSONUtils:
    """JSON serialization utilities."""
    
    @staticmethod
    def save_json(data, filepath):
        """Save data to JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"JSON saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            return False
    
    @staticmethod
    def load_json(filepath):
        """Load data from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return None


class ReportFormatter:
    """Formats reports for different output formats."""
    
    @staticmethod
    def format_as_text(report_dict):
        """Format report dictionary as readable text."""
        lines = []
        
        for key, section in report_dict.items():
            if isinstance(section, dict) and 'section_title' in section:
                lines.append("\n" + "="*80)
                lines.append(section.get('section_title', 'Section'))
                lines.append("="*80 + "\n")
                lines.append(section.get('content', ''))
        
        return '\n'.join(lines)
    
    @staticmethod
    def format_as_html(report_dict):
        """Format report dictionary as HTML."""
        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '<meta charset="UTF-8">',
            '<title>Ultrasound Diagnosis Report</title>',
            '<style>',
            'body { font-family: Arial, sans-serif; margin: 20px; }',
            '.section { margin-bottom: 20px; }',
            '.section-title { background: #667eea; color: white; padding: 10px; font-weight: bold; }',
            '.section-content { background: #f5f5f5; padding: 15px; }',
            '</style>',
            '</head>',
            '<body>',
            '<h1>Fetal Ultrasound Diagnosis Report</h1>'
        ]
        
        for key, section in report_dict.items():
            if isinstance(section, dict) and 'section_title' in section:
                html_parts.append(f'<div class="section">')
                html_parts.append(f'<div class="section-title">{section.get("section_title", "")}</div>')
                html_parts.append(f'<div class="section-content"><pre>{section.get("content", "")}</pre></div>')
                html_parts.append(f'</div>')
        
        html_parts.extend(['</body>', '</html>'])
        return '\n'.join(html_parts)


class ValidationUtils:
    """Validation utilities for inputs and data."""
    
    @staticmethod
    def validate_confidence_score(score):
        """Validate confidence score is between 0-100."""
        return 0 <= score <= 100
    
    @staticmethod
    def validate_class_index(index, num_classes=3):
        """Validate class index."""
        return 0 <= index < num_classes
    
    @staticmethod
    def validate_image_tensor_shape(tensor, expected_shape=(1, 3, 224, 224)):
        """Validate image tensor has expected shape."""
        return tensor.shape == expected_shape


class DiagnosisUtils:
    """Utility functions for diagnosis logic."""
    
    @staticmethod
    def get_risk_level(confidence_score):
        """Determine risk level from confidence score."""
        if confidence_score >= 75:
            return {
                'level': 'high',
                'label': 'HIGH RISK - URGENT',
                'description': 'Significant concern requiring immediate specialist consultation'
            }
        elif confidence_score >= 60:
            return {
                'level': 'moderate',
                'label': 'MODERATE RISK',
                'description': 'Borderline concern; close monitoring recommended'
            }
        else:
            return {
                'level': 'low',
                'label': 'LOW RISK',
                'description': 'Normal development pattern; routine follow-up'
            }
    
    @staticmethod
    def get_follow_up_recommendation(risk_level, class_label):
        """Get follow-up recommendation based on risk level and diagnosis."""
        recommendations = {
            'low': 'Continue routine prenatal care with ultrasound in 4-6 weeks',
            'moderate': 'Close monitoring with follow-up ultrasound in 1-2 weeks',
            'high': 'Urgent specialist consultation; frequent monitoring (2-3 times weekly recommended)'
        }
        return recommendations.get(risk_level, 'Follow-up ultrasound recommended')


class PerformanceMonitor:
    """Monitor and log system performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, event_name):
        """Start timing an event."""
        self.metrics[event_name] = {'start': datetime.now()}
    
    def end_timer(self, event_name):
        """End timing an event and calculate duration."""
        if event_name in self.metrics:
            self.metrics[event_name]['end'] = datetime.now()
            duration = (self.metrics[event_name]['end'] - self.metrics[event_name]['start']).total_seconds()
            self.metrics[event_name]['duration_ms'] = duration * 1000
            return duration * 1000
        return None
    
    def get_metrics(self):
        """Get all metrics."""
        return self.metrics
    
    def log_metrics(self):
        """Log all metrics to logger."""
        for event, data in self.metrics.items():
            if 'duration_ms' in data:
                logger.info(f"{event}: {data['duration_ms']:.2f}ms")


class SystemHealth:
    """System health checks."""
    
    @staticmethod
    def check_memory_available():
        """Check if sufficient memory is available."""
        import psutil
        try:
            mem = psutil.virtual_memory()
            return mem.available >= 500 * 1024 * 1024  # At least 500MB
        except:
            return True  # Assume OK if can't check
    
    @staticmethod
    def check_model_files():
        """Verify model files exist."""
        from pathlib import Path
        required_files = [
            'rag/knowledge_base.json',
            'templates/index.html'
        ]
        
        missing = []
        for file in required_files:
            if not Path(file).exists():
                missing.append(file)
        
        return len(missing) == 0, missing
    
    @staticmethod
    def get_system_info():
        """Get system information."""
        import platform
        import torch
        
        return {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': 'CUDA' if torch.cuda.is_available() else 'CPU'
        }
