#!/usr/bin/env python
"""
Quick Start Script
==================
Simplifies the process of running the application.
"""

import sys
# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import os
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERROR] Python 3.8+ required")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_pip_packages():
    """Check if required packages are installed."""
    # Map: pip package name -> actual import module name
    required_packages = {
        'flask': 'flask',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'sentence_transformers': 'sentence_transformers',
        'faiss-cpu': 'faiss',
        'Pillow': 'PIL',
    }

    missing = []
    for pip_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        return False

    print("[OK] All required packages installed")
    return True


def check_directories():
    """Check if required directories exist."""
    directories = [
        'model',
        'rag',
        'llm',
        'templates',
        'static',
        'uploads'
    ]
    
    all_exist = True
    for directory in directories:
        if Path(directory).exists():
            print(f"[OK] {directory}/")
        else:
            print(f"[!] Creating {directory}/")
            Path(directory).mkdir(exist_ok=True)
            all_exist = False
    
    return True


def main():
    """Main startup function."""
    print("\n" + "="*80)
    print("[*] AI-Enhanced Fetal Ultrasound Diagnosis System")
    print("="*80 + "\n")
    
    print("[*] Performing startup checks...\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check packages
    if not check_pip_packages():
        print("\n📦 Installing missing packages...")
        print("   Run: pip install -r requirements.txt\n")
        sys.exit(1)
    
    # Check directories
    print("\n[*] Checking directories...")
    check_directories()
    
    print("\n[OK] All checks passed!\n")
    
    print("[>>] Starting Flask application...\n")
    print("-" * 80)
    print("Server will be available at:")
    print("   http://localhost:5000")
    print("   http://127.0.0.1:5000")
    print("-" * 80 + "\n")
    
    # Import and run Flask app
    try:
        from app import app
        
        print("\n[Tips]:")
        print("   * Upload ultrasound images in PNG, JPG, or GIF format")
        print("   * Maximum file size: 50MB")
        print("   * Press Ctrl+C to stop the server")
        print("   * Open http://localhost:5000 in your browser\n")
        
        # Run on all interfaces
        app.run(debug=True, host='0.0.0.0', port=5000)
    
    except ImportError as e:
        print(f"[ERROR] Error importing app: {e}")
        print("   Make sure app.py exists in the current directory")
        sys.exit(1)
    
    except Exception as e:
        print(f"[ERROR] Error starting application: {e}")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[STOP] Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL] Fatal error: {e}")
        sys.exit(1)
