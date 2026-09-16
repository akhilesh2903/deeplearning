#!/usr/bin/env python
"""
Quick Start Script
==================
Simplifies the process of running the application.
"""

import sys
import os
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_pip_packages():
    """Check if required packages are installed."""
    required_packages = [
        'flask',
        'torch',
        'torchvision',
        'sentence_transformers',
        'faiss',
        'Pillow'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('_', ''))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        return False
    
    print("✓ All required packages installed")
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
            print(f"✓ {directory}/")
        else:
            print(f"⚠ Creating {directory}/")
            Path(directory).mkdir(exist_ok=True)
            all_exist = False
    
    return True


def main():
    """Main startup function."""
    print("\n" + "="*80)
    print("🏥 AI-Enhanced Fetal Ultrasound Diagnosis System")
    print("="*80 + "\n")
    
    print("📋 Performing startup checks...\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check packages
    if not check_pip_packages():
        print("\n📦 Installing missing packages...")
        print("   Run: pip install -r requirements.txt\n")
        sys.exit(1)
    
    # Check directories
    print("\n📁 Checking directories...")
    check_directories()
    
    print("\n✅ All checks passed!\n")
    
    print("🚀 Starting Flask application...\n")
    print("-" * 80)
    print("Server will be available at:")
    print("   🌐 http://localhost:5000")
    print("   🌐 http://127.0.0.1:5000")
    print("-" * 80 + "\n")
    
    # Import and run Flask app
    try:
        from app import app
        
        print("\n💡 Tips:")
        print("   • Upload ultrasound images in PNG, JPG, or GIF format")
        print("   • Maximum file size: 50MB")
        print("   • Press Ctrl+C to stop the server")
        print("   • Open http://localhost:5000 in your browser\n")
        
        # Run on all interfaces
        app.run(debug=True, host='0.0.0.0', port=5000)
    
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        print("   Make sure app.py exists in the current directory")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
