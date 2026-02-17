import streamlit as st
import subprocess
import sys
import importlib

st.set_page_config(page_title="Solar Panel Inspector", layout="wide")

st.title("☀️ Solar Panel Defect Detection")
st.write("Installing packages one by one...")

# List of packages to install
packages = [
    "numpy==1.23.5",
    "opencv-python-headless==4.7.0.68",
    "pandas==1.5.3", 
    "Pillow==9.4.0",
    "plotly==5.14.1"
]

# Install each package individually
for package in packages:
    package_name = package.split('==')[0]
    
    # Create a placeholder for status
    status = st.empty()
    status.info(f"📦 Installing {package_name}...")
    
    try:
        # Try to import first (check if already installed)
        importlib.import_module(package_name.replace('-', '_'))
        status.success(f"✅ {package_name} already installed")
    except ImportError:
        # Install the package
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", package],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            status.success(f"✅ {package_name} installed successfully")
        else:
            status.error(f"❌ {package_name} FAILED to install")
            st.code(result.stderr)
            st.stop()  # Stop if any package fails

st.success("✅ All packages installed!")
st.write("---")
st.write("🎉 App is ready! Loading main interface...")
st.rerun()
