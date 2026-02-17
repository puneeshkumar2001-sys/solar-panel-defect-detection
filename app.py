import streamlit as st
import subprocess
import sys

# Page setup
st.set_page_config(page_title="Simple Test")
st.title("🚀 Streamlit Test App")

# Try to import numpy
try:
    import numpy as np
    st.success("✅ NumPy is working!")
except:
    st.warning("📦 Installing NumPy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np
    st.success("✅ NumPy installed!")

# Try to import cv2
try:
    import cv2
    st.success("✅ OpenCV is working!")
except:
    st.warning("📦 Installing OpenCV...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2
    st.success("✅ OpenCV installed!")

st.write("---")
st.write("🎉 App is running!")
st.write(f"Python version: {sys.version}")
