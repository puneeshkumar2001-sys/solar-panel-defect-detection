import streamlit as st
import sys
import traceback

st.set_page_config(page_title="Debug Mode")
st.title("🔍 Debug Version")

st.write("Python version:", sys.version)
st.write("Streamlit version:", st.__version__)

# Try importing each package separately
packages = {
    "numpy": "numpy",
    "OpenCV": "cv2", 
    "Pandas": "pandas",
    "PIL": "PIL",
    "Plotly": "plotly"
}

for name, import_name in packages.items():
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        st.success(f"✅ {name} {version} loaded")
    except Exception as e:
        st.error(f"❌ {name} failed: {str(e)}")
        st.code(traceback.format_exc())

st.write("---")
st.write("If you see this, the app is running!")
