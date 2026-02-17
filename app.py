import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

# Page setup
st.set_page_config(page_title="Solar Panel Checker", page_icon="☀️")
st.title("☀️ Solar Panel Defect Checker")
st.write("Upload a solar panel photo to check for defects")

# Initialize counters
if 'total' not in st.session_state:
    st.session_state.total = 0
    st.session_state.passed = 0
    st.session_state.failed = 0

# Upload photo
uploaded = st.file_uploader("Choose a solar panel photo", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    # Show original photo
    image = Image.open(uploaded)
    st.image(image, caption="Original Photo", use_container_width=True)
    
    # Convert to format OpenCV can use
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Simple defect detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Find edges/defects
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw defects on photo
    result = img_array.copy()
    defect_count = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Ignore tiny spots
            defect_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Show result
    st.subheader("Result")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if defect_count == 0:
            st.success("✅ PASS - No defects found")
            st.session_state.passed += 1
        else:
            st.error(f"❌ FAIL - Found {defect_count} defect(s)")
            st.session_state.failed += 1
    
    with col2:
        # Show image with defects circled
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="Defects Circled in Red", use_container_width=True)
    
    st.session_state.total += 1

# Show statistics
st.markdown("---")
st.subheader("📊 Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Total Checked", st.session_state.total)
col2.metric("Passed", st.session_state.passed)
col3.metric("Failed", st.session_state.failed)

# Reset button
if st.button("Reset Counters"):
    st.session_state.total = 0
    st.session_state.passed = 0
    st.session_state.failed = 0
    st.rerun()

st.markdown("---")
st.caption("Simple Solar Panel Defect Checker - Phase 1")
