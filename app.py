import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page config
st.set_page_config(page_title="Solar Panel Inspector", layout="wide")

# Title
st.title("☀️ Solar Panel Defect Detection")
st.write("Simple and Working Version")

# Sidebar
with st.sidebar:
    st.header("Controls")
    sensitivity = st.slider("Sensitivity", 0.1, 0.9, 0.3)
    
    if st.button("Clear Log"):
        if 'log' in st.session_state:
            st.session_state.log = []

# Initialize session state
if 'log' not in st.session_state:
    st.session_state.log = []
if 'total' not in st.session_state:
    st.session_state.total = 0
if 'defects' not in st.session_state:
    st.session_state.defects = 0

# Main area
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📸 Upload Panel Image")
    uploaded = st.file_uploader("Choose image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded is not None:
        # Read image
        image = Image.open(uploaded)
        img_array = np.array(image)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Show original
        st.image(image, caption="Uploaded Panel", use_container_width=True)
        
        # Simple defect detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        min_area = img_array.shape[0] * img_array.shape[1] * 0.001
        defects_found = 0
        result = img_array.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                defects_found += 1
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Log result
        st.session_state.total += 1
        if defects_found > 0:
            st.session_state.defects += 1
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.log.append({
            'time': timestamp,
            'defects': defects_found,
            'status': 'FAIL' if defects_found > 0 else 'PASS'
        })
        
        # Show result
        if defects_found > 0:
            st.error(f"❌ Defects Found: {defects_found}")
            st.image(result, caption="Defects Highlighted", use_container_width=True)
        else:
            st.success("✅ Panel Passed")
            st.image(result, caption="No Defects Found", use_container_width=True)

with col2:
    st.subheader("📊 Dashboard")
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Inspected", st.session_state.total)
    with m2:
        st.metric("Defects", st.session_state.defects)
    with m3:
        yield_rate = 100
        if st.session_state.total > 0:
            yield_rate = ((st.session_state.total - st.session_state.defects) / st.session_state.total) * 100
        st.metric("Yield", f"{yield_rate:.1f}%")
    
    # Recent log
    st.subheader("📋 Recent")
    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log[-5:])
        st.dataframe(df, use_container_width=True)
    
    # Simple chart
    if len(st.session_state.log) > 1:
        log_df = pd.DataFrame(st.session_state.log)
        pass_fail = log_df['status'].value_counts()
        fig = go.Figure(data=[go.Bar(x=pass_fail.index, y=pass_fail.values)])
        fig.update_layout(height=200, title="Pass/Fail Distribution")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Simple Solar Panel Inspector v1.0")
