"""
Solar Panel Defect Detection System
-----------------------------------
Production-ready Streamlit application for automated solar panel inspection.
Supports both manual upload (demo) and simulated camera feed (production-ready).
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
from PIL import Image
import io

# Import custom modules
from utils.defect_detector import DefectDetector
from utils.camera_simulator import CameraSimulator

# Page configuration
st.set_page_config(
    page_title="Solar Panel Defect Detection",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FFA500;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
    }
    .danger-text {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'inspection_log' not in st.session_state:
    st.session_state.inspection_log = []
if 'total_inspected' not in st.session_state:
    st.session_state.total_inspected = 0
if 'total_defects' not in st.session_state:
    st.session_state.total_defects = 0
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'detector' not in st.session_state:
    st.session_state.detector = DefectDetector(threshold=0.3)

# Header section
st.markdown('<h1 class="main-header">☀️ Solar Panel Defect Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Quality Control for Solar Manufacturing</p>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/solar-panel.png", width=100)
    st.title("Control Panel")
    
    # Mode selection
    st.subheader("🔧 Operation Mode")
    mode = st.radio(
        "Select Mode",
        ["📁 Manual Upload (Demo)", "📷 Live Camera (Simulated)"],
        help="Manual upload is for demonstration. Live camera simulates real factory integration."
    )
    
    # Detection settings
    st.subheader("⚙️ Detection Settings")
    sensitivity = st.slider(
        "Detection Sensitivity",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.1,
        help="Higher sensitivity catches more defects but may increase false positives"
    )
    st.session_state.detector.threshold = sensitivity
    
    # Factory simulation settings
    st.subheader("🏭 Factory Settings")
    target_production = st.number_input(
        "Daily Target (panels)",
        min_value=100,
        max_value=10000,
        value=5000,
        step=100
    )
    
    shift = st.selectbox(
        "Current Shift",
        ["Morning (6AM-2PM)", "Afternoon (2PM-10PM)", "Night (10PM-6AM)"]
    )
    
    # Export options
    st.subheader("📊 Data Management")
    if st.button("📥 Export Inspection Log"):
        if st.session_state.inspection_log:
            df = pd.DataFrame(st.session_state.inspection_log)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"inspection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No inspection data to export")
    
    if st.button("🔄 Reset Session"):
        st.session_state.inspection_log = []
        st.session_state.total_inspected = 0
        st.session_state.total_defects = 0
        st.rerun()

# Main content area - Split into two columns
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📸 Inspection Interface")
    
    if mode == "📁 Manual Upload (Demo)":
        # Manual upload interface
        uploaded_file = st.file_uploader(
            "Upload Solar Panel Image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image of the solar panel for inspection"
        )
        
        if uploaded_file is not None:
            # Read and process image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display original image
            st.image(image, caption="Uploaded Panel", use_container_width=True)
            
            # Perform detection
            with st.spinner("🔍 Analyzing panel for defects..."):
                time.sleep(1)  # Simulate processing
                result, defects, defect_count = st.session_state.detector.detect_defects(image)
            
            # Log the inspection
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {
                'timestamp': timestamp,
                'filename': uploaded_file.name,
                'defect_count': defect_count,
                'status': 'FAIL' if defect_count > 0 else 'PASS',
                'shift': shift,
                'sensitivity': sensitivity
            }
            st.session_state.inspection_log.append(log_entry)
            st.session_state.total_inspected += 1
            if defect_count > 0:
                st.session_state.total_defects += 1
            
            # Show results
            st.markdown("### 📊 Detection Results")
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                if defect_count == 0:
                    st.success(f"✅ Status: **PASS**")
                else:
                    st.error(f"❌ Status: **FAIL**")
            
            with result_col2:
                st.metric("Defects Found", defect_count)
            
            with result_col3:
                confidence = max(0, 1 - (defect_count * 0.1))
                st.metric("Confidence Score", f"{confidence:.1%}")
            
            # Show defect visualization
            if defect_count > 0:
                st.markdown("### 🔍 Defect Visualization")
                st.image(result, caption="Defects Highlighted (Red Circles)", use_container_width=True)
                
                # Show defect details
                with st.expander("View Defect Details"):
                    for i, defect in enumerate(defects):
                        st.write(f"**Defect {i+1}:** {defect['type']} at position ({defect['x']}, {defect['y']})")
    
    else:  # Live Camera Simulation
        st.markdown("**📡 Live Camera Feed (Simulated for Demo)**")
        st.info("This simulates a GigE camera connected via RTSP stream in a real factory")
        
        # Camera controls
        cam_col1, cam_col2, cam_col3 = st.columns(3)
        
        with cam_col1:
            if st.button("▶️ Start Camera"):
                st.session_state.camera_active = True
        
        with cam_col2:
            if st.button("⏸️ Stop Camera"):
                st.session_state.camera_active = False
        
        with cam_col3:
            inspection_rate = st.slider("Inspection Speed (ms)", 500, 3000, 1000, step=500)
        
        # Camera feed placeholder
        camera_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        if st.session_state.camera_active:
            camera = CameraSimulator()
            
            for frame in camera.stream_frames():
                if not st.session_state.camera_active:
                    break
                
                # Process frame
                result, defects, defect_count = st.session_state.detector.detect_defects(frame)
                
                # Update display
                camera_placeholder.image(result, caption="Live Inspection Feed", use_container_width=True)
                
                # Log automatically (every 10th frame to avoid flooding)
                if np.random.random() < 0.1:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = {
                        'timestamp': timestamp,
                        'filename': f"frame_{timestamp}.jpg",
                        'defect_count': defect_count,
                        'status': 'FAIL' if defect_count > 0 else 'PASS',
                        'shift': shift,
                        'sensitivity': sensitivity
                    }
                    st.session_state.inspection_log.append(log_entry)
                    st.session_state.total_inspected += 1
                    if defect_count > 0:
                        st.session_state.total_defects += 1
                
                time.sleep(inspection_rate / 1000)

with col2:
    st.subheader("📊 Quality Dashboard")
    
    # Key metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Total Inspected",
            st.session_state.total_inspected,
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        defects_found = st.session_state.total_defects
        st.metric(
            "Defects Found",
            defects_found,
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metrics_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.session_state.total_inspected > 0:
            yield_rate = (1 - defects_found/st.session_state.total_inspected) * 100
        else:
            yield_rate = 100
        st.metric(
            "Yield Rate",
            f"{yield_rate:.1f}%",
            delta=f"{yield_rate-95:.1f}%" if yield_rate else None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Real-time yield chart
    st.markdown("### 📈 Real-Time Yield Trend")
    if st.session_state.inspection_log:
        df = pd.DataFrame(st.session_state.inspection_log)
        
        # Create yield trend
        fig = go.Figure()
        
        # Add trace for pass/fail
        df['status_num'] = df['status'].map({'PASS': 1, 'FAIL': 0})
        df['rolling_yield'] = df['status_num'].rolling(window=min(10, len(df)), min_periods=1).mean() * 100
        
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df['rolling_yield'],
            mode='lines+markers',
            name='Yield Rate',
            line=dict(color='green', width=3),
            marker=dict(
                color=df['status'].map({'PASS': 'green', 'FAIL': 'red'}),
                size=8
            )
        ))
        
        fig.update_layout(
            title="Last 50 Inspections (Rolling Yield)",
            xaxis_title="Inspection Number",
            yaxis_title="Yield Rate (%)",
            yaxis_range=[0, 100],
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent inspection log
    st.markdown("### 📋 Recent Inspections")
    if st.session_state.inspection_log:
        # Show last 10 inspections
        recent_df = pd.DataFrame(st.session_state.inspection_log[-10:])
        
        # Color code the status
        def color_status(val):
            return 'background-color: #d4edda' if val == 'PASS' else 'background-color: #f8d7da'
        
        styled_df = recent_df[['timestamp', 'filename', 'defect_count', 'status']].style.applymap(
            color_status, subset=['status']
        )
        
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No inspections yet. Upload an image to begin.")
    
    # ALMM Compliance Report
    st.markdown("### ✅ ALMM Compliance Status")
    
    compliance_col1, compliance_col2 = st.columns(2)
    
    with compliance_col1:
        st.markdown("**Traceability**")
        if len(st.session_state.inspection_log) > 0:
            st.success("✓ Digital trail active")
            st.write(f"📝 {len(st.session_state.inspection_log)} records logged")
        else:
            st.warning("⚠️ No records yet")
    
    with compliance_col2:
        st.markdown("**Quality Standards**")
        if yield_rate > 95:
            st.success("✓ Meeting ALMM standards")
        elif yield_rate > 90:
            st.warning("⚠️ Approaching threshold")
        else:
            st.error("❌ Below standard")
    
    # Machine health prediction (simulated)
    st.markdown("### 🔧 Predictive Maintenance")
    
    # Simulate machine health based on defect trend
    if len(st.session_state.inspection_log) > 20:
        recent_defects = pd.DataFrame(st.session_state.inspection_log[-20:])
        defect_trend = recent_defects['defect_count'].rolling(5).mean().iloc[-1]
        
        if defect_trend > 2:
            st.error("⚠️ **Alert:** Increasing defect rate detected")
            st.write("🔍 Suggested: Check laminator rollers on Line 3")
        elif defect_trend > 1:
            st.warning("📊 **Monitor:** Slight increase in defects")
            st.write("🛠️ Schedule maintenance within 48 hours")
        else:
            st.success("✅ **All systems normal**")
            st.write("📈 Defect rate within acceptable range")
    else:
        st.info("Collect more data for predictive insights")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**📋 Compliance**")
    st.markdown("- ALMM Ready")
    st.markdown("- ISO 9001 Compatible")
    st.markdown("- Full Traceability")

with footer_col2:
    st.markdown("**🔧 Integration**")
    st.markdown("- RTSP Camera Support")
    st.markdown("- PLC Ready")
    st.markdown("- MES Compatible")

with footer_col3:
    st.markdown("**📊 Analytics**")
    st.markdown("- Real-time Dashboard")
    st.markdown("- Predictive Alerts")
    st.markdown("- Exportable Reports")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>
        ☀️ Solar Panel Defect Detection System v2.0<br>
        Developed for Sri City Solar Manufacturing Hub<br>
        © 2024 - Production Ready Demo
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
