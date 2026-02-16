# ☀️ Solar Panel Defect Detection System

<div align="center">
  
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**Production-Ready AI Inspection System for Solar Panel Manufacturing**

<img src="https://img.icons8.com/color/96/000000/solar-panel.png" width="120">

</div>

## 🎯 The Problem We Solve

In high-volume solar manufacturing hubs like **Sri City**, manual inspection creates critical bottlenecks:

| Challenge | Impact |
|-----------|--------|
| 👁️ **Human Fatigue** | Inspectors miss 15-20% of defects after 8-hour shifts |
| 📏 **Inconsistency** | Different inspectors = different quality standards |
| 📝 **No Digital Trail** | Paper logs can't provide ALMM compliance traceability |
| ⏱️ **Speed Limit** | Manual inspection caps production line speed at source |

## 💡 Our Solution

An **AI-Powered Automated Inspection System** that acts as an "Always-On Digital Inspector":

```mermaid
graph LR
    A[Camera Feed] --> B[OpenCV Detection]
    B --> C{Defect Found?}
    C -->|Yes| D[Alert + Log + Reject]
    C -->|No| E[Pass + Log]
    D --> F[Live Dashboard]
    E --> F
    F --> G[Analytics + Compliance]
