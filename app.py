"""
Stryktipset Predictor - Main Dashboard
"""

import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Stryktipset Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚽ Stryktipset Predictor</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Football Match Predictions")
st.markdown("---")

# Welcome section
col1, col2, col3 = st.columns(3)

with col1:
    st.info("### 📊 Data Manager\nFetch and explore football data from API-Football")
    if st.button("Go to Data Manager", use_container_width=True):
        st.switch_page("pages/2_📊_Data_Manager.py")

with col2:
    st.success("### 🤖 Train Models\nTrain ML models with custom parameters")
    if st.button("Go to Model Training", use_container_width=True):
        st.switch_page("pages/3_🤖_Train_Models.py")

with col3:
    st.warning("### 🎯 Predictions\nPredict matches and generate coupons")
    if st.button("Go to Predictions", use_container_width=True):
        st.switch_page("pages/1_🎯_Predictions.py")

st.markdown("---")

# Quick stats
st.subheader("📈 System Status")

col1, col2, col3, col4 = st.columns(4)

# Check data availability
data_dir = Path("data/raw")
data_files = list(data_dir.glob("*.json"))

# Check models
model_dir = Path("models")
model_files = list(model_dir.glob("*.pkl"))

with col1:
    st.metric("Data Files", len(data_files))

with col2:
    st.metric("Trained Models", len(model_files))

with col3:
    leagues_count = len(set([f.stem.rsplit('_', 1)[0] for f in data_files]))
    st.metric("Leagues Available", leagues_count)

with col4:
    from config import API_FOOTBALL_KEY
    api_status = "✅ Configured" if API_FOOTBALL_KEY != "your_api_key_here" else "❌ Not Set"
    st.metric("API Status", api_status)

# Recent activity
st.subheader("📋 Quick Start Guide")

st.markdown("""
1. **Configure API** - Set your API-Football key in `.env` or `config/config.py`
2. **Fetch Data** - Go to Data Manager to download match data
3. **Train Model** - Go to Model Training to create prediction models
4. **Make Predictions** - Go to Predictions to generate your Stryktipset coupon

### Need Help?
- Check the sidebar for navigation
- Each page has its own help section
- Visit [Documentation](https://github.com/johedvi/stryktipset-predictor) for details
""")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'><p>Made with ⚽ and 🤖 | Powered by API-Football</p></div>",
    unsafe_allow_html=True
)