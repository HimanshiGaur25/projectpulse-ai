import importlib

import streamlit as st

st.set_page_config(
    page_title="ProjectPulse AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Importing the module runs the Streamlit UI defined in ui.py.
importlib.import_module("ui")
