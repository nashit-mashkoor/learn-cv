import base64
import os
import streamlit as st
from pathlib import Path

# Custom imports 
from multiapp import MultiApp
from apps import home, basic

# Helper Functions
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



st.set_page_config(page_title='OpenCV', page_icon='📷', layout='wide', initial_sidebar_state='collapsed')

# Load css styles
with open("assets/style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# Create an instance of the app 
app = MultiApp()
# st.set_page_config()
# Title of the main page
_, mid, _ = st.columns(3)
with mid:
    st.title("Power of OpenCV 📷")
# Add all your applications (pages) here
app.add_page("Home", home.app)
app.add_page("Basics", basic.app)

# The main app
app.run()