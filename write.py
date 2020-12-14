"""This script allows learning to write letters."""

import streamlit as st

language = st.sidebar.selectbox("Choose your language", ("Fançais", "English"))

if language == "English":
    st.balloons()
