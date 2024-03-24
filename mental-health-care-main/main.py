import streamlit as st
st.set_page_config( page_title = "this is a multipage")
st.title("homepage")
st.sidebar.success("Let's go to the....")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 