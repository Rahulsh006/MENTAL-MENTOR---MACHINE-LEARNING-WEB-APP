import streamlit as st
from streamlit_lottie import st_lottie
import requests
import json

def load_lottiefile(file_path):
    with open(file_path, 'r') as file:
        lottie_content = file.read()
    return lottie_content

def load_lottiefile_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        st.error(f"Failed to fetch Lottie file. Status code: {r.status_code}")
        return None
    try:
        return r.json()
    except ValueError as e:
        st.error(f"Failed to parse JSON: {e}")
        return None



lottie_hello = load_lottiefile(r"pages\image.json")



st.title("Its Just Your Mind")


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.title("Its Just Your Mind")
st.write("Stress is any demand placed on your brain or physical body. Any event or scenario that makes you feel frustrated or nervous can trigger it.Anxiety is a feeling of fear, worry, or unease. While it can occur as a reaction to stress, it can also happen without any obvious trigger.Both stress and anxiety involve mostly identical symptoms, including:")
st.write("trouble sleeping")
st.write("digestive issues")
st.write("difficulty concentrating")
st.write("muscle tension")
st.write("irritability or anger")
st.write("Most people experience some feelings of stress and anxiety at some point, and that isn’t necessarily a “bad” thing. After all, stress and anxiety can sometimes be a helpful motivator to accomplish daunting tasks or do things you’d rather not (but really should).")
st.write("But unmanaged stress and anxiety can start to interfere with your daily life and take a toll on your mental and physical health.")



