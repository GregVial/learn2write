"""This script allows learning to write letters."""

import base64
import json
import random
import string
from io import BytesIO

import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

URL = "http://127.0.0.1:5002/api/word"


def submit_img(img, lang="EN", URL=URL):
    """Submit a b64 encoded image and get letter back."""
    myobj = {"img": img, "word": "", "lang": "en", "nb_output": 1}
    x = requests.post(URL, data=json.dumps(myobj))
    return x.text


def rgb2gray(rgb):
    """RGB image to grayscale."""
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# with open('text.json') as json_file:
#     text = json.load(json_file)

# language = st.sidebar.selectbox("Choose your language", \
#  tuple(text['language']))
# stroke_color = st.sidebar.color_picker("Stroke color hex: ")


@st.cache(ttl=5)
def get_letter():
    letter = random.choice(string.ascii_uppercase)
    return letter


letter = get_letter()
st.text("Write the letter {}".format(letter))


canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=5,
    stroke_color="#D01111",  # "#000",
    background_color="#fff",
    background_image=None,
    update_streamlit=True,
    height=250,
    width=250,
    drawing_mode="freedraw",
    key="canvas",
)

erase = st.button("Erase")
submit = st.button("Submit")

if canvas_result.image_data is not None:
    img = canvas_result.image_data[..., :3]
    img = rgb2gray(img)
    img = img.astype(np.uint8)

    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")

    b64img = base64.b64encode(buff.getvalue()).decode("utf-8")

    # Need to check why double json loads is required!
    js = json.loads(json.loads(submit_img(b64img)))
    res = js["word"]

    st.text("Letter found: {}, expected: {}".format(res, letter))
    if res == letter:

        st.balloons()
