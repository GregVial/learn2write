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

from session_state import get

URL = "http://127.0.0.1:5002/api/word"

BLANK = (
    "iVBORw0KGgoAAAANSUhEUgAAAPoAAAD6AQAAAACgl2eQAAAAOklEQVR4nO3KsQ"
    + "0AMAzDMLf/P5XL0g+8F6BW8Wxqc/tPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD4"
    + "FzzqWAO0JciZyAAAAABJRU5ErkJggg=="
)


def submit_img(img, lang="en", url=URL):
    """Submit a b64 encoded image and get letter back."""
    if img == BLANK:
        return json.dumps(json.dumps({"word": None}))
    myobj = {"img": img, "word": "", "lang": lang, "nb_output": 1}
    req = requests.post(url, data=json.dumps(myobj))
    st.write(req.text)
    return req.text


def img2b64(img):
    """Converts a color image to black&white b64."""
    img = img[..., :3]
    img = img.astype(np.uint8)

    pil_img = Image.fromarray(img)
    pil_img = pil_img.convert("1")
    buff = BytesIO()
    pil_img.save(buff, format="PNG")

    return base64.b64encode(buff.getvalue()).decode("utf-8")


def get_letter():
    """Choose a random upper case letter."""
    letter = random.choice(string.ascii_uppercase)
    return letter


# with open('text.json') as json_file:
#     text = json.load(json_file)

# language = st.sidebar.selectbox("Choose your language", \
#  tuple(text['language']))
# stroke_color = st.sidebar.color_picker("Stroke color hex: ")


if __name__ == "__main__":
    # Session initialization
    session = get(count=1, letter="", successes=0)

    # Create canva
    CanvasResult = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=10,
        stroke_color="#D01111",  # "#000",
        background_color="#fff",
        background_image=None,
        update_streamlit=True,
        height=250,
        width=250,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Empty slot for expected letter
    res_text = st.empty()

    # Next letter button
    change = st.button("Next letter")

    # Empty slot for score
    score = st.empty()

    # Logical flow
    # Get letter to be drawn
    if session.count == 1:
        session.count += 1
        session.letter = get_letter()
    elif change:
        session.letter = get_letter()

    # Display letter
    res_text.text("Please draw a {}".format(session.letter))

    # Display score
    if session.successes:
        score.write("You correctly draw {} letters".format(session.successes))

    # Process drawing
    if CanvasResult.image_data is not None:
        b64img = img2b64(CanvasResult.image_data)
        js = json.loads(json.loads(submit_img(b64img)))
        user_letter = js["word"]

        if user_letter is None:
            st.stop()
        else:
            res_text.text(
                "Letter found: {}, expected: {}".format(user_letter, session.letter)
            )
            if user_letter == session.letter:
                st.balloons()
                session.successes += 1
                score.write("You correctly draw {} letters".format(session.successes))
