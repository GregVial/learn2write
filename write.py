"""Simple streamlit app to learn writing digits."""


import json
import random

import numpy as np
import streamlit as st
import torch
from ftfy import fix_encoding as f_e
from streamlit_drawable_canvas import st_canvas

from mnist import Net
from session_state import get

CANVA_SIZE = 252
NN_IMG_SIZE = 28


def resize(img, in_size, out_size):
    """Resize image."""
    bin_size = in_size // out_size
    small_image = img.reshape((out_size, bin_size, out_size, bin_size)).mean(3).mean(1)
    return small_image


def to_grayscale(img):
    """Turn RGB image to grayscale."""
    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_img = np.dot(img[..., :3], rgb_weights)
    return grayscale_img


def submit_img(img, net):
    """Preprocess image and call neural net for prediction."""
    grayscale_img = to_grayscale(img)
    small_img = resize(grayscale_img, CANVA_SIZE, NN_IMG_SIZE)
    ref = np.full((NN_IMG_SIZE, NN_IMG_SIZE), 0.9999)
    if np.allclose(small_img, ref):
        return None
    normalized_img = (small_img - 0.5) / 0.5
    inverted_img = -normalized_img
    torch_ready_img = inverted_img[np.newaxis, np.newaxis, :]
    torch_img = torch.from_numpy(torch_ready_img).float()
    logits = net(torch_img).detach().numpy()
    return np.argmax(logits)


def get_digit():
    """Choose a random digit."""
    digit = random.choice(range(10))
    return digit


if __name__ == "__main__":
    # Session initialization
    session = get(count=1, expected="", successes=0)
    with st.spinner("Loading neural network..."):
        mnist = Net()
        mnist.load_state_dict(torch.load("models/mnist_cnn.pt"))

    # Read text file
    with open("texts.json") as json_file:
        texts = json.load(json_file)
    languages = texts["languages"]

    # language choice
    language_list = [f_e(l) for l in languages.keys()]
    language = st.sidebar.selectbox(" ", language_list)
    text = texts[language[:2]]

    # target choice
    target = st.sidebar.selectbox(
        f_e(text["what"]), (f_e(text["digits"]), f_e(text["letters"]))
    )
    if target == text["digits"]:
        target = "digits"
    else:
        target = "letters"

    # Create canva
    CanvasResult = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=20,
        stroke_color="#D01111",
        background_color="#fff",
        background_image=None,
        update_streamlit=True,
        height=CANVA_SIZE,
        width=CANVA_SIZE,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Empty slot for expected input
    res_text = st.empty()

    # Next input button
    change = st.button(f_e(text["next"][target]))

    # Empty slot for score
    score = st.empty()

    # Logical flow
    # Get input to be drawn
    if session.count == 1:
        session.count += 1
        session.expected = get_digit()
    elif change:
        session.expected = get_digit()

    # Display letter
    res_text.text(f_e(text["target"].format(session.expected)))

    # Display score
    if session.successes > 0:
        if session.successes == 1:
            score.write(f_e(text["score1"][target].format(session.successes)))
        else:
            score.write(f_e(text["score"][target].format(session.successes)))

    # Process drawing
    if CanvasResult.image_data is not None:
        std_img = CanvasResult.image_data / 255
        res = submit_img(std_img, mnist)

        if res is None:
            st.stop()
        else:
            res_text.text(f_e(text["result"].format(res, session.expected)))
            if res == session.expected:
                st.balloons()
                session.expected = get_digit()
                session.successes += 1
                if session.successes == 1:
                    score.write(f_e(text["score1"][target].format(session.successes)))
                else:
                    score.write(f_e(text["score"][target].format(session.successes)))
