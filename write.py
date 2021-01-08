"""Simple streamlit app to learn writing digits."""


import json
import random

import numpy as np
import streamlit as st
import torch
from ftfy import fix_file
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
    empty_img = np.full((NN_IMG_SIZE, NN_IMG_SIZE), 0.9999)
    if np.allclose(small_img, empty_img):
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


def fix_file_encoding(in_file, out_file):
    """Fix unicode encoding to ensure proper display."""
    stream = fix_file(
        in_file,
        encoding=None,
        fix_entities=False,
        remove_terminal_escapes=False,
        fix_encoding=True,
        fix_latin_ligatures=False,
        fix_character_width=False,
        uncurl_quotes=False,
        fix_line_breaks=False,
        fix_surrogates=False,
        remove_control_chars=False,
        remove_bom=False,
        normalization="NFC",
    )
    stream_iterator = iter(stream)
    while stream_iterator:
        try:
            line = next(stream_iterator)
            out_file.write(line)
        except StopIteration:
            break
    output_file.close()


if __name__ == "__main__":
    # Session initialization
    session = get(count=1, expected="", successes=0)
    with st.spinner("Loading neural network..."):
        mnist = Net().to("cpu")
        mnist.load_state_dict(torch.load("models/mnist_cnn.pt"))

    # Read text file
    input_file = open("texts.json", "r")
    output_file = open("texts_unicode.json", "w")
    fix_file_encoding(input_file, output_file)
    with open("texts_unicode.json") as json_file:
        texts = json.load(json_file)
    languages = texts["languages"]

    # language choice
    language = st.sidebar.radio(" ", list(languages.keys()))
    text = texts[languages[language]]

    # target choice
    target = st.sidebar.selectbox(text["what"], (text["digits"], text["letters"]))
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
    change = st.button(text["next"][target])

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
    res_text.text(text["target"].format(session.expected))

    # Display score
    if session.successes > 0:
        if session.successes == 1:
            score.write(text["score1"][target].format(session.successes))
        else:
            score.write(text["score"][target].format(session.successes))

    # Process drawing
    if CanvasResult.image_data is not None:
        std_img = CanvasResult.image_data / 255
        res = submit_img(std_img, mnist)

        if res is None:
            st.stop()
        else:
            res_text.text(text["result"].format(res, session.expected))
            if res == session.expected:
                st.balloons()
                session.expected = get_digit()
                session.successes += 1
                if session.successes == 1:
                    score.write(text["score1"][target].format(session.successes))
                else:
                    score.write(text["score"][target].format(session.successes))
