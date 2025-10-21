import streamlit as st
from apputil import MarkovText
import requests
import re

# --- Page setup ---
st.set_page_config(page_title="Markov Text Generator", layout="centered")

st.write(
    """
    # Week 8: Markov Chains & Text Generation

    This app demonstrates a simple Markov chain text generator using inspirational quotes.
    Enter how many words you'd like the generator to produce, and optionally a starting word.
    """
)

# --- Load and clean the corpus ---
@st.cache_data
def load_corpus():
    url = 'https://raw.githubusercontent.com/leontoddjohnson/datasets/main/text/inspiration_quotes.txt'
    content = requests.get(url).text
    quotes = content.replace('\n', ' ')
    quotes = re.split("[“”]", quotes)
    quotes = quotes[1::2]  # extract only the quote text
    corpus = ' '.join(quotes)
    corpus = re.sub(r"\s+", " ", corpus).strip()
    return corpus

corpus = load_corpus()

# --- Inputs ---
st.subheader("Generate Text")
term_count = st.number_input(
    "Number of words to generate:", value=30, step=1, format="%d"
)
seed = st.text_input("Optional seed word (leave blank for random start):")

# --- Generate text ---
if st.button("Generate"):
    text_gen = MarkovText(corpus, state_size=1)
    text_gen.get_term_dict()

    try:
        if seed.strip() == "":
            generated = text_gen.generate(term_count=term_count)
        else:
            generated = text_gen.generate(term_count=term_count, seed_term=seed.strip())
        st.success("Generated Text:")
        st.write(generated)
    except ValueError as e:
        st.error(str(e))

# --- Footer ---
st.markdown("---")
st.caption("Built with NumPy • Streamlit • Markov Chains")
