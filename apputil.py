import numpy as np
from collections import defaultdict
import re
import random

class MarkovText:
    def __init__(self, corpus, tokenizer=None, state_size=1, rng=None):
        """
        corpus: str
        tokenizer: callable(str)->list[str] (optional). Defaults to simple whitespace split.
        state_size: int, number of tokens per state (k-gram). Use 1 for the base exercise.
        rng: np.random.Generator or random.Random for reproducibility (optional)
        """
        self.corpus = corpus
        self.tokenizer = tokenizer or (lambda s: s.split())
        self.state_size = max(1, int(state_size))
        self.tokens = self.tokenizer(self.corpus)
        self.term_dict = None
        self._np_rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
        self._py_rng = rng if isinstance(rng, random.Random) else random.Random()

    def _make_state(self, tokens, i):
        """Return the state tuple of length state_size starting at i."""
        return tuple(tokens[i:i + self.state_size])

    def get_term_dict(self):
        """
        Build a transition dictionary:
        - Keys: states (single token for state_size=1; otherwise a tuple of tokens)
        - Values: list of followers (with duplicates to preserve empirical frequencies)
        """
        followers = defaultdict(list)
        n = len(self.tokens)
        if n <= self.state_size:  # not enough tokens to form a state + follower
            self.term_dict = dict(followers)
            return self.term_dict

        for i in range(n - self.state_size):
            state = self._make_state(self.tokens, i)
            nxt = self.tokens[i + self.state_size]
            followers[state].append(nxt)

        self.term_dict = dict(followers)
        return self.term_dict

    def _pick_start_state(self, seed_term=None):
        """
        Pick a valid starting state. For state_size=1, seed_term is a string token.
        For state_size>1, seed_term can be a tuple/list of length state_size or a string to be split.
        """
        if self.term_dict is None or not self.term_dict:
            raise ValueError("Term dictionary is empty. Call get_term_dict() after providing a non-empty corpus.")

        if seed_term is None:
            return self._py_rng.choice(list(self.term_dict.keys()))

        # Normalize seed for state_size
        if self.state_size == 1:
            state = (seed_term,)
        else:
            # Allow a space-separated seed or a tuple/list
            if isinstance(seed_term, (tuple, list)):
                state = tuple(seed_term)
            else:
                candidate = self.tokenizer(str(seed_term))
                state = tuple(candidate[:self.state_size])

            if len(state) != self.state_size:
                raise ValueError(f"seed_term must have {self.state_size} tokens.")

        if state not in self.term_dict:
            raise ValueError(f"Seed state {state} not found in corpus.")
        return state

    def generate(self, term_count=20, seed_term=None):
        """
        Generate text using the Markov property.
        - term_count: number of tokens to generate
        - seed_term: optional starting token (k-gram if state_size>1)
        Behavior when a state has no followers (e.g., you hit the very end of the corpus once):
        -> chooses a fresh random valid state so generation continues smoothly.
        """
        if term_count <= 0:
            return ""

        if self.term_dict is None:
            self.get_term_dict()

        # choose starting state
        state = self._pick_start_state(seed_term)

        # Initialize output with the state tokens
        output = list(state)

        # Generate next tokens
        for _ in range(term_count - self.state_size):
            followers = self.term_dict.get(state, [])
            if not followers:
                # Dead end (e.g., last unique occurrence) — jump to a random valid state
                state = self._py_rng.choice(list(self.term_dict.keys()))
                output.extend(list(state))  # seed the stream to keep flow natural
                continue

            # Sample next token proportional to frequency (because duplicates retained)
            next_tok = self._np_rng.choice(followers)
            output.append(next_tok)

            # Slide the window to form the next state
            if self.state_size == 1:
                state = (next_tok,)
            else:
                state = tuple(output[-self.state_size:])

        # Basic detokenization for k>1 is straightforward join; for k=1 punctuation spacing may be improved if desired.
        return " ".join(output)


# 1) Build the model (k = 1 for the base exercise)
text_gen = MarkovText(corpus, state_size=1)

# 2) Exercise 1: build the transition dictionary
term_dict = text_gen.get_term_dict()
# Example peek:
# {('Healing',): ['comes'], ('comes',): ['from', 'the', 'the', ...], ('from',): ['taking', 'aesthetic', 'a'], ...}

# 3) Exercise 2: generate text
print(text_gen.generate(term_count=30))                     # random start
print(text_gen.generate(term_count=30, seed_term="Life"))   # seeded start
