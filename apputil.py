# ---------------------------------------------------------
# apputil.py
# ---------------------------------------------------------
# Utility classes for assignments:
# 1) MarkovText class for Markov text generation
# ---------------------------------------------------------

import numpy as np
from collections import defaultdict
import random


class MarkovText:
    def __init__(self, corpus, tokenizer=None, state_size=1, rng=None):
        """
        corpus: str
        tokenizer: callable(str) -> list[str], optional (default: whitespace split)
        state_size: int, number of tokens per state (k-gram)
        rng: np.random.Generator or random.Random for reproducibility (optional)
        """
        self.corpus = corpus
        self.tokenizer = tokenizer or (lambda s: s.split())
        self.state_size = max(1, int(state_size))
        self.tokens = self.tokenizer(self.corpus)
        self.term_dict = None

        # RNG handling
        self._np_rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
        self._py_rng = rng if isinstance(rng, random.Random) else random.Random()

    def _make_state(self, tokens, i):
        """Return the state tuple of length state_size starting at index i."""
        return tuple(tokens[i:i + self.state_size])

    def get_term_dict(self):
        """
        Build and return the Markov transition dictionary.

        Keys:
            state tuples (length = state_size)

        Values:
            list of follower tokens (duplicates preserved)
        """
        followers = defaultdict(list)
        n = len(self.tokens)

        if n <= self.state_size:
            self.term_dict = dict(followers)
            return self.term_dict

        for i in range(n - self.state_size):
            state = self._make_state(self.tokens, i)
            next_token = self.tokens[i + self.state_size]
            followers[state].append(next_token)

        self.term_dict = dict(followers)
        return self.term_dict

    def _pick_start_state(self, seed_term=None):
        """
        Choose a valid starting state.
        - For state_size=1, seed_term should be a single token (string)
        - For state_size>1, seed_term can be tuple/list of tokens or a string to tokenize
        """
        if self.term_dict is None or not self.term_dict:
            raise ValueError("Term dictionary is empty. Call get_term_dict() first.")

        if seed_term is None:
            return self._py_rng.choice(list(self.term_dict.keys()))

        if self.state_size == 1:
            state = (seed_term,)
        else:
            if isinstance(seed_term, (tuple, list)):
                state = tuple(seed_term)
            else:
                state = tuple(self.tokenizer(str(seed_term))[:self.state_size])

            if len(state) != self.state_size:
                raise ValueError(f"seed_term must have {self.state_size} tokens")

        if state not in self.term_dict:
            raise ValueError(f"Seed state {state} not found in corpus")

        return state

    def generate(self, term_count=20, seed_term=None):
        """
        Generate text using the Markov property.

        term_count: total number of tokens to generate (must be int-castable)
        seed_term: optional starting token or k-gram

        If a state has no followers (dead end), generation "jumps" to a random valid
        state and continues, without exceeding term_count tokens.
        """
        try:
            term_count = int(term_count)
        except Exception as e:
            raise ValueError("term_count must be an integer") from e

        if term_count <= 0:
            return ""

        if self.term_dict is None:
            self.get_term_dict()

        state = self._pick_start_state(seed_term)
        output = list(state)[:term_count]

        while len(output) < term_count:
            followers = self.term_dict.get(state, [])

            if not followers:
                # Dead end: jump to a random valid state and add minimal tokens
                state = self._py_rng.choice(list(self.term_dict.keys()))
                if self.state_size == 1:
                    output.append(state[0])
                else:
                    for tok in state:
                        if len(output) >= term_count:
                            break
                        output.append(tok)
                continue

            next_token = self._np_rng.choice(followers)
            output.append(next_token)

            if self.state_size == 1:
                state = (next_token,)
            else:
                state = tuple(output[-self.state_size:])

        return " ".join(output)


# ---------------------------------------------------------
# Example usage (ONLY runs when executed directly)
# ---------------------------------------------------------
if __name__ == "__main__":
    corpus = "Life is short and art is long life is beautiful"
    text_gen = MarkovText(corpus, state_size=1)

    term_dict = text_gen.get_term_dict()
    print("Transition dictionary size:", len(term_dict))

    print("\nGenerated text:")
    print(text_gen.generate(term_count=30))
    print(text_gen.generate(term_count=30, seed_term="Life"))


