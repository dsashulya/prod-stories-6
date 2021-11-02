from typing import Tuple, Dict, Set, List
from textdistance import damerau_levenshtein, hamming, jaro_winkler
from pylcs import lcs
from spylls.hunspell import Dictionary
from tqdm import tqdm
import numpy as np
from catboost import CatBoostClassifier


class SpellChecker:
    def __init__(self, dict_name: str, iterations: int = 20, lr: float = 1, depth: int = 5):
        self.dict = Dictionary.from_files(dict_name)
        self.classifier = CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=depth)

    def __call__(self, word):
        if self.lookup(word):
            return [word]
        else:
            return self.suggest_rank(word)

    def lookup(self, word: str) -> bool:
        return self.dict.lookup(word)

    def suggest(self, word: str):
        suggestions = self.dict.suggest(word)
        return suggestions

    @staticmethod
    def _get_features(erroneous_word, suggestion):
        distances = [damerau_levenshtein, hamming, jaro_winkler, lcs]
        return [distance(erroneous_word, suggestion) for distance in distances]

    def transform_data(self, erroneous: List, correct: List, suggestions: List[str]):
        X, y = [], []
        for er, cor, sug in tqdm(zip(erroneous, correct, suggestions), total=len(erroneous),
                                 leave=True, position=0):
            if sug is None:
                continue
            features = self._get_features(er, sug)
            X.append(features)
            y.append(0)

            features = self._get_features(er, cor)
            X.append(features)
            y.append(1)
        return np.array(X), np.array(y)

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def predict_proba(self, word):
        suggestions = self.suggest(word)
        X = []
        words = []
        for w in suggestions:
            words.append(w)
            features = self._get_features(w, word)
            X.append(features)
        if not len(X):
            return np.array([[0, 0]]), [None]
        return self.classifier.predict_proba(np.array(X)), words

    def suggest_best(self, word):
        proba, words = self.predict_proba(word)
        print(proba, words)
        return words[np.argmax(proba[:, 1])]

    def suggest_rank(self, word):
        proba, words = self.predict_proba(word)
        idx = np.argsort(-proba[:, 1])
        return [words[index] for index in idx]


