import csv
from typing import Dict, List

from tqdm import tqdm

from spellchecker import SpellChecker
import numpy as np


def read_dict(filename: str) -> Dict:
    """
    Reads from dictionary file into python dict
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        dictionary = {}
        for row in reader:
            if not row:
                continue
            try:
                dictionary[row[0]] = int(row[1])
            except ValueError:
                continue
    return dictionary


def read_data(filename: str) -> List:
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        output = []
        for line in reader:
            output.append(line)
    return output


def write_list(filename: str, ll: List):
    with open(filename, 'w') as file:
        for word in ll:
            file.write("%s\n" % word)


def read_list(filename: str):
    ll = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            ll.append(line.strip())
    return ll


def precision_at_k(true_words: List[str], suggestions: List[List[str]], k: int = 10) -> float:
    cnt = 0
    total = len(true_words)
    for true, sug in zip(true_words, suggestions):
        if len(sug) > k:
            sug = sug[:k]
        if true in sug:
            cnt += 1
    return cnt / total


if __name__ == '__main__':
    test = read_data('test.txt')
    train = read_data('train.tsv')

    checker = SpellChecker('en_US')
    erroneous, correct, suggestions = [], [], []

    idx = np.random.choice(np.arange(len(train)), 5000, replace=False)
    for index in tqdm(idx, leave=True, position=0):
        pair = train[index]
        error, true = pair
        suggestion = checker(error)
        rand_sug = None
        for sug in suggestion:
            if sug != true:
                rand_sug = sug
                break

        erroneous.append(error)
        correct.append(true)
        suggestions.append(rand_sug)

    X, y = checker.transform_data(erroneous, correct, suggestions)
    checker.fit(X, y)

    all_suggestions, all_true = [], []
    for pair in tqdm(test, position=0, leave=True):
        error, true = pair
        suggestions = checker(error)
        all_suggestions.append(suggestions)
        all_true.append(true)

    print(f'Precision@1 = {precision_at_k(all_true, all_suggestions, k=1)}')
    print(f'Precision@10 = {precision_at_k(all_true, all_suggestions, k=10)}')
