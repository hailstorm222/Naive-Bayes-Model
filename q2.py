import numpy as np

from tests.utils import build_dataset
from src.naive_bayes_em import NaiveBayesEM

def format_row(word, prob0, prob1, width):
    # Adjust the format as needed
    return f"{word:<{width}} {prob0:<10} {prob1:<10}"

def print_table(headers, rows):
    width = max(len(word) for word, _, _ in rows) + 1  # Plus one for padding
    header_row = format_row(*headers, width)
    print(header_row)
    print("-" * len(header_row))  # Separator line
    for row in rows:
        print(format_row(*row, width))

def main():
    # Load the dataset; fit the NB+EM model
    data, labels, speeches, vocab = build_dataset("data", num_docs=100, max_words=2000, vocab_size=1000)
    nb = NaiveBayesEM(max_iter=10)
    nb.fit(data, labels)

    f_scores = nb.beta[:, 1] / nb.beta[:, 0]
    f_idxs = np.argsort(f_scores)

    headers = ["Word", "Beta[:, 0]", "Beta[:, 1]"]
    rows = []

    n_to_print = 3
    for idx in f_idxs[:n_to_print]:
        rows.append((vocab[idx], f"{nb.beta[idx, 0]:.3f}", f"{nb.beta[idx, 1]:.3f}"))

    for idx in reversed(f_idxs[-n_to_print:]):
        rows.append((vocab[idx], f"{nb.beta[idx, 0]:.3f}", f"{nb.beta[idx, 1]:.3f}"))

    closest_to_one = np.argsort(np.max(np.stack([f_scores, 1 / f_scores], axis=1), axis=1))
    for idx in closest_to_one[:n_to_print]:
        rows.append((vocab[idx], f"{nb.beta[idx, 0]:.3f}", f"{nb.beta[idx, 1]:.3f}"))

    print_table(headers, rows)

if __name__ == "__main__":
    main()
