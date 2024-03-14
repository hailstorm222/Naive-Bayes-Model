import numpy as np

from tests.utils import build_dataset
from src.naive_bayes_em import NaiveBayesEM


def main():
    # Load the dataset; fit the NB+EM model
    data, labels, speeches, vocab = build_dataset(
        "data", num_docs=100, max_words=2000, vocab_size=1000)
    isfinite = np.isfinite(labels)
    nb = NaiveBayesEM(max_iter=10)
    nb.fit(data, labels)

    # Use predict_proba to see output probabilities
    probs = nb.predict_proba(data)[isfinite]
    preds = nb.predict(data)
    correct = preds[isfinite] == labels[isfinite]

    # The model's "confidence" in its predicted output when right 
    right_label = labels[isfinite][correct].astype(int)
    prob_when_correct = np.mean(probs[correct, right_label])

    # The model's "confidence" in its predicted output when wrong 
    incorrect = np.logical_not(correct)
    wrong_label = 1 - labels[isfinite][incorrect].astype(int)
    prob_when_incorrect = np.mean(probs[incorrect, wrong_label])

    # Use these number to answer FRQ 3a
    print(" ".join([
        f"Across {np.sum(correct):d} correct predictions,",
        f"NBEM has average confidence {prob_when_correct * 100:.2f}%"]))
    print(" ".join([
        f"Across {np.sum(incorrect):d} incorrect predictions,",
        f"NBEM has average confidence {prob_when_incorrect * 100:.2f}%"]))


if __name__ == "__main__":
    main()
