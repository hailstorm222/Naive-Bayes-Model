import numpy as np

from tests.utils import build_dataset
from src.naive_bayes import NaiveBayes
from src.naive_bayes_em import NaiveBayesEM


def main():
    """
    Helper code for the Free Response Questions
    """
    # Load the dataset
    data, labels, speeches, vocab = build_dataset(
        "data", num_docs=100, max_words=2000, vocab_size=1000)
    isfinite = np.isfinite(labels)

    ### Question 1

    # Fit and evaluate the NB model
    naive_bayes = NaiveBayes()
    naive_bayes.fit(data, labels)
    nb_likelihood = naive_bayes.likelihood(data, labels)
    nb_preds = naive_bayes.predict(data)
    nb_correct = nb_preds[isfinite] == labels[isfinite]

    # Add these numbers to table in FRQ 1
    print(f"NB log likelihood: {nb_likelihood}")
    print(f"NB accuracy: {np.mean(nb_correct)}")

    max_iters = [1, 2, 10]
    for max_iter in max_iters:
        # Fit and evaluate the NB+EM model
        naive_bayes_em = NaiveBayesEM(max_iter=max_iter)
        naive_bayes_em.fit(data, labels)
        nbem_likelihood = naive_bayes_em.likelihood(data, labels)
        nbem_preds = naive_bayes_em.predict(data)
        nbem_correct = nbem_preds[isfinite] == labels[isfinite]

        # Add these numbers to table in FRQ 1
        print(f"NBEM {max_iter} log likelihood: {nbem_likelihood}")
        print(f"NBEM {max_iter} accuracy: {np.mean(nbem_correct)}")


if __name__ == "__main__":
    main()
