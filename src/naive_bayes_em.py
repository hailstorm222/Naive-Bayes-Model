import warnings
import numpy as np

from src.utils import softmax, stable_log_sum
from src.sparse_practice import flip_bits_sparse_matrix
from src.naive_bayes import NaiveBayes


class NaiveBayesEM(NaiveBayes):
    """
    A NaiveBayes classifier for binary data, that uses both unlabeled and
        labeled data in the Expectation-Maximization algorithm
    """

    def __init__(self, max_iter=10, smoothing=1):
        """
        Args:
            max_iter: the maximum number of iterations in the EM algorithm
            smoothing: controls the smoothing behavior when computing beta
        """
        self.max_iter = max_iter
        self.smoothing = smoothing

    def initialize_params(self, vocab_size, n_labels):
        """
        Initialize self.alpha such that
            `p(y_i = k) = 1 / n_labels`
            for all k
        and initialize self.beta such that
            `p(w_j | y_i = k) = 1/2`
            for all j, k.
        """

        self.alpha = np.ones(n_labels) / n_labels
        self.beta = np.ones((vocab_size, n_labels)) / 2

    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        self.initialize_params(vocab_size, n_labels)
        

        # Convert y to an array of integers with -1 for missing labels
        is_labeled = ~np.isnan(y)
        y_integer = np.where(is_labeled, y, -1).astype(int)

        for t in range(self.max_iter):
            # E-step:
            probs = self.predict_proba(X)  # Shape: (n_docs, n_labels)
            probs[is_labeled, :] = 0
            probs[np.arange(n_docs)[is_labeled], y_integer[is_labeled].astype(int)] = 1

            # M-step:
            # Update self.alpha
            self.alpha = np.sum(probs, axis=0) / n_docs

            self.beta = np.zeros((vocab_size, n_labels), dtype=float)

            beta_numerator = X.T.dot(probs) + self.smoothing
            self.beta =  beta_numerator / (np.sum(probs, axis = 0) + 2 * self.smoothing)

        



    def likelihood(self, X, y):
        r"""
        Using the self.alpha and self.beta that were already computed in
            `self.fit`, compute the LOG likelihood of the data.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the log likelihood of the data.
        """

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2

        is_labeled = ~np.isnan(y)
        y_integer = np.where(is_labeled, y, -1).astype(int)

        # Initialize log likelihood
        log_likelihood = 0.0

        probs = self.predict_proba(X) 
        probs[is_labeled, :] = 0
        probs[np.arange(n_docs)[is_labeled], y_integer[is_labeled].astype(int)] = 1

        # Compute log probabilities
        log_beta = np.log(self.beta)
        log_one_minus_beta = np.log(1 - self.beta)

        # Perform matrix multiplication to get the log likelihood of words that are present
        log_likelihood_present = X.dot(log_beta)

        # Flip the bits in X to handle words that are absent
        X_flipped = flip_bits_sparse_matrix(X)  
        log_likelihood_absent = X_flipped.dot(log_one_minus_beta)

        # Calculate the total log likelihood by combining both present and absent word contributions
        log_likelihood_total = log_likelihood_present + log_likelihood_absent

        # Add the log prior to the log likelihood
        log_prior = np.log(self.alpha)
        log_posterior = log_likelihood_total + log_prior + np.log(probs)

        log_likelihood = stable_log_sum(log_posterior)

        return log_likelihood

