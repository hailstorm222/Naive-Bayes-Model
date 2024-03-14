import numpy as np
import warnings

from src.utils import softmax
from src.sparse_practice import flip_bits_sparse_matrix


class NaiveBayes:
    """
    A Naive Bayes classifier for binary data.
    """

    def __init__(self, smoothing=1):
        """
        Args:
            smoothing: controls the smoothing behavior when calculating beta
        """
        self.smoothing = smoothing

    def predict(self, X):
        """
        Return the most probable label for each row x of X.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Using self.alpha and self.beta, compute the probability p(y | X[i, :])
            for each row X[i, :] of X.  
        Args:
            X: a sparse matrix of shape `[n_documents, vocab_size]` on which to
               predict p(y | x)

        Returns 
            probs: an array of shape `[n_documents, n_labels]` where probs[i, j] contains
                the probability `p(y=j | X[i, :])`. Thus, for a given row of this array,
                np.sum(probs[i, :]) == 1.
        """
        n_docs, vocab_size = X.shape
        n_labels = 2

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"
        assert vocab_size == self.vocab_size, "Vocab size mismatch"

        # Get the log probabilities
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
        log_posterior = log_likelihood_total + log_prior.reshape(1, -1)

        # Use softmax to convert log posterior to probabilities
        probs = softmax(log_posterior, axis=1)

        return probs

    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None; sets self.alpha and self.beta
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        valid_indices = ~np.isnan(y)
        X_filtered = X[valid_indices]
        y_filtered = y[valid_indices]


        self.alpha = np.zeros(n_labels)
        self.beta = np.zeros((vocab_size, n_labels))

        for label in range(n_labels):
            self.alpha[label] = np.mean(y_filtered == label)
            class_indices = y_filtered == label
            X_class = X_filtered[class_indices]


            # For binary features, count the presence of each word in documents of class 'label'
            word_presence = (X_class.sum(axis=0)).A.squeeze()
            document_count = np.sum(y_filtered == label)
        
            # Adjusting the formula for binary feature context with smoothing
            self.beta[:, label] = (word_presence + self.smoothing) / (document_count + 2 * self.smoothing)



    def likelihood(self, X, y):
        """
        Args: X, a sparse matrix of binary word counts; Y, an array of labels
        Returns: the log likelihood of the data
        """
        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2

         # Filter out rows where y is NaN
        valid_indices = ~np.isnan(y)
        X_filtered = X[valid_indices]
        y_filtered = y[valid_indices]

        # Compute log probabilities
        log_beta = np.log(self.beta)
        log_one_minus_beta = np.log(1 - self.beta)

        # Initialize log likelihood
        log_likelihood = 0

        # Compute the log likelihood for each document
        for i in range(X_filtered.shape[0]):
            label = int(y_filtered[i])

            # Compute log prior for the current label
            log_prior = np.log(self.alpha[label])

            # Get the document row as a dense array
            x_i_dense = X_filtered[i].toarray().ravel()

            # Compute log likelihood for document with the current label
            # Sum log probabilities of observed words and absent words given the label
            doc_log_likelihood = x_i_dense.dot(log_beta[:, label]) + \
                             (1 - x_i_dense).dot(log_one_minus_beta[:, label])

            # Add the log prior to the document's log likelihood
            log_likelihood += log_prior + doc_log_likelihood

        return log_likelihood
