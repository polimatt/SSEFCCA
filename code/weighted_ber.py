"""
This module contains the implementation of the
FR Test Statistic based estimate and the
KNN based estimate for the :term:`Bayes error rate<Bayes Error Rate (BER)>`

Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4)
https://arxiv.org/abs/1811.06419

"""


__all__ = []


import numpy as np

# from dataeval.core._ber import ber_knn, ber_mst
from dataeval.core._mst import minimum_spanning_tree
from numpy.typing import NDArray
from dataeval.protocols import Array

# _BER_FN_MAP = {"KNN": ber_knn, "MST": ber_mst}

def _get_classes_counts(labels: NDArray[np.intp]) -> tuple[int, int]:
    classes, counts = np.unique(labels, return_counts=True)
    M = len(classes)
    if M < 2:
        raise ValueError("Label vector contains less than 2 classes!")
    N = int(np.sum(counts))
    return M, N


def weighted_ber(data: NDArray[np.float64], labels: NDArray[np.intp], weights: NDArray[np.float64]) -> tuple[float, float]:
    """
    An estimator for Multi-class :term:`Bayes error rate<Bayes Error Rate (BER)>` \
    using FR with a minimum spanning tree (MST) test statistic basis.

    Parameters
    ----------
    data : NDArray[np.float64]
        Array of image :term:`embeddings<Embeddings>`
    labels : NDArray[np.intp]
        Array of labels for each image

    Returns
    -------
    tuple[float, float]
        The upper and lower bounds, respectively, of the Bayes Error Rate

    References
    ----------
    [1] `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    Examples
    --------
    >>> import sklearn.datasets as dsets
    >>> from dataeval.core._ber import ber_mst

    >>> images, labels = dsets.make_blobs(n_samples=50, centers=2, n_features=2, random_state=0)
    >>> ber_mst(images, labels)
    (0.04, 0.020416847668728033)
    """

    M, N = _get_classes_counts(labels)

    rows, cols = minimum_spanning_tree(data)  # get rows and cols directly

    # Symmetric edge weighting: average of endpoint weights
    edge_weights = 0.5 * (weights[rows] + weights[cols])
    mismatches = edge_weights * (labels[rows] != labels[cols])

    number_of_mismatches = np.sum(mismatches)
    deltas = number_of_mismatches / (2 * N)
    upper = float(2 * deltas)
    lower = float(((M - 1) / (M)) * (1 - max(1 - 2 * ((M) / (M - 1)) * deltas, 0) ** 0.5))
    return upper
