import numpy as np
from sklearn.utils.random import (sample_without_replacement,
                                  sample_erdos_renyi_gnm,
                                  check_random_state)


def testall():
    test_correctness_1_sample_erdos_renyi_gnm()
    test_correctness_2_sample_erdos_renyi_gnm()
    test_correctness_3_sample_erdos_renyi_gnm()
    test_consistency_sample_erdos_renyi_gnm()


def test_correctness_1_sample_erdos_renyi_gnm():
    """Tests that the adjacency matrix output is correct for a very simple
    hard-coded example."""
    n = 5
    m = 5
    samples = 1
    random_state = 1337
    A = np.array([[0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 0, 0, 0, 0]])
    B = sample_erdos_renyi_gnm(
        n, m, samples, random_state=random_state, return_as="adjacency_matrix")
    assert np.all(A == np.tril(B[0].todense()))


def test_correctness_2_sample_erdos_renyi_gnm():
    """Tests that sampling all edges returns a matrix with entries of
    one everywhere but on the diagonal.
    """
    n = 10
    m = 10*9//2
    samples = 1
    A = sample_erdos_renyi_gnm(
        n, m, samples, return_as="adjacency_matrix")
    assert np.all(A[0].todense() == np.ones((n, n))-np.eye(n))


def test_correctness_3_sample_erdos_renyi_gnm():
    """Tests that the edge array output is equal to that of a simple
    imlementation for more than one sample.
    """
    n = 20
    m = 60
    samples = 20
    random_state = 42
    A = sample_erdos_renyi_gnm(
        n, m, samples, random_state=random_state, return_as="edge_array")

    random_state = check_random_state(random_state)
    for j in range(samples):
        edge_indices = sample_without_replacement(
            n*(n-1)//2, m, random_state=random_state)
        row_indices = []
        column_indices = []
        for ind in edge_indices:
            k = 1
            while ind != ind % k:
                ind -= k
                k += 1
            row_indices.append(k)
            column_indices.append(ind % k)

        assert np.all(np.array([row_indices, column_indices]) == A[:, :, j])


def test_consistency_sample_erdos_renyi_gnm():
    """Tests that the edge_array output is consistent with the
    adjacency_matrix output."""
    n = 50
    m = 50
    samples = 10
    random_state = 1337
    A = sample_erdos_renyi_gnm(
        n, m, samples, random_state=random_state, return_as="edge_array")
    B = sample_erdos_renyi_gnm(
        n, m, samples, random_state=random_state, return_as="adjacency_matrix")

    for i in range(samples):
        AA = set(tuple(a) for a in A[:, :, i].T)
        BB1 = set(tuple(b)
                  for b in np.stack(np.nonzero(np.tril(B[i].todense()))).T)
        BB2 = set(tuple(b)[::-1]
                  for b in np.stack(np.nonzero(np.triu(B[i].todense()))).T)
        assert AA == BB1
        assert AA == BB2
