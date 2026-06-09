"""
Graph utility functions: BFS/DFS orderings and adjacency construction.
"""

import numpy as np
from collections import deque


def bfs_ordering(adjacency_matrix, seed_idx=0):
    """
    Compute BFS visit ordering of nodes given an adjacency matrix.

    Args:
        adjacency_matrix: NxN numpy array (weighted or binary).
        seed_idx: Starting node index for BFS.

    Returns:
        list: Node indices in BFS visit order.
    """
    n = adjacency_matrix.shape[0]
    visited = [False] * n
    order = []
    queue = deque([seed_idx])
    visited[seed_idx] = True

    while queue:
        node = queue.popleft()
        order.append(node)
        # Get neighbors sorted by edge weight (descending) for deterministic ordering
        neighbors = []
        for j in range(n):
            if adjacency_matrix[node, j] > 0 and not visited[j]:
                neighbors.append((j, adjacency_matrix[node, j]))
        neighbors.sort(key=lambda x: -x[1])
        for neighbor, _ in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    # Add any unvisited nodes (disconnected components)
    for i in range(n):
        if not visited[i]:
            order.append(i)

    return order


def dfs_ordering(adjacency_matrix, seed_idx=0):
    """
    Compute DFS visit ordering of nodes given an adjacency matrix.

    Args:
        adjacency_matrix: NxN numpy array (weighted or binary).
        seed_idx: Starting node index for DFS.

    Returns:
        list: Node indices in DFS visit order.
    """
    n = adjacency_matrix.shape[0]
    visited = [False] * n
    order = []

    def _dfs(node):
        visited[node] = True
        order.append(node)
        # Get neighbors sorted by edge weight (descending) for deterministic ordering
        neighbors = []
        for j in range(n):
            if adjacency_matrix[node, j] > 0 and not visited[j]:
                neighbors.append((j, adjacency_matrix[node, j]))
        neighbors.sort(key=lambda x: -x[1])
        for neighbor, _ in neighbors:
            if not visited[neighbor]:
                _dfs(neighbor)

    _dfs(seed_idx)

    # Add any unvisited nodes (disconnected components)
    for i in range(n):
        if not visited[i]:
            order.append(i)

    return order


def gaussian_adjacency(distance_matrix, sigma=1.0):
    """
    Convert a distance matrix to an adjacency matrix using a Gaussian kernel.

    A_ij = exp(-d_ij^2 / (2 * sigma^2))

    Args:
        distance_matrix: NxN numpy array of pairwise distances.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        numpy.ndarray: NxN adjacency matrix with values in (0, 1].
    """
    adjacency = np.exp(-distance_matrix ** 2 / (2 * sigma ** 2))
    # Set diagonal to zero (no self-loops)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency
