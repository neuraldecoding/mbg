"""
Standard 10-20 system electrode positions (2D projected coordinates).
Supports 19-channel (clinical), 22-channel (BCI Competition IV 2a), and 64-channel montages.
"""

import numpy as np


# Standard 10-20 system 2D projected coordinates (x, y)
# Coordinates are normalized to approximately [-1, 1] range
ELECTRODE_POSITIONS_19 = {
    'Fp1': (-0.31, 0.95), 'Fp2': (0.31, 0.95),
    'F7': (-0.81, 0.59), 'F3': (-0.39, 0.59), 'Fz': (0.0, 0.59),
    'F4': (0.39, 0.59), 'F8': (0.81, 0.59),
    'T3': (-1.0, 0.0), 'C3': (-0.5, 0.0), 'Cz': (0.0, 0.0),
    'C4': (0.5, 0.0), 'T4': (1.0, 0.0),
    'T5': (-0.81, -0.59), 'P3': (-0.39, -0.59), 'Pz': (0.0, -0.59),
    'P4': (0.39, -0.59), 'T6': (0.81, -0.59),
    'O1': (-0.31, -0.95), 'O2': (0.31, -0.95),
}

# BCI Competition IV 2a: 22 EEG channels
ELECTRODE_POSITIONS_22 = {
    'Fz': (0.0, 0.59), 'FC3': (-0.39, 0.30), 'FC1': (-0.19, 0.30),
    'FCz': (0.0, 0.30), 'FC2': (0.19, 0.30), 'FC4': (0.39, 0.30),
    'C5': (-0.75, 0.0), 'C3': (-0.5, 0.0), 'C1': (-0.25, 0.0),
    'Cz': (0.0, 0.0), 'C2': (0.25, 0.0), 'C4': (0.5, 0.0),
    'C6': (0.75, 0.0), 'CP3': (-0.39, -0.30), 'CP1': (-0.19, -0.30),
    'CPz': (0.0, -0.30), 'CP2': (0.19, -0.30), 'CP4': (0.39, -0.30),
    'P1': (-0.19, -0.59), 'Pz': (0.0, -0.59), 'P2': (0.19, -0.59),
    'POz': (0.0, -0.77),
}

# Extended 64-channel montage (10-10 system subset)
ELECTRODE_POSITIONS_64 = {
    'Fp1': (-0.31, 0.95), 'Fpz': (0.0, 0.95), 'Fp2': (0.31, 0.95),
    'AF7': (-0.59, 0.81), 'AF3': (-0.25, 0.81), 'AF4': (0.25, 0.81), 'AF8': (0.59, 0.81),
    'F7': (-0.81, 0.59), 'F5': (-0.59, 0.59), 'F3': (-0.39, 0.59),
    'F1': (-0.19, 0.59), 'Fz': (0.0, 0.59), 'F2': (0.19, 0.59),
    'F4': (0.39, 0.59), 'F6': (0.59, 0.59), 'F8': (0.81, 0.59),
    'FT7': (-0.90, 0.30), 'FC5': (-0.59, 0.30), 'FC3': (-0.39, 0.30),
    'FC1': (-0.19, 0.30), 'FCz': (0.0, 0.30), 'FC2': (0.19, 0.30),
    'FC4': (0.39, 0.30), 'FC6': (0.59, 0.30), 'FT8': (0.90, 0.30),
    'T7': (-1.0, 0.0), 'C5': (-0.75, 0.0), 'C3': (-0.5, 0.0),
    'C1': (-0.25, 0.0), 'Cz': (0.0, 0.0), 'C2': (0.25, 0.0),
    'C4': (0.5, 0.0), 'C6': (0.75, 0.0), 'T8': (1.0, 0.0),
    'TP7': (-0.90, -0.30), 'CP5': (-0.59, -0.30), 'CP3': (-0.39, -0.30),
    'CP1': (-0.19, -0.30), 'CPz': (0.0, -0.30), 'CP2': (0.19, -0.30),
    'CP4': (0.39, -0.30), 'CP6': (0.59, -0.30), 'TP8': (0.90, -0.30),
    'P7': (-0.81, -0.59), 'P5': (-0.59, -0.59), 'P3': (-0.39, -0.59),
    'P1': (-0.19, -0.59), 'Pz': (0.0, -0.59), 'P2': (0.19, -0.59),
    'P4': (0.39, -0.59), 'P6': (0.59, -0.59), 'P8': (0.81, -0.59),
    'PO7': (-0.59, -0.77), 'PO3': (-0.25, -0.77), 'POz': (0.0, -0.77),
    'PO4': (0.25, -0.77), 'PO8': (0.59, -0.77),
    'O1': (-0.31, -0.95), 'Oz': (0.0, -0.95), 'O2': (0.31, -0.95),
    'Iz': (0.0, -1.05),
    'FT9': (-1.0, 0.30), 'FT10': (1.0, 0.30),
    'TP9': (-1.0, -0.30),
}


def get_electrode_positions(num_channels):
    """
    Get electrode positions for a given number of channels.

    Args:
        num_channels: Number of EEG channels (19, 22, or 64 supported directly;
                     other values return evenly spaced positions on a unit circle).

    Returns:
        dict: Mapping of electrode name -> (x, y) coordinates.
    """
    if num_channels == 19:
        return ELECTRODE_POSITIONS_19.copy()
    elif num_channels == 22:
        return ELECTRODE_POSITIONS_22.copy()
    elif num_channels == 64:
        return ELECTRODE_POSITIONS_64.copy()
    else:
        # Generate evenly spaced positions on a unit circle for arbitrary channel counts
        positions = {}
        for i in range(num_channels):
            angle = 2 * np.pi * i / num_channels
            positions[f'Ch{i+1}'] = (np.cos(angle), np.sin(angle))
        return positions


def compute_distance_matrix(positions):
    """
    Compute pairwise Euclidean distance matrix from electrode positions.

    Args:
        positions: dict of electrode name -> (x, y) coordinates.

    Returns:
        numpy.ndarray: NxN distance matrix.
    """
    coords = np.array(list(positions.values()))
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.sqrt(
                (coords[i, 0] - coords[j, 0]) ** 2 +
                (coords[i, 1] - coords[j, 1]) ** 2
            )
    return dist_matrix
