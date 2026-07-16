import numpy as np


def find_reconstruction_ROI_external(positions, Np_o, Np_p):
    positions = positions[:, [1, 0]]  # Swap columns to match MATLAB's 1-indexing
    positions += np.ceil((Np_o / 2) - (Np_p / 2))
    sub_px_shift = positions - np.round(positions)

    # Return to the original XY coordinates
    sub_px_shift = sub_px_shift[:, [1, 0]]

    positions = np.round(positions).astype(int)

    # Calculate the range
    range_min = np.min(positions, axis=0)
    range_max = np.max(positions, axis=0) + Np_p

    if np.any(range_min < 0) or np.any(range_max > Np_o):
        raise ValueError(
            f"Object size is too small, not enough space for probes !! \nposition range: {range_min[0]} {range_min[1]} {range_max[0]} {range_max[1]}, \nobject size: {Np_o[0]} {Np_o[1]}")

    oROI = []
    for dim in range(2):
        oROI_dim = np.vstack((positions[:, dim], positions[:, dim] + Np_p[dim] - 1)).T
        oROI.append(oROI_dim.astype(np.uint32))

    Npos = len(positions)
    oROI_vec = [[None, None] for _ in range(Npos)]
    for ii in range(Npos):
        for i in range(2):
            oROI_vec[ii][i] = np.arange(oROI[i][ii, 0], oROI[i][ii, 1] + 1)

    return oROI, oROI_vec, sub_px_shift


if __name__ == '__main__':
    positions = np.array([[1, 2], [3, 4]])
    Np_o = np.array([10, 10])
    Np_p = np.array([2, 2])

    # Call the function
    oROI, oROI_vec, sub_px_shift = find_reconstruction_ROI_external(positions, Np_o, Np_p)

