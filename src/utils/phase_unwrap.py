import numpy as np
from scipy.fftpack import dct, idct


def phase_unwrap(psi, weight=None):
    if weight is None:  # unweighted phase unwrap
        dx = np.concatenate([
            np.zeros((psi.shape[0], 1)),
            np.angle(np.exp(1j * np.diff(psi, 1, axis=1))),  # 直接处理差分结果
            np.zeros((psi.shape[0], 1))
        ], axis=1)

        # 计算dy，使用np.angle(np.exp(1j * ...))处理差分结果
        dy = np.concatenate([
            np.zeros((1, psi.shape[1])),
            np.angle(np.exp(1j * np.diff(psi, 1, axis=0))),  # 直接处理差分结果
            np.zeros((1, psi.shape[1]))
        ], axis=0)

        rho = np.diff(dx, 1, axis=1) + np.diff(dy, 1, axis=0)

        # solve the poisson equation
        phi = solve_poisson(rho)
    else:  # weighted phase unwrap
        if weight.shape != psi.shape:
            raise ValueError("Size of the weight must be the same as size of the wrapped phase")

        dx = np.diff(psi, 1, axis=1)
        dy = np.diff(psi, 1, axis=0)

        WW = weight ** 2
        WWdx = WW[:, :-1] * dx
        WWdy = WW[:-1, :] * dy

        WWdx2 = np.concatenate((np.zeros((WWdx.shape[0], 1)), WWdx), axis=1)
        WWdy2 = np.concatenate((np.zeros((1, WWdy.shape[1])), WWdy), axis=0)
        rk = np.diff(WWdx2, 1, axis=1) + np.diff(WWdy2, 1, axis=0)

        eps = 1e-6
        k = 0
        phi = np.zeros_like(psi)
        while not np.all(rk == 0):
            zk = solve_poisson(rk)
            if k == 0:
                pk = zk
            else:
                betak = np.sum(rk * zk) / np.sum(rkprev * zkprev)
                pk = zk + betak * pk

            Qpk = apply_Q(pk, WW)
            alphak = np.sum(rk * zk) / np.sum(pk * Qpk)
            phi += alphak * pk
            rk -= alphak * Qpk

            rkprev = rk
            zkprev = zk
            k += 1

            if k >= np.prod(psi.shape) or np.linalg.norm(rk) < eps * np.linalg.norm(rk):
                break

    return phi


def solve_poisson(rho):
    dct_rho = dct(dct(rho.T, norm='ortho').T, norm='ortho')
    N, M = rho.shape
    I, J = np.meshgrid(np.arange(M), np.arange(N))
    denominator = 2 * (np.cos(np.pi * I / M) + np.cos(np.pi * J / N) - 2)
    dct_phi = np.where(denominator != 0, dct_rho / denominator, 0)  # Avoid division by zero
    dct_phi[0, 0] = 0  # Handle the inf/nan value

    phi = idct(idct(dct_phi.T, norm='ortho').T, norm='ortho')
    return phi


# def solvePoisson(rho):
#     # 计算rho的二维DCT
#     dctRho = dct(dct(rho, axis=0, norm='ortho'), axis=1, norm='ortho')
#     N, M = rho.shape
#     I, J = np.meshgrid(np.arange(M), np.arange(N))
#
#     # 计算分母，避免除零
#     denominator = 2 * (np.cos(np.pi * I / M) + np.cos(np.pi * J / N) - 2)
#     denominator[0, 0] = 1  # 通过设置虚拟值防止除以零
#
#     # 计算dctPhi并单独处理(0,0)元素
#     dctPhi = np.where(denominator == 0, 0, dctRho / denominator)
#     dctPhi[0, 0] = 0  # 将(0,0)元素设置为0
#
#     # 对dctPhi进行二维逆DCT，得到phi
#     phi = idct(idct(dctPhi, axis=0, norm='ortho'), axis=1, norm='ortho')
#
#     return phi


def apply_Q(p, WW):
    dx = np.diff(p, 1, axis=1)
    dy = np.diff(p, 1, axis=0)

    WWdx = WW[:, :-1] * dx
    WWdy = WW[:-1, :] * dy

    WWdx2 = np.concatenate((np.zeros((WWdx.shape[0], 1)), WWdx), axis=1)
    WWdy2 = np.concatenate((np.zeros((1, WWdy.shape[1])), WWdy), axis=0)
    Qp = np.diff(WWdx2, 1, axis=1) + np.diff(WWdy2, 1, axis=0)

    return Qp
