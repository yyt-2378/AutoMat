import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import warnings


def near_field_evolution(u_0, z, lambda_, extent, use_ASM_only=False):
    H = []
    h = []
    u_1 = []
    dH = []

    extent = np.array(extent).reshape(-1) * np.ones(2)

    if z == 0:
        H = 1
        u_1 = u_0
        return u_1, H, h, dH

    if z == np.inf:
        return u_1, H, h, dH

    Npix = np.array(u_0.shape)

    xgrid = (0.5 + np.arange(-Npix[0] / 2, Npix[0] / 2)) / Npix[0]
    ygrid = (0.5 + np.arange(-Npix[1] / 2, Npix[1] / 2)) / Npix[1]

    k = 2 * np.pi / lambda_[0]

    # Undersampling parameter
    F = np.mean(extent ** 2 / (lambda_[0] * z * Npix))

    if abs(F) < 1 and not use_ASM_only:
        # Farfield propagation
        warnings.warn('Farfield regime, F/Npix={}'.format(F))
        Xrange = xgrid * extent[0]
        Yrange = ygrid * extent[1]
        X, Y = np.meshgrid(Xrange, Yrange)
        h = np.exp(1j * k * z + 1j * k / (2 * z) * (X.T ** 2 + Y.T ** 2))

        # This serves as low pass filter for the far nearfield
        H = ifftshift(fft2(fftshift(h)))
        H = H / abs(H[Npix[0] // 2, Npix[1] // 2])  # Renormalize to conserve flux in image
    else:
        # Standard ASM
        kx = 2 * np.pi * xgrid / extent[0] * Npix[0]
        ky = 2 * np.pi * ygrid / extent[1] * Npix[1]
        Kx, Ky = np.meshgrid(kx, ky)

        dH = (-1j * (Kx.T ** 2 + Ky.T ** 2) / (2 * k))

        H = np.exp(1j * z * np.sqrt(k ** 2 - Kx.T ** 2 - Ky.T ** 2))  # It makes it a bit more sensitive to z distance
        h = []

    u_1 = ifft2(np.multiply(ifftshift(H), fft2(u_0)))
    return u_1, H, h, dH
