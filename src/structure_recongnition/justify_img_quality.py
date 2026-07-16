"""No-reference quality assessment for restored STEM images."""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
from scipy import fft as spfft
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu


def imread_gray(path: str) -> np.ndarray:
    """Read and robustly normalize a grayscale image to [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99))
    if p99 > p1:
        return np.clip((img - p1) / (p99 - p1), 0, 1)

    value_range = img.max() - img.min()
    return (img - img.min()) / (value_range + 1e-8)


def _robust_noise_sigma(values: np.ndarray) -> float:
    median = np.median(values)
    mad = np.median(np.abs(values - median)) + 1e-12
    return float(1.4826 * mad)


def _fft_log_magnitude(img: np.ndarray) -> np.ndarray:
    spectrum = spfft.fftshift(spfft.fft2(img))
    return np.log1p(np.abs(spectrum))


def _bandpass_mask(height: int, width: int, rmin: int = 6, rmax: int | None = None) -> np.ndarray:
    if rmax is None:
        rmax = min(height, width) // 2 - 2
    center_y, center_x = height // 2, width // 2
    yy, xx = np.ogrid[:height, :width]
    radius_squared = (yy - center_y) ** 2 + (xx - center_x) ** 2
    return ((radius_squared >= rmin**2) & (radius_squared <= rmax**2)).astype(np.float32)


def _bragg_peakiness_score(log_magnitude: np.ndarray, k_peaks: int = 12, neighborhood: int = 9) -> float:
    height, width = log_magnitude.shape
    band = log_magnitude * _bandpass_mask(height, width)
    coordinates = peak_local_max(
        band,
        min_distance=max(4, neighborhood // 2),
        threshold_rel=0.25,
        num_peaks=k_peaks,
    )
    if coordinates.size == 0:
        return 0.0

    def local_contrast(y: int, x: int, inner_radius: int = 2, outer_radius: int = 6) -> float:
        y0, y1 = max(0, y - inner_radius), min(height, y + inner_radius + 1)
        x0, x1 = max(0, x - inner_radius), min(width, x + inner_radius + 1)
        peak_value = float(np.max(log_magnitude[y0:y1, x0:x1]))

        local_y, local_x = np.ogrid[
            y - outer_radius : y + outer_radius + 1,
            x - outer_radius : x + outer_radius + 1,
        ]
        clipped_y = np.clip(local_y, 0, height - 1)
        clipped_x = np.clip(local_x, 0, width - 1)
        radius = np.sqrt((local_y - y) ** 2 + (local_x - x) ** 2)
        ring = log_magnitude[clipped_y, clipped_x][
            (radius >= 0.7 * outer_radius) & (radius <= outer_radius)
        ]
        background = float(np.median(ring)) if ring.size else 0.0
        return max(0.0, peak_value - background)

    contrasts = [local_contrast(y, x) for y, x in coordinates]
    normalization = float(np.median(band[band > 0])) + 1e-6
    peakiness = float(np.sum(contrasts) / (len(contrasts) * normalization))
    return float(np.tanh(0.3 * peakiness))


def _autocorrelation_offcenter_ratio(img: np.ndarray) -> float:
    spectrum = spfft.fft2(img)
    autocorrelation = np.real(spfft.ifft2(np.abs(spectrum) ** 2))
    autocorrelation = np.fft.fftshift(autocorrelation)
    autocorrelation = (autocorrelation - autocorrelation.min()) / (
        autocorrelation.max() - autocorrelation.min() + 1e-12
    )

    height, width = autocorrelation.shape
    center_y, center_x = height // 2, width // 2
    center = float(autocorrelation[center_y, center_x])
    outer_radius = min(height, width) // 2 - 4
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    ring = autocorrelation[(radius >= 6) & (radius <= outer_radius)]
    if ring.size == 0:
        return 0.0

    offcenter_max = float(np.max(ring))
    return float(np.tanh(0.8 * offcenter_max / (center + 1e-12)))


def _periodicity_metrics(img: np.ndarray) -> Dict[str, float]:
    bragg_peakiness = _bragg_peakiness_score(_fft_log_magnitude(img))
    autocorrelation_ratio = _autocorrelation_offcenter_ratio(img)
    return {
        "periodicity_score": 0.6 * bragg_peakiness + 0.4 * autocorrelation_ratio,
        "bragg_peakiness": bragg_peakiness,
        "autocorr_peak_ratio": autocorrelation_ratio,
    }


def _snr_cnr_metrics(img: np.ndarray) -> Tuple[float, float, float]:
    low_pass = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0, sigmaY=2.0)
    high_pass = img - low_pass
    noise_sigma = _robust_noise_sigma(high_pass)
    snr = float(np.std(low_pass)) / (noise_sigma + 1e-12)

    try:
        threshold = threshold_otsu(img)
        foreground = img[img >= threshold]
        background = img[img < threshold]
        if foreground.size < 20 or background.size < 20:
            raise ValueError("Otsu classes are too small")
        cnr = abs(float(foreground.mean() - background.mean())) / np.sqrt(
            float(foreground.var() + background.var()) + 1e-12
        )
    except (ValueError, TypeError):
        cnr = 0.0

    return float(snr), float(cnr), float(noise_sigma)


def assess_image(
    img: np.ndarray,
    min_periodicity: float = 0.35,
    min_snr: float = 2.0,
    min_cnr: float = 1.5,
) -> Dict[str, object]:
    """Assess periodicity and denoising quality without a reference image."""
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    periodicity = _periodicity_metrics(img)
    snr, cnr, noise_sigma = _snr_cnr_metrics(img)
    passed = periodicity["periodicity_score"] >= min_periodicity and (
        snr >= min_snr or cnr >= min_cnr
    )

    return {
        "periodicity_score": round(periodicity["periodicity_score"], 4),
        "bragg_peakiness": round(periodicity["bragg_peakiness"], 4),
        "autocorr_peak_ratio": round(periodicity["autocorr_peak_ratio"], 4),
        "snr": round(snr, 3),
        "cnr": round(cnr, 3),
        "noise_sigma": round(noise_sigma, 4),
        "thresholds": {
            "min_periodicity": float(min_periodicity),
            "min_snr": float(min_snr),
            "min_cnr": float(min_cnr),
        },
        "decision": "PASS" if passed else "FAIL",
    }


def assess_image_path(
    path: str,
    min_periodicity: float = 0.35,
    min_snr: float = 2.0,
    min_cnr: float = 1.5,
) -> Dict[str, object]:
    """Assess a restored STEM image loaded from a file path."""
    result = assess_image(
        imread_gray(path),
        min_periodicity=min_periodicity,
        min_snr=min_snr,
        min_cnr=min_cnr,
    )
    return {"image": str(path), **result}
