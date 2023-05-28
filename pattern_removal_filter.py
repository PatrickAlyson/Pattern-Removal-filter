"""
    This script implements a technique for removal of interference patterns (periodic noise) from images.
    This algorithm is based on Cannon[1].

    @author: Patrick Alyson

    Basic usage
    ----------
    >>> import pattern_filter from pattern_removal_filter
    >>> filtered_image = pattern_filter(input_image)

    References
    ----------
    ..[1] CANNON, M.; LEHAR, A.; PRESTON, F. Background pattern removal by power
          spectral Altering. Appl. Opt., Optica Publishing Group, v. 22, n. 6, p. 777-779, Mar 1983.
          http://opg.optica.org/ao/abstract.cfm?URI=ao-22-6-777
"""

import numpy as np
import cv2
from scipy import interpolate
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion
import cv2


def radial_profile(data, center):
    """This function returns the radial distribution of intensity (radial profile).

    Args:
        data (Numpy Array): Image/Two-dimensional Array
        center (tuple/list): Position

    Returns:
        Numpy Array: Radial profile of the input
    """
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int64)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def DFT_2D(image):
    """This function returns the Fourier transform and magnitude spectrum of the input image

    Args:
        image (Numpy Array): Input image

    Returns:
        Numpy Array: Shifted Fourier transform
        Numpy Array: Magnitude spectrum
    """
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0],
                                                    dft_shift[:, :, 1]))+1)
    return dft_shift, magnitude_spectrum


def find_patches(image, size_patches):
    """This function divides the image into overlapping subsections to be used in the Welch method

    Args:
        image (Numpy Array): Input image
        size_patches (Integer): Subsections size

    Returns:
        List: Overlapping subsections of the image
    """
    rows, cols = image.shape  
    han = np.hanning(size_patches)  # 1D Hanning window
    han_2d = np.outer(han, han)  # 2D Hanning window
    han_patches = []
    step_patches = size_patches//4

    for i in range(0, rows-size_patches, step_patches):
        for j in range(0, cols-size_patches, step_patches):
            han_patches.append(
                image[i:i+size_patches, j:j+size_patches] * han_2d)
    return han_patches


def log_power_spectral_estimate(patches):
    """This function computes the log of the power spectral estimate

    Args:
        patches (list): Overlapping subsections of the image

    Returns:
        Numpy Array: Log of the power spectral estimate
    """
    power_spectral_estimate = np.zeros(patches[0].shape)
    for patch in patches:
        fft = np.fft.fftshift(
            cv2.dft(np.float32(patch), flags=cv2.DFT_COMPLEX_OUTPUT))
        mag = np.power(cv2.magnitude(fft[:, :, 0], fft[:, :, 1]), 2)
        power_spectral_estimate = power_spectral_estimate + mag

    power_spectral_estimate = power_spectral_estimate/len(patches)
    return 20*np.log(power_spectral_estimate)


def angular_average(spectral_estimate):
    """This function computes the radial average of the log power spectral estimate

    Args:
        spectral_estimate (Numpy Array ): Log power spectral estimate

    Returns:
        Numpy Array: Radial average of the log power spectral estimate
    """
    nrow, ncol = spectral_estimate.shape
    radial_average = radial_profile(spectral_estimate, (nrow//2, ncol//2))

    x = np.linspace(0, len(radial_average), len(radial_average))
    f = interpolate.interp1d(x, radial_average, kind='linear')
    x_new = np.linspace(0, len(radial_average), 8*len(radial_average))
    radial_average_interpolated = f(x_new)

    return radial_average, radial_average_interpolated


def angular_average_2D(ang_average, size_patches, radius):
    """This function computes a two-dimensional angular average of the log power spectral estimate

    Args:
        ang_average (Numpy Array): Radial average of the log power spectral estimate
        size_patches (Integer): Size of the overlapping subsections of the image
        radius (Integer): Maximum radius

    Returns:
        Numpy Array: Two-dimensional angular average of the log power spectral estimate
    """

    rows, cols = len(ang_average)*2, len(ang_average)*2
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.ones((rows, cols))
    mask = mask * np.min(ang_average)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    list_of_radius = np.linspace(0, radius, len(ang_average))

    tolerance = 0.2
    indice = 0
    for r in list_of_radius:
        res = np.where(np.abs(mask_area - r) <= tolerance)
        mask[res] = ang_average[indice]
        indice += 1

    mask[crow, ccol] = np.max(ang_average)
    angular_average = mask[crow - (size_patches//2):crow + (size_patches//2),
                           ccol - (size_patches//2):ccol + (size_patches//2)]
    return angular_average


def distance(p1, p2):
    """Calculate the Euclidean distance

    Args:
        p1 (tuple/list): First point
        p2 (tuple/list): Second point

    Returns:
        Float: Euclidean distance between the two points
    """
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def image_back(fft):
    """Compute the inverse Fourier transform and returns the real part

    Args:
        fft (Numpy Array): Fourier transform of the image

    Returns:
        Numpy Array: Returns the real part of the inverse Fourier transform
    """
    res = np.fft.ifftshift(fft)
    res = cv2.idft(res)
    return res[:, :, 0]*255/np.max(res[:, :, 0])


def detect_peaks(image):
    """
    Compute peaks of an image by local maximum filter
    Returns:
        Numpy Array: Peaks
    """
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    background = (image == 0)
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


def pattern_filter(image, size_patches=128, threshold=0.5, notch_radius=5, exclusion_radius=10):
    """
    Removes interference patterns of images/periodic noise from images

    Args:
        image (numpy Array): Imagem
        size_patches (int, optional): Size of the overlapping subsections of the image. Defaults to 128.
        threshold (float, optional): A number between 0 and 1. Values closer to zero indicate a more aggressive filter. 
    Defaults to 0.5.
        notch_radius (int, optional): Radius of the circular notch filter. Higher values indicate a more aggressive filter.
    Defaults to 5.
        exclusion_radius (int, optional): The radius of the exclusion region from the center of the Fourier transform. This value must always be larger than the notch_radius. Defalts to 10.

    Returns:
        Numpy array: Filtered image.
    """

    patches = find_patches(
        image, size_patches=size_patches) 
    log_pow_estimate = log_power_spectral_estimate(patches)

    radial_average, radial_average_interpolated = angular_average(
        log_pow_estimate)

    angular_average_itself = angular_average_2D(
        radial_average_interpolated, size_patches=size_patches, radius=len(radial_average))
    
    modified_log_power_inter = cv2.subtract(log_pow_estimate.astype(
        np.float64), angular_average_itself.astype(np.float64))

    modified_log_power_inter = modified_log_power_inter / \
        np.max(modified_log_power_inter)
    smaller = np.where(np.abs(modified_log_power_inter) < threshold)
    modified_log_power_inter[smaller] = 0
    smaller = np.where((modified_log_power_inter) < 0)
    modified_log_power_inter[smaller] = 0
    # Resizing
    modified_log_power_r = cv2.resize(
        modified_log_power_inter, image.shape[::-1])
    # Finding peaks
    peaks = detect_peaks(modified_log_power_r)
    peaks = np.where(peaks == True)
    # Creating notch filter
    rows, cols = modified_log_power_r.shape
    filter = np.ones((rows, cols, 2))
    x, y = np.ogrid[:rows, :cols]
    img_center = (rows//2, cols//2)
    for peak in zip(peaks[0], peaks[1]):
        if distance(img_center, peak) < exclusion_radius:
            continue
        else:
            mask_area = (x - peak[0]) ** 2 + (y - peak[1]
                                              ) ** 2 <= np.power(notch_radius, 2)
            filter[mask_area] = 0

    fft, img_mag = DFT_2D(image)
    fshift = fft * filter
    result = image_back(fshift)

    return abs(result)
