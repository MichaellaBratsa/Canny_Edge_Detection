from collections import deque
import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')  # or 'Qt5Agg'


def normalization(arr, min_val, max_val):

    arr = np.array(arr, dtype=np.float64)
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    normalized_arr = min_val + (max_val - min_val) * (arr - arr_min) / (arr_max - arr_min)

    return normalized_arr


def gaussian_kernel_2D(ksize, sigma):
    """
    Calculate a 2D Gaussian kernel
    :param ksize: int, size of 2d kernel, always needs to be an odd number
    :param sigma: float, standard deviation of gaussian
    :return:
    kernel: numpy.array(float), ksize x ksize gaussian kernel with mean=0
    """

    assert ksize > 0 and ksize % 2 == 1, "ksize must be an odd, non-zero positive number"
    assert sigma > 0, "sigma must be positive"

    kernel_1d = np.zeros(ksize, dtype=np.float64)
    c = int(ksize / 2)

    for i in range(ksize):
        kernel_1d[i] = i - c

    # For debugging reasons
    # print(kernel_1d)

    for i in range(ksize):
        kernel_1d[i] = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.pow((kernel_1d[i] / sigma), 2))  # Gaussian formula

    kernel_2d = np.outer(kernel_1d, kernel_1d)  # Outer product with 1d kernel to take 2d kernel

    return kernel_2d


def convolution_2D(arr, kernel, border_type):
    """
    Calculate the 2D convolution kernel*arr
    :param arr: numpy.array(float), input array
    :param kernel: numpy.array(float), convolution kernel of nxn size (only odd dimensions are allowed)
    :param border_type: int, padding method (OpenCV)
    :return:
    conv_arr: numpy.array(float), convolution output
    """

    #Check if the kernel is of square odd size
    h = kernel.shape[0]
    w = kernel.shape[1]

    assert h == w and h % 2 == 1, "Input kernel  must be square with odd dimensions"

    # Flip 180 deg the kernel
    kernel = np.rot90(kernel, k=2)

    # For debugging reasons
    # print(kernel)

    pad_size = 1
    constant = 0

    arr = cv2.copyMakeBorder(src=arr, top=pad_size, bottom=pad_size, left=pad_size, right=pad_size, borderType=border_type, value=constant)

    n = np.zeros((arr.shape[0] - 2, arr.shape[1] - 2), dtype=np.float64)

    # Filter apply
    for i in range(0, arr.shape[0] - 2):
        for j in range(0, arr.shape[1] - 2):
            sum = 0
            for k in range(0, kernel.shape[0]):
                for l in range(0, kernel.shape[0]):
                    sum += arr[i + k][j + l] * kernel[k][l]
            n[i][j] = sum

    return n


def sobel_x(arr):
    """
     Calculate the 1st order partial derivatives along x-axis
     :param arr:numpy.array(float),input image
     :return:
     dx:numpy.array(float),output partial derivative
     """

    # Sobel filter in x-axis
    sobel_kernel_x = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]])

    dx = convolution_2D(arr, sobel_kernel_x, cv2.BORDER_CONSTANT)

    return dx


def sobel_y(arr):
    """
    Calculate 1st the order partial derivatives along y-axis
    :param arr: numpy.array(float), input image
    :return:
    dy: numpy.array(float), output partial derivatives
    """

    # Sobel filter in y-axis
    sobel_kernel_y = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]
                               ])

    dy = convolution_2D(arr, sobel_kernel_y, cv2.BORDER_CONSTANT)

    return dy


def magnitude_calculation(Ix, Iy):

    # Magnitude calculation using the corresponding formula
    return np.sqrt(Ix**2 + Iy**2)


def direction_calculation(Ix, Iy):

    # Direction calculation using the arctan2 function
    return np.arctan2(Iy, Ix)


def gradient_direction_calculation(dx, dy):

    magnitude = magnitude_calculation(dx, dy)
    direction = direction_calculation(dx, dy)

    magnitude_normalized = normalization(magnitude, 0, 1)
    direction_degrees = np.degrees(direction) % 360

    hue_mapping = ((direction_degrees + 180) / 2 + 90) % 180
    hue_values = hue_mapping.astype(np.uint8)

    # Create HSV image with specific color mapping
    hsv_image = np.zeros((direction.shape[0], direction.shape[1], 3), dtype=np.uint8)
    hsv_image[:, :, 0] = hue_values
    hsv_image[:, :, 1] = (magnitude_normalized * 255).astype(np.uint8)
    hsv_image[:, :, 2] = 255

    # Convert HSV to RGB for visualization
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    return rgb_image


def non_maximum_suppression(arr_mag, arr_dir):
    """
    Find all local maxima along image gradient direction
    :param arr_mag: numpy.array(float), input image gradient magnitude
    :param arr_dir: numpy.array(float), input image gradient direction
    :return:
    arr_local_maxima: numpy.array(float)
    """

    rows = arr_mag.shape[0]
    cols = arr_mag.shape[1]
    arr_local_maxima = np.zeros((rows, cols), dtype=np.float32)

    # Normalize gradient directions to 0-180 degrees
    arr_dir = arr_dir % 180

    neighbor_1 = 0
    neighbor_2 = 0

    # Define direction bins (8 principal directions)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            direction = arr_dir[i][j]
            magnitude = arr_mag[i][j]

            if -22.5 <= direction and direction > 22.5:
                neighbor_1 = arr_mag[i + 1][j + 1]
                neighbor_2 = arr_mag[i + 1][j - 1]
            elif 22.5 <= direction and direction > 67.5:
                neighbor_1 = arr_mag[i + 1][j]
                neighbor_2 = arr_mag[i][j - 1]
            elif 67.5 <= direction and direction > 112.5:
                neighbor_1 = arr_mag[i + 1][j - 1]
                neighbor_2 = arr_mag[i - 1][j - 1]
            elif 112.5 <= direction and direction > 157.5:
                neighbor_1 = arr_mag[i - 1][j]
                neighbor_2 = arr_mag[i][j - 1]
            elif -67.5 <= direction and direction < -22.5:
                neighbor_1 = arr_mag[i][j + 1]
                neighbor_2 = arr_mag[i + 1][j]
            elif -112.5 <= direction and direction < -67.5:
                neighbor_1 = arr_mag[i + 1][j + 1]
                neighbor_2 = arr_mag[i - 1][j + 1]
            elif -157.5 <= direction and direction < -112.5:
                neighbor_1 = arr_mag[i][j + 1]
                neighbor_2 = arr_mag[i - 1][j]
            elif -157.5 <= direction and direction <= 157.5:
                neighbor_1 = arr_mag[i - 1][j + 1]
                neighbor_2 = arr_mag[i - 1][j - 1]

            if magnitude >= neighbor_1 and magnitude >= neighbor_2:
                arr_local_maxima[i][j] = magnitude

    return arr_local_maxima


def hysteresis_thresholding(arr, low_ratio, high_ratio):
    """
    Apply hysteresis thresholding to a non-maximum suppression image.

    :param arr: numpy.array, input non-maximum suppression image (gradient magnitudes)
    :param low_ratio: float, low threshold ratio (should be between 0 and 1)
    :param high_ratio: float, high threshold ratio (should be between 0 and 1)
    :return: numpy.Array, the output edges image (0: no edge, 255: edge)
    """

    high_threshold = arr.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    # Create an empty output image
    edges = np.zeros_like(arr, dtype=np.uint8)

    # Classify strong and weak edges
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] >= high_threshold:
                edges[i, j] = 255  # Strong edge
            elif arr[i, j] >= low_threshold:
                edges[i, j] = 128  # Weak edge

    # 8 possible neighbors (up, down, left, right, diagonals)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


    q = deque()

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if edges[i][j] == 255:  # If it's a strong edge
                q.append((i, j))  # Add to the queue

    while q:
        x, y = q.popleft()

        for dx, dy in neighbors:
            nx = x + dx
            ny = y + dy

            if 0 <= x < arr.shape[0] and 0 <= y < arr.shape[1] and edges[nx][ny] == 128:
                edges[nx][ny] = 255  # Convert weak edge to strong
                q.append((nx, ny))  # Add to queue

    # Remove unconnected weak edges
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j] == 128:
                edges[i][j] = 0

    return edges


def main():
    # Read the image and convert to grayscale
    img = cv2.imread('building.jpg', cv2.IMREAD_GRAYSCALE)

    blurred_img = convolution_2D(img, gaussian_kernel_2D(3, 1.0), cv2.BORDER_CONSTANT)

    dx = sobel_x(img)
    dy = sobel_y(img)

    magnitude = magnitude_calculation(dx, dy)
    direction = direction_calculation(dx, dy)

    magnitude_normalized = normalization(magnitude, 0, 1)
    direction_degrees = np.degrees(direction) % 360

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(blurred_img, cmap='gray')
    plt.title("Blurred Image")
    plt.axis('off')
    plt.show()


    plt.subplot(1, 2, 1)
    plt.imshow(dx, cmap='gray')
    plt.title("Sobel-X")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(dy, cmap='gray')
    plt.title("Sobel-Y")
    plt.axis('off')
    plt.show()

    plt.subplot(1, 2, 1)
    plt.imshow(magnitude_calculation(dx, dy), cmap='gray')
    plt.title("Gradient Magnitude")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gradient_direction_calculation(dx, dy), cmap='gray')
    plt.title("Gradient Direction")
    plt.axis('off')
    plt.show()


    plt.subplot(1, 1, 1)
    plt.imshow(non_maximum_suppression(magnitude_normalized, direction_degrees), cmap='gray')
    plt.title("Non-maximum suppression")
    plt.axis('off')
    plt.show()

    plt.subplot(1, 2, 1)
    plt.imshow(hysteresis_thresholding(non_maximum_suppression(magnitude_normalized, direction_degrees), 0.1, 0.3), cmap='gray')
    plt.title("Hysteresis Thresholding")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.Canny(image=np.uint8(blurred_img), threshold1=100, threshold2=200), cmap='gray')
    plt.title("OpenCV")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
