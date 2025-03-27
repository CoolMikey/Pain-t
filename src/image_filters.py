from CustomImageClass import CustomImageClass
from tqdm import tqdm  # Progress bar library
import numpy as np

class ImageFilters:
    """Class containing image filtering methods for CustomImageClass."""

    custom_kernel = None

    @staticmethod
    def apply_filter(image: CustomImageClass, kernel, filter_name="Processing", apply_fast_apply = False):
        """Applies a convolution filter to an image with a progress indicator."""

        if apply_fast_apply:
            return ImageFilters.fast_apply(image, kernel=kernel)

        output_image = image.copy()

        # Define kernel size
        k_height = len(kernel)
        k_width = len(kernel[0])
        k_offset = k_height // 2


        print(f"Applying {filter_name} filter...")

        # Iterate over the image with a progress bar
        for y in tqdm(range(k_offset, image.height - k_offset), desc=f"{filter_name} Progress"):
            for x in range(k_offset, image.width - k_offset):
                sum_r, sum_g, sum_b = 0, 0, 0

                # Apply the kernel to the surrounding pixels
                for ky in range(k_height):
                    for kx in range(k_width):
                        pixel_x = x + kx - k_offset
                        pixel_y = y + ky - k_offset
                        r, g, b = image.pixel_data[pixel_y][pixel_x]

                        weight = kernel[ky][kx]
                        sum_r += r * weight
                        sum_g += g * weight
                        sum_b += b * weight

                # Normalize pixel values
                output_image.pixel_data[y][x] = (
                    max(0, min(255, int(sum_r))),
                    max(0, min(255, int(sum_g))),
                    max(0, min(255, int(sum_b)))
                )
        output_image.update_image()
        print(f"{filter_name} filter applied successfully!\n")
        return output_image

    @staticmethod
    def fast_apply(image: CustomImageClass, kernel, filter_name="Processing"):
        """
        Apply a convolution-like filter by:
          1. Creating a full convolution result with output size = original size + (kernel size - 1) in each dimension.
          2. Adding shifted versions of the image multiplied by the corresponding kernel values.
          3. Cropping the extra border so the result has the original image dimensions.
        """
        # Get a numpy array copy of the image and convert to float32 to allow negative values
        output_image = image.copy()
        image_copy = output_image.to_numpy_array().astype(np.float32)  # shape: (H, W, 3)
        H, W, _ = image_copy.shape
        k_height = len(kernel)
        k_width = len(kernel[0])
        half_h = k_height // 2
        half_w = k_width // 2

        # Create the accumulator with the appropriate shape and dtype
        out_shape = (H + k_height - 1, W + k_width - 1, 3)
        accumulator = np.zeros(out_shape, dtype=np.float32)

        # Loop through kernel offsets and accumulate shifted image contributions
        for i in range(-half_h, half_h + 1):
            for j in range(-half_w, half_w + 1):
                kernel_value = kernel[i + half_h][j + half_w]
                row_start = i + half_h
                row_end = row_start + H
                col_start = j + half_w
                col_end = col_start + W
                accumulator[row_start:row_end, col_start:col_end, :] += image_copy * kernel_value

        # Crop the accumulator to remove the extra border so that the final image has the original dimensions
        cropped = accumulator[half_h:half_h + H, half_w:half_w + W, :]
        # Clip values to [0, 255] and cast back to uint8 for valid RGB data
        cropped = np.clip(cropped, 0, 255).astype(np.uint8)
        # Update the image with the filtered result
        output_image.set_image_from_numpy(cropped)
        return output_image


    @staticmethod
    def averaging_filter(image: CustomImageClass, fast_apply = False):
        """Applies an averaging (smoothing) filter."""
        kernel = [
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ]
        return ImageFilters.apply_filter(image, kernel, "Averaging", apply_fast_apply = fast_apply)

    @staticmethod
    def generate_gaussian_kernel(kernel_size, sigma):
        """Generates a Gaussian kernel of given size and sigma."""
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        kernel = kernel / np.sum(kernel)
        return kernel.tolist()  # Convert numpy array to list of lists

    @staticmethod
    def gaussian_filter(image: CustomImageClass, kernel_size=3, sigma=None, fast_apply = False):
        """Applies a Gaussian blur filter with an adjustable kernel size.

        Args:
            image: The CustomImageClass image to filter.
            kernel_size: The size of the Gaussian kernel (must be odd, e.g., 3, 5, 7).
            sigma: Standard deviation of the Gaussian. If None, sigma is set to kernel_size/3.
        """
        if sigma is None:
            sigma = kernel_size / 3.0
        kernel = ImageFilters.generate_gaussian_kernel(kernel_size, sigma)
        return ImageFilters.apply_filter(image, kernel, f"Gaussian Blur {kernel_size}x{kernel_size}", apply_fast_apply = fast_apply)

    @staticmethod
    def sharpening_filter(image: CustomImageClass, fast_apply = False):
        """Applies a sharpening filter."""
        kernel = [
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ]
        return ImageFilters.apply_filter(image, kernel, "Sharpening", apply_fast_apply = fast_apply)

    @staticmethod
    def edge_detection_filter(image: CustomImageClass, fast_apply = False):
        """Applies an edge detection filter using a Laplacian kernel."""
        kernel = [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]
        return ImageFilters.apply_filter(image, kernel, "Edge Detection", apply_fast_apply = fast_apply)

    @staticmethod
    def emboss_filter(image: CustomImageClass, fast_apply = False):
        """Applies an emboss filter to give the image a 3D, raised effect."""
        kernel = [
            [-2, -1,  0],
            [-1,  1,  1],
            [ 0,  1,  2]
        ]
        return ImageFilters.apply_filter(image, kernel, "Emboss", apply_fast_apply = fast_apply)

    @staticmethod
    def horizontal_sobel_filter(image: CustomImageClass, fast_apply = False):
        """Applies a horizontal Sobel filter for edge detection in the vertical direction."""
        kernel = [
            [ 1,  2,  1],
            [ 0,  0,  0],
            [-1, -2, -1]
        ]
        return ImageFilters.apply_filter(image, kernel, "Horizontal Sobel", apply_fast_apply = fast_apply)

    @staticmethod
    def vertical_sobel_filter(image: CustomImageClass, fast_apply = False):
        """Applies a vertical Sobel filter for edge detection in the horizontal direction."""
        kernel = [
            [ 1,  0, -1],
            [ 2,  0, -2],
            [ 1,  0, -1]
        ]
        return ImageFilters.apply_filter(image, kernel, "Vertical Sobel", apply_fast_apply = fast_apply)

    @staticmethod
    def motion_blur_filter(image: CustomImageClass, kernel_size=5, fast_apply = False):
        """Applies a diagonal motion blur filter using a custom kernel."""
        # Create a diagonal kernel of size kernel_size x kernel_size
        kernel = [[0 for _ in range(kernel_size)] for _ in range(kernel_size)]
        for i in range(kernel_size):
            kernel[i][i] = 1 / kernel_size
        return ImageFilters.apply_filter(image, kernel, f"Motion Blur {kernel_size}x{kernel_size}", apply_fast_apply = fast_apply)

    @staticmethod
    def custom_filter(image: CustomImageClass, kernel_size=3, fast_apply = False):
        """Applies a custom kernel provided by the user (set via ImageFilters.custom_kernel)."""
        if ImageFilters.custom_kernel is None:
            print("No custom kernel defined. Using identity kernel.")
            # Identity kernel as a fallback
            kernel = [[0]*kernel_size for _ in range(kernel_size)]
            center = kernel_size // 2
            kernel[center][center] = 1
        else:
            kernel = ImageFilters.custom_kernel

        return ImageFilters.apply_filter(image, kernel, "Custom Filter", apply_fast_apply = fast_apply)

    @staticmethod
    def get_averaging_kernel():
        return [
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ]

    @staticmethod
    def get_sharpening_kernel():
        return [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]

    @staticmethod
    def get_edge_detection_kernel():
        return [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]

    @staticmethod
    def get_emboss_kernel():
        return [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ]

    @staticmethod
    def get_horizontal_sobel_kernel():
        return [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]

    @staticmethod
    def get_vertical_sobel_kernel():
        return [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]

    @staticmethod
    def get_motion_blur_kernel(kernel_size=5):
        kernel = [[0 for _ in range(kernel_size)] for _ in range(kernel_size)]
        for i in range(kernel_size):
            kernel[i][i] = 1 / kernel_size
        return kernel

    @staticmethod
    def get_gaussian_kernel(kernel_size=3, sigma=None):
        if sigma is None:
            sigma = kernel_size / 3.0
        return ImageFilters.generate_gaussian_kernel(kernel_size, sigma)

    @staticmethod
    def get_custom_kernel(kernel_size=3):
        if ImageFilters.custom_kernel is None:
            # Identity kernel fallback
            kernel = [[0] * kernel_size for _ in range(kernel_size)]
            center = kernel_size // 2
            kernel[center][center] = 1
            return kernel
        return ImageFilters.custom_kernel


if __name__ == "__main__":
    import time
    input_image_path = "../sample_images/bird-at-zoo-1579028.jpg"

    img = CustomImageClass(input_image_path)

    k = [
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ]

    # Time the apply_filter method
    start_time = time.perf_counter()
    ImageFilters.apply_filter(img, kernel=k)
    elapsed_apply = time.perf_counter() - start_time
    print(f"apply_filter took {elapsed_apply:.4f} seconds")

    # Time the fast_apply method
    start_time = time.perf_counter()
    ImageFilters.fast_apply(img, kernel=k)
    elapsed_fast_apply = time.perf_counter() - start_time
    print(f"fast_apply took {elapsed_fast_apply:.4f} seconds")

    start_time = time.perf_counter()
    ImageFilters.apply_filter(img, kernel=k, apply_fast_apply=True)
    elapsed_apply = time.perf_counter() - start_time
    print(f"apply_filter but actually fast_apply took {elapsed_apply:.4f} seconds")
    #
    # # Apply filters
    # avg_filtered = ImageFilters.averaging_filter(img)
    # gauss_filtered = ImageFilters.gaussian_filter(img)
    # sharp_filtered = ImageFilters.sharpening_filter(img)
    # edge_filtered = ImageFilters.edge_detection_filter(img)
    #
    # # Show results
    # avg_filtered.show()
    # gauss_filtered.show()
    # sharp_filtered.show()
    # edge_filtered.show()
    #
    # # Save results
    # avg_filtered.save("../sample_images/filter_outputs/avg_filtered.png")
    # gauss_filtered.save("../sample_images/filter_outputs/gauss_filtered.png")
    # sharp_filtered.save("../sample_images/filter_outputs/sharp_filtered.png")
    # edge_filtered.save("../sample_images/filter_outputs/edge_filtered.png")