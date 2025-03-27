import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image, ImageFile
import numpy as np
import weakref

# Allow truncated images to load faster
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomImageClass:
    def __init__(self, image_path=None, height=None, width=None, color=(255, 255, 255), pixel_data=None):
        """Initialize the image class.

        If `image_path` is provided, it loads the image.
        If `height` and `width` are provided, it creates a blank image with the given dimensions.
        If `pixel_data` is provided, it initializes from an existing list.
        """
        # Dictionary to cache thumbnails of different sizes
        self._thumbnail_cache = {}

        if image_path:
            # Load image from file using optimized settings
            try:
                # Use reduced mode for faster initial loading
                self.image = Image.open(image_path)

                # Only convert to RGB if needed
                if self.image.mode != 'RGB':
                    self.image = self.image.convert("RGB")

                self.width, self.height = self.image.size
                # Don't convert to list yet - do it lazily when needed
                self._pixel_data = None
            except Exception as e:
                print(f"Error loading image: {e}")
                raise
        elif height is not None and width is not None:
            # Create a blank image with the given size and color
            self.width = width
            self.height = height
            self.image = Image.new("RGB", (self.width, self.height), color)
            self._pixel_data = pixel_data if pixel_data else [[color for _ in range(self.width)] for _ in
                                                              range(self.height)]
        else:
            raise ValueError("Either provide an image path or specify width and height.")

    @property
    def pixel_data(self):
        """Lazy loading of pixel data"""
        if self._pixel_data is None:
            self._pixel_data = self._convert_to_list()
        return self._pixel_data

    @pixel_data.setter
    def pixel_data(self, value):
        self._pixel_data = value
        # Clear thumbnail cache when pixel data changes
        self._thumbnail_cache.clear()

    def get_thumbnail(self, size=64):
        """Get a thumbnail of the image with the specified size.
        Uses caching for better performance."""
        # Check if we already have this size in cache
        if size in self._thumbnail_cache:
            # Get from cache if it exists
            thumb_ref = self._thumbnail_cache[size]
            if thumb_ref is not None:
                return thumb_ref

        # Create a new thumbnail using the efficient PIL method
        # This creates a thumbnail in-place without a full copy
        thumb = self.image.copy()
        thumb.thumbnail((size, size), Image.LANCZOS)

        # Store in cache
        self._thumbnail_cache[size] = thumb
        return thumb

    def _convert_to_list(self):
        """Convert image pixels into a nested list."""
        pixels = list(self.image.getdata())  # Get pixel values as a flat list
        return [pixels[i * self.width:(i + 1) * self.width] for i in range(self.height)]

    def get_pixel(self, x, y):
        """Get the pixel value at (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.pixel_data[y][x]
        else:
            raise ValueError("Coordinates out of bounds")

    def set_pixel(self, x, y, value):
        """Set the pixel value at (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixel_data[y][x] = value
            # Clear thumbnail cache as image has changed
            self._thumbnail_cache.clear()
        else:
            raise ValueError("Coordinates out of bounds")

    def update_image(self):
        """Update self.image with the modified pixel_data."""
        # Skip if pixel data hasn't been initialized or modified
        if not hasattr(self, '_pixel_data') or self._pixel_data is None:
            return

        flat_pixels = [pixel for row in self.pixel_data for pixel in row]  # Flatten list
        self.image = Image.new("RGB", (self.width, self.height))  # Create new image
        self.image.putdata(flat_pixels)  # Set the new pixel data

        # Clear thumbnail cache as image has changed
        self._thumbnail_cache.clear()

    def show(self):
        self.update_image()
        self.image.show()

    def save(self, output_path):
        """Save the modified image to a file."""
        self.update_image()
        self.image.save(output_path)
        print(f"Image saved to {output_path}")

    def copy(self):
        """Create a fast copy of this object."""
        # Create a new instance with same dimensions but don't copy pixels yet
        new_img = CustomImageClass(height=self.height, width=self.width)

        # Copy the image directly (faster than pixel by pixel)
        new_img.image = self.image.copy()

        # Only copy pixel data if it has been initialized
        if hasattr(self, '_pixel_data') and self._pixel_data is not None:
            new_img._pixel_data = [row[:] for row in self.pixel_data]

        return new_img

    def count_pixel_colors(self):
        print("Counting pixel colors...")
        # Flatten the pixel data (assumed to be a 2D list of [R, G, B] pixels)
        flat_pixels = [pixel for row in self.pixel_data for pixel in row]
        color_counts = Counter(flat_pixels)
        return color_counts

    def show_histogram(self):
        """Display individual histograms for the R, G, B channels and a grayscale histogram."""
        print("Generating histograms...")

        # Convert the pixel data (a list of lists of pixels) into a NumPy array
        pixel_array = np.array(self.pixel_data, dtype=np.uint8)

        # Extract each color channel and flatten them for the histogram
        red_values = pixel_array[:, :, 0].flatten()
        green_values = pixel_array[:, :, 1].flatten()
        blue_values = pixel_array[:, :, 2].flatten()

        # Compute grayscale values using the standard luminance formula
        gray_values = (0.2989 * red_values + 0.5870 * green_values + 0.1140 * blue_values).astype(np.uint8)

        # Create a 2x2 grid for the histograms
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Red channel histogram
        axes[0, 0].hist(red_values, bins=256, color='red')
        axes[0, 0].set_title("Red Channel Histogram")
        axes[0, 0].set_xlim([0, 256])

        # Green channel histogram
        axes[0, 1].hist(green_values, bins=256, color='green')
        axes[0, 1].set_title("Green Channel Histogram")
        axes[0, 1].set_xlim([0, 256])

        # Blue channel histogram
        axes[1, 0].hist(blue_values, bins=256, color='blue')
        axes[1, 0].set_title("Blue Channel Histogram")
        axes[1, 0].set_xlim([0, 256])

        # Grayscale histogram (displayed in grey)
        axes[1, 1].hist(gray_values, bins=256, color='gray')
        axes[1, 1].set_title("Grayscale Histogram")
        axes[1, 1].set_xlim([0, 256])

        # Set common labels for clarity
        for ax in axes.flat:
            ax.set_xlabel("Pixel Intensity (0-255)")
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def to_numpy_array(self):  # for fast filter apply
        """Convert image to NumPy array directly from PIL image for faster processing"""
        return np.array(self.image)

    def set_image_from_numpy(self, np_array):
        if np_array.ndim != 3 or np_array.shape[2] != 3:
            raise ValueError("Numpy array must have shape (height, width, 3)")
        self.height, self.width, _ = np_array.shape
        self.image = Image.fromarray(np_array.astype(np.uint8), 'RGB')
        # Clear pixel data and thumbnail cache
        self._pixel_data = None
        self._thumbnail_cache.clear()


if __name__ == "__main__":
    input_image_path = "../sample_images/bird-at-zoo-1579028.jpg"
    output_image_path = "../sample_images/modified_bird.png"

    img = CustomImageClass(input_image_path)

    # Example: Count pixels for each color
    color_counts = img.count_pixel_colors()
    print(f"Top 5 most common colors: {color_counts.most_common(5)}")  # Show top 5 most frequent colors

    # Show histogram
    img.show_histogram()

    img.show()  # Show updated image
    img.save(output_image_path)  # Save the modified image