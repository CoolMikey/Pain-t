from CustomImageClass import CustomImageClass


def grayscale(image: CustomImageClass):
    """Convert an image to grayscale by averaging RGB values."""
    output_image = image.copy()

    for y in range(image.height):
        for x in range(image.width):
            r, g, b = output_image.pixel_data[y][x]
            avg = (r + g + b) // 3  # Compute grayscale value
            output_image.pixel_data[y][x] = (avg, avg, avg)  # Set grayscale pixel

    return output_image


def adjust_brightness(image: CustomImageClass, factor: float):
    """Adjust image brightness by multiplying RGB values with a factor."""
    output_image = image.copy()

    for y in range(image.height):
        for x in range(image.width):
            r, g, b = output_image.pixel_data[y][x]
            new_pixel = (
                max(0, min(255, int(r * factor))),
                max(0, min(255, int(g * factor))),
                max(0, min(255, int(b * factor)))
            )
            output_image.pixel_data[y][x] = new_pixel

    return output_image


def adjust_contrast(image: CustomImageClass, factor: float):
    """Adjust image contrast by scaling pixel values."""
    output_image = image.copy()

    # Compute the average brightness (gray level)
    avg_brightness = sum(sum(pixel) // 3 for row in output_image.pixel_data for pixel in row) // (
                image.width * image.height)

    for y in range(image.height):
        for x in range(image.width):
            r, g, b = output_image.pixel_data[y][x]
            new_pixel = (
                max(0, min(255, int(avg_brightness + (r - avg_brightness) * factor))),
                max(0, min(255, int(avg_brightness + (g - avg_brightness) * factor))),
                max(0, min(255, int(avg_brightness + (b - avg_brightness) * factor)))
            )
            output_image.pixel_data[y][x] = new_pixel

    return output_image


def negative(image: CustomImageClass):
    """Convert an image to its negative."""
    output_image = image.copy()

    for y in range(image.height):
        for x in range(image.width):
            r, g, b = output_image.pixel_data[y][x]
            output_image.pixel_data[y][x] = (255 - r, 255 - g, 255 - b)

    return output_image


def binarize(image: CustomImageClass, threshold: int = 128):
    """Convert an image to black and white (binarization) based on a threshold."""
    output_image = image.copy()

    for y in range(image.height):
        for x in range(image.width):
            r, g, b = output_image.pixel_data[y][x]
            avg = (r + g + b) // 3  # Compute grayscale value
            output_image.pixel_data[y][x] = (255, 255, 255) if avg >= threshold else (0, 0, 0)

    return output_image


if __name__ == "__main__":
    input_image_path = "../sample_images/bird-at-zoo-1579028.jpg"
    output_image_path = "../sample_images/pixel_wise_outputs/modified_bird.png"

    img = CustomImageClass(input_image_path)

    bnwImage = grayscale(img)
    bnwImage.show()
    bnwImage.save("../sample_images/pixel_wise_outputs/grayscale.png")


    brightImage = adjust_brightness(img, 1.2)
    brightImage.show()
    brightImage.save("../sample_images/pixel_wise_outputs/brighter.png")

    # Adjust contrast (increase by 1.5x)
    contrastImage = adjust_contrast(img, 1.5)
    contrastImage.show()
    contrastImage.save("../sample_images/pixel_wise_outputs/contrast.png")

    # Convert to negative
    negativeImage = negative(img)
    negativeImage.show()
    negativeImage.save("../sample_images/pixel_wise_outputs/negative.png")

    # Apply binarization (threshold at 128)
    binaryImage = binarize(img, threshold=128)
    binaryImage.show()
    binaryImage.save("../sample_images/pixel_wise_outputs/binary.png")

    # Show original image for comparison
    img.show()
