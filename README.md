# Pain't - Image Processing Application

Pain't is a user-friendly Python application designed for image processing, enabling users to apply various pixel operations, convolution filters, histogram analysis, and image projections.

## Features

### Pixel Operations:
- **Grayscale Conversion**: Transform images to grayscale by averaging RGB values.
- **Brightness Adjustment**: Modify image brightness with adjustable factors.
- **Contrast Adjustment**: Enhance or reduce contrast relative to the average brightness.
- **Negative Image**: Create negative images by inverting RGB channels.
- **Binarization**: Convert images to black-and-white based on a brightness threshold.

### Convolution Filters:
- **Averaging Filter**: Smooth images by averaging neighboring pixels.
- **Gaussian Filter**: Natural smoothing using Gaussian distribution.
- **Sharpening Filter**: Highlight edges and enhance image sharpness.
- **Edge Detection Filter**: Detect and highlight edges within images.
- **Emboss Filter**: Create a pseudo-3D embossed effect.
- **Sobel Filters**: Detect horizontal or vertical edges using Sobel operators.
- **Motion Blur Filter**: Simulate the blur caused by camera motion.
- **Custom Filter**: User-defined convolution kernels for specialized image effects.

## User Interface
Built with Python's Tkinter library, the intuitive GUI includes:
- Drag-and-drop operations and real-time previews.
- Easy loading of user images.
- Histogram and projection analysis.
- Adjustable parameters for precise control.
- Fast processing mode with optimized performance using NumPy.

## Dependencies
- **Python**
- **Tkinter**: Graphical User Interface.
- **PIL (Pillow)**: Image processing.
- **NumPy**: Efficient numerical computations.
- **Matplotlib**: Data visualization (histograms, projections).

## Usage
1. Clone the repository:
```bash
git clone https://github.com/CoolMikey/image-editor.git
```
2. Navigate to the project directory:
```bash
cd image-editor
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Navigate to src:
```bash
cd src
```
5. Run the application:
```bash
python GUI_fast.py
```

## Future Improvements
- Morphological operations (erosion, dilation).
- Automatic histogram analysis.
- Integration with object detection and OCR algorithms.

## Author
- Michał Matuszyk

---

Enjoy exploring the powerful capabilities of image processing with Pain't!

