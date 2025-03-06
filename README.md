<img src="APA_logo.png" alt="Advanced Particle Analyzer Logo" width="200">

# Advanced Particle Analyzer

An interactive Streamlit application for advanced analysis of particle images from micrographies (TEM/SEM). Easily adjust filtering, enhancement, and detection parameters on the fly while visualizing both the processed image and particle size distribution. This tool provides robust options such as noise reduction (Gaussian, Median, Bilateral), contrast enhancement (Histogram Equalization, CLAHE), morphological operations, edge detection, Hough Circle Transform, and contour filtering.

## Features

- **Noise Reduction:**  
  Choose from Gaussian Blur, Median Blur, or Bilateral Filter to remove noise.

- **Contrast Enhancement:**  
  Enhance image contrast via Histogram Equalization or CLAHE.

- **Morphological Operations:**  
  Apply erosion, dilation, opening, closing, or morphological gradient to refine image features.

- **Edge & Threshold Detection:**  
  Utilize Canny, Adaptive Thresholding, Sobel, or Otsu's Binarization methods.

- **Edge Enhancement:**  
  Enhance edges using Morphological Gradient or Laplacian of Gaussian (LoG).

- **Hough Circle Transform:**  
  Optionally detect circular features (e.g., particles) using Hough Circle Transform.

- **Contour Filtering:**  
  Filter results based on shape (circularity), perimeter, and area thresholds.

- **Interactive Histogram:**  
  Visualize particle size distribution (either as diameters in µm or areas in pixels²).

- **Save and Load Analysis:**  
  Save your settings and results (images, plots, and quantified data) for later review.

- **Examples Included:**  
  Two example scripts in the `examples` folder demonstrate analysis on TEM and SEM micrographies.

## Installation

1. **Clone the Repository:**

   Open your terminal and run:
   ```
   git clone https://github.com/yourusername/advanced-particle-analyzer.git
   ```

2. **Use a Virtual Environment (Optional):**

   For example, using venv:
   ```
   python3 -m venv env
   # On Unix or MacOS:
   source env/bin/activate
   # On Windows:
   env\Scripts\activate
   ```

3. **Install the Required Packages:**

   Install Python 3.x dependencies via pip:
   ```
   pip install -r requirements.txt
   ```
   Ensure that your requirements.txt includes packages such as:
   - streamlit
   - opencv-python
   - numpy
   - matplotlib

## Usage

1. **Run the Application:**

   Launch the Streamlit app by running:
   ```
   streamlit run streamlit_app.py
   ```

2. **Interact with the App:**

   - Use the sidebar to upload your image for particle counting, loading JSON settings from a previous analysis is also possible.
   - Adjust parameters for noise reduction, contrast enhancement, morphological operations, thresholding, edge detection, and more.
   - The app will update in real time, displaying the final processed image along with an optional view of intermediate steps.
   - Save your analysis results to a designated output folder.

3. **Explore the Examples:**

   Check out the `examples` folder, which includes usage examples for TEM and SEM image analysis.

## Contributing

Contributions, bug reports, and feature requests are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:
   ```
   git checkout -b feature/my-feature
   ```
3. Commit your changes:
   ```
   git commit -m 'Add some feature'
   ```
4. Push the branch:
   ```
   git push origin feature/my-feature
   ```
5. Open a pull request.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

