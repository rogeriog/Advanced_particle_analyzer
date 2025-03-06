import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from datetime import datetime
import io
import json

# Set the title of the app
st.title("🔬 Advanced Particle Analyzer")

st.markdown("""
This application allows you to adjust advanced filtering parameters for analyzing noisy, low-contrast, and heterogeneous particle images.
Use the sidebar to modify parameters related to noise reduction, contrast enhancement, morphological operations, thresholding, edge enhancement, 
Hough Circle Transform, and contour filtering. The processed image with detected particles and their size distribution will update automatically.
""")

# --- Load Settings ---
st.sidebar.header("Load Saved Settings")
uploaded_json = st.sidebar.file_uploader(
    "Load previous settings, or just upload a image to start fresh.",
    type=["json"],
    key="json_uploader"
)

# Initialize session state for settings
if 'settings' not in st.session_state:
    st.session_state['settings'] = {}  # Initialize as a dictionary
if 'settings_loaded' not in st.session_state:
    st.session_state.settings_loaded = False
if 'image_loaded' not in st.session_state:
    st.session_state.image_loaded = False


# --- Image Upload ---
st.sidebar.header("1. Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="image_uploader")

# Process uploaded image *outside* the settings loading block
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Error: Unable to read the uploaded image. Please ensure it's a valid image file.")
        st.stop()  # Correctly stop execution
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    st.session_state.image_loaded = True
elif not st.session_state.image_loaded:
    st.error("Please upload an image file to proceed.")
    st.stop()


# --- Load and apply settings from JSON ---
if uploaded_json is not None and not st.session_state.settings_loaded:
    try:
        settings_json = json.load(uploaded_json)
        st.session_state.settings = settings_json  # Store loaded settings
        st.session_state.settings_loaded = True
        st.sidebar.success("Settings loaded successfully!")
    except Exception as e:
        st.error(f"Error loading settings: {e}")


# --- Parameter Sliders (with defaults and session state) ---
# Helper function for creating sliders and loading values
def create_slider(label, min_value, max_value, default_value, step=1, key=None, setting_group=None, setting_name=None, sub_setting = None):
    if setting_group and setting_name:
        # Load from session state if available, else use default
        if sub_setting:
            loaded_value = st.session_state.settings.get(setting_group, {}).get(sub_setting, {}).get(setting_name, default_value)
        else:
            loaded_value = st.session_state.settings.get(setting_group, {}).get(setting_name, default_value)

        default_value = loaded_value #Simplified
    return st.sidebar.slider(label, min_value, max_value, default_value, step=step, key=key)

def create_selectbox(label, options, key=None, setting_group=None, setting_name=None):
    # Get index of default value from session state
    default_option = options[0]  # Default to the first option
    if setting_group and setting_name:
        default_option = st.session_state.settings.get(setting_group,{}).get(setting_name, default_option)
    # Find index for selectbox default
    default_index = options.index(default_option) if default_option in options else 0

    return st.sidebar.selectbox(label, options, index=default_index, key=key)

def create_checkbox(label, default_value, key=None, setting_group=None, setting_name=None, sub_setting=None):
     if setting_group and setting_name:
        if sub_setting:  # Check if it's a sub-setting
            default_value = st.session_state.settings.get(setting_group, {}).get(sub_setting, {}).get(setting_name, default_value)
        else:
            default_value = st.session_state.settings.get(setting_group, {}).get(setting_name, default_value)
     return st.sidebar.checkbox(label, value=default_value, key=key)

def create_number_input(label, value, step, format=None, key=None, setting_group=None, setting_name=None, sub_setting = None):
    if setting_group and setting_name:
        if sub_setting:
            value = st.session_state.settings.get(setting_group,{}).get(sub_setting, {}).get(setting_name, value)
        else:
            value = st.session_state.settings.get(setting_group,{}).get(setting_name, value)

    return st.sidebar.number_input(label, value=value, step=step, format=format, key=key)


# 2. Noise Reduction
st.sidebar.header("2. Noise Reduction")
noise_reduction_type = create_selectbox("Select Noise Reduction Method", ("None", "Gaussian Blur", "Median Blur", "Bilateral Filter"), key="noise_reduction_type", setting_group="Noise Reduction", setting_name="Method")

if noise_reduction_type == "Gaussian Blur":
    gaussian_kernel_size = create_slider("Gaussian Kernel Size (odd)", 3, 21, 5, step=2, key="gaussian_kernel", setting_group="Noise Reduction", setting_name="Gaussian Kernel Size")
    gaussian_sigma = create_slider("Gaussian Sigma", 0.0, 10.0, 0.0, step=0.1, key="gaussian_sigma", setting_group="Noise Reduction", setting_name="Gaussian Sigma")
elif noise_reduction_type == "Median Blur":
    median_kernel_size = create_slider("Median Kernel Size (odd)", 3, 21, 5, step=2, key="median_kernel", setting_group="Noise Reduction", setting_name="Median Kernel Size")
elif noise_reduction_type == "Bilateral Filter":
    bilateral_diameter = create_slider("Bilateral Filter Diameter", 1, 30, 9, key="bilateral_diameter", setting_group="Noise Reduction", setting_name="Bilateral Diameter")
    bilateral_sigma_color = create_slider("Bilateral Sigma Color", 10, 300, 75, key="bilateral_sigma_color", setting_group="Noise Reduction", setting_name="Bilateral Sigma Color")
    bilateral_sigma_space = create_slider("Bilateral Sigma Space", 10, 300, 75, key="bilateral_sigma_space", setting_group="Noise Reduction", setting_name="Bilateral Sigma Space")

# 3. Contrast Enhancement
st.sidebar.header("3. Contrast Enhancement")
contrast_enhancement = create_selectbox("Select Contrast Enhancement Method", ("None", "Histogram Equalization", "CLAHE"), key="contrast_enhancement", setting_group="Contrast Enhancement", setting_name="Method")

if contrast_enhancement == "CLAHE":
    clahe_clipLimit = create_slider("CLAHE Clip Limit", 1.0, 10.0, 2.0, step=0.5, key="clahe_clip", setting_group="Contrast Enhancement", setting_name="CLAHE Clip Limit")
    tile_grid_size_value = create_slider("CLAHE Tile Grid Size", 2, 16, 8, key="tile_grid_size", setting_group="Contrast Enhancement", setting_name="CLAHE Tile Grid Size")
    tile_grid_size = tile_grid_size_value  # Create the tuple here
# 4. Morphological Operations
st.sidebar.header("4. Morphological Operations")
morph_operation = create_selectbox("Select Morphological Operation", ("None", "Erosion", "Dilation", "Opening", "Closing", "Morphological Gradient"), key="morph_operation", setting_group="Morphological Operations", setting_name="Operation")

if morph_operation != "None":
    morph_kernel_size = create_slider("Morphology Kernel Size (odd)", 3, 21, 5, step=2, key="morph_kernel", setting_group="Morphological Operations", setting_name="Kernel Size")
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))


# 5. Thresholding / Edge Detection
st.sidebar.header("5. Thresholding / Edge Detection")
thresholding_type = create_selectbox("Select Thresholding/Edge Detection Method", ("Canny Edge Detection", "Adaptive Thresholding", "Sobel Edge Detection", "Otsu's Binarization"), key="thresholding_type", setting_group="Thresholding Edge Detection", setting_name="Method")

if thresholding_type == "Canny Edge Detection":
    # Load Canny thresholds, handling potential lists from JSON
    canny_thresholds = st.session_state.settings.get("Thresholding Edge Detection", {}).get("Canny Thresholds", [50, 150])  # Default as a list
    if isinstance(canny_thresholds, tuple):
        canny_thresholds = list(canny_thresholds)  # Convert to list if needed

    canny_threshold1 = create_slider("Canny Threshold 1", 0, 500, canny_thresholds[0], key="canny_t1", setting_group="Thresholding Edge Detection", setting_name="Canny Thresholds1")
    canny_threshold2 = create_slider("Canny Threshold 2", 0, 500, canny_thresholds[1], key="canny_t2", setting_group="Thresholding Edge Detection", setting_name="Canny Thresholds2")

elif thresholding_type == "Adaptive Thresholding":
    adaptive_method = create_selectbox("Adaptive Method", ("Mean", "Gaussian"), key="adaptive_method", setting_group="Thresholding Edge Detection", setting_name="Adaptive Method")
    adaptive_block_size = create_slider("Adaptive Block Size (odd)", 3, 51, 11, step=2, key="adaptive_block", setting_group="Thresholding Edge Detection", setting_name="Adaptive Block Size")
    adaptive_C = create_slider("Adaptive Constant (C)", -20, 20, 2, key="adaptive_c", setting_group="Thresholding Edge Detection", setting_name="Adaptive Constant")
elif thresholding_type == "Sobel Edge Detection":
    sobel_kernel_size = create_slider("Sobel Kernel Size (odd)", 3, 7, 3, step=2, key="sobel_kernel", setting_group="Thresholding Edge Detection", setting_name="Sobel Kernel Size")

# 6. Edge Enhancement
st.sidebar.header("6. Edge Enhancement")
edge_enhancement_method = create_selectbox("Select Edge Enhancement Method", ("None", "Morphological Gradient", "Laplacian of Gaussian (LoG)"), key="edge_enhancement", setting_group="Edge Enhancement", setting_name="Method")

if edge_enhancement_method == "Laplacian of Gaussian (LoG)":
    log_kernel_size = create_slider("LoG Kernel Size (odd)", 3, 7, 3, step=2, key="log_kernel", setting_group="Edge Enhancement", setting_name="LoG Kernel Size")
    log_sigma = create_slider("LoG Sigma", 0.0, 5.0, 0.5, step=0.1, key="log_sigma", setting_group="Edge Enhancement", setting_name="LoG Sigma")

# 7. Hough Circle Transform
st.sidebar.header("7. Hough Circle Transform (Optional)")
use_hough = create_checkbox("Enable Hough Circle Transform", False, key="use_hough", setting_group="Hough Circle Transform", setting_name="Enabled")

if use_hough:
    hough_dp = create_slider("Hough Transform dp", 1.0, 5.0, 1.2, step=0.1, key="hough_dp", setting_group="Hough Circle Transform", setting_name="dp")
    hough_minDist = create_slider("Hough Transform Min Distance", 10, 200, 20, step=10, key="hough_minDist", setting_group="Hough Circle Transform", setting_name="Min Distance")
    hough_param1 = create_slider("Hough Transform Param1", 50, 300, 100, step=10, key="hough_param1", setting_group="Hough Circle Transform", setting_name="Param1")
    hough_param2 = create_slider("Hough Transform Param2", 10, 200, 30, step=10, key="hough_param2", setting_group="Hough Circle Transform", setting_name="Param2")
    hough_minRadius = create_slider("Hough Transform Min Radius", 0, 100, 10, step=1, key="hough_minRadius", setting_group="Hough Circle Transform", setting_name="Min Radius")
    hough_maxRadius = create_slider("Hough Transform Max Radius", 0, 300, 50, step=1, key="hough_maxRadius", setting_group="Hough Circle Transform", setting_name="Max Radius")

# 8. Contour Filtering
st.sidebar.header("8. Contour Filtering Based on Shape")
apply_shape_filter = create_checkbox("Apply Shape-Based Filtering", False, key="apply_shape", setting_group="Contour Filtering", setting_name="Enabled", sub_setting="Shape-Based")
if apply_shape_filter:
    circularity_min = create_slider("Minimum Circularity", 0.0, 1.0, 0.7, step=0.05, key="circularity_min", setting_group="Contour Filtering", setting_name="Min Circularity", sub_setting="Shape-Based")

st.sidebar.header("9. Additional Contour Filtering")
filter_contours_by_perimeter = create_checkbox("Filter Contours by Perimeter", False, key="filter_perimeter", setting_group="Contour Filtering", setting_name="Enabled", sub_setting="Perimeter Based")
if filter_contours_by_perimeter:
    min_perimeter = create_slider("Minimum Perimeter", 0, 1000, 50, step=10, key="min_perimeter", setting_group="Contour Filtering", setting_name="Min Perimeter", sub_setting="Perimeter Based")

filter_contours_by_area = create_checkbox("Enable Contour Area Filtering", True, key="filter_area", setting_group="Contour Filtering", setting_name="Enabled", sub_setting="Area-Based")

# Load min_area *before* the conditional slider creation
min_area = st.session_state.settings.get("Contour Filtering", {}).get("Area-Based", {}).get("Min Area", 1) # Default to 1

if filter_contours_by_area:
    min_area = create_slider("Minimum Particle Area", 1, 5000, min_area, step=10, key="min_area", setting_group="Contour Filtering", setting_name="Min Area", sub_setting="Area-Based")

# 10. Scale Settings
st.sidebar.header("10. Scale Settings")
use_diameter_distribution = create_checkbox("Use Diameter for Distribution", True, key="use_diameter", setting_group="Scale Parameters", setting_name="Use Diameter Distribution")
pixel_scale = create_number_input("Pixel to Micrometer Scale", value=0.1, step=0.01, format="%.4f", key="pixel_scale", setting_group="Scale Parameters", setting_name="Pixel Scale")


# 11. Histogram Settings
st.sidebar.header("11. Histogram Settings")
num_bins = create_slider("Number of Histogram Bins", 5, 100, 20, key="num_bins", setting_group="Histogram Settings", setting_name="Number of Bins")
use_custom_range = create_checkbox("Use Custom Histogram Range", False, key="use_custom_range", setting_group="Histogram Settings", setting_name="Use Custom Range")
if use_custom_range:
    hist_min = create_number_input("Minimum Range Value", value=0.0, step=1.0, key="hist_min", setting_group="Histogram Settings", setting_name="Min Range")
    hist_max = create_number_input("Maximum Range Value", value=100.0, step=1.0, key="hist_max", setting_group="Histogram Settings", setting_name="Max Range")



# 12. Contour Approximation
st.sidebar.header("13. Contour Approximation")
apply_contour_approx = create_checkbox("Apply Contour Approximation", False, key="apply_approx", setting_group="Contour Approximation", setting_name="Enabled")
approximation_epsilon = create_slider("Approximation Epsilon", 0.0, 1.0, 0.01, step=0.01, key="approx_epsilon", setting_group="Contour Approximation", setting_name="Epsilon") if apply_contour_approx else 0.01

# 13. Save/Load
st.sidebar.header("12. Save Analysis")
save_button = st.sidebar.button("Save Analysis Results")




# --- Main Window Checkboxes ---

show_intermediate = st.checkbox("Show Intermediate Processing Steps", value=True)
show_hough = st.checkbox("Show Hough Circles Overlay", value=False)

# --- Image Processing Pipeline ---

# Step 1: Noise Reduction
if noise_reduction_type == "Gaussian Blur":
    blurred = cv2.GaussianBlur(image_gray, (gaussian_kernel_size, gaussian_kernel_size), gaussian_sigma)
elif noise_reduction_type == "Median Blur":
    blurred = cv2.medianBlur(image_gray, median_kernel_size)
elif noise_reduction_type == "Bilateral Filter":
    blurred = cv2.bilateralFilter(image_gray, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)
else:
    blurred = image_gray.copy()

# Step 2: Contrast Enhancement
if contrast_enhancement == "Histogram Equalization":
    enhanced = cv2.equalizeHist(blurred)
elif contrast_enhancement == "CLAHE":
    clahe = cv2.createCLAHE(clipLimit=clahe_clipLimit, tileGridSize=(tile_grid_size_value, tile_grid_size_value))
    enhanced = clahe.apply(blurred)
else:
    enhanced = blurred.copy()

# Step 3: Morphological Operations
if morph_operation == "Erosion":
    morphed = cv2.erode(enhanced, morph_kernel, iterations=1)
elif morph_operation == "Dilation":
    morphed = cv2.dilate(enhanced, morph_kernel, iterations=1)
elif morph_operation == "Opening":
    morphed = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, morph_kernel)
elif morph_operation == "Closing":
    morphed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, morph_kernel)
elif morph_operation == "Morphological Gradient":
    morphed = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, morph_kernel)
else:
    morphed = enhanced.copy()

# Step 4: Thresholding / Edge Detection
if thresholding_type == "Canny Edge Detection":
     edges = cv2.Canny(morphed, canny_threshold1, canny_threshold2)

elif thresholding_type == "Adaptive Thresholding":
    adaptive_method_cv = cv2.ADAPTIVE_THRESH_MEAN_C if adaptive_method == "Mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    edges = cv2.adaptiveThreshold(morphed, 255, adaptive_method_cv, cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_C)
elif thresholding_type == "Sobel Edge Detection":
    sobelx = cv2.Sobel(morphed, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    sobely = cv2.Sobel(morphed, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
    edges = cv2.magnitude(sobelx, sobely)
    edges = np.uint8(edges)
elif thresholding_type == "Otsu's Binarization":
    _, edges = cv2.threshold(morphed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
else:
    edges = morphed.copy()


# Step 5: Edge Enhancement
if edge_enhancement_method == "Morphological Gradient":
    gradient_kernel_size = st.sidebar.slider("Morphological Gradient Kernel Size (odd)", 3, 21, 3, step=2, key="gradient_kernel_size_display") if edge_enhancement_method == "Morphological Gradient" else 3
    gradient_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gradient_kernel_size, gradient_kernel_size))
    edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, gradient_kernel)

elif edge_enhancement_method == "Laplacian of Gaussian (LoG)":
    log = cv2.GaussianBlur(morphed, (log_kernel_size, log_kernel_size), log_sigma)
    log = cv2.Laplacian(log, cv2.CV_64F, ksize=log_kernel_size)
    log = cv2.convertScaleAbs(log)
    edges = cv2.addWeighted(edges, 0.5, log, 0.5, 0)  # Combine

# Step 6: Hough Circle Transform (Optional)
hough_circles = None
hough_display = None
if use_hough:
    circles = cv2.HoughCircles(enhanced, cv2.HOUGH_GRADIENT, dp=hough_dp, minDist=hough_minDist,
                               param1=hough_param1, param2=hough_param2,
                               minRadius=hough_minRadius, maxRadius=hough_maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        hough_circles = circles[0, :]

# Step 7: Finding Contours
contours_info = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours_info) == 2:
    contours, hierarchy = contours_info
elif len(contours_info) == 3:
    _, contours, hierarchy = contours_info
else:
    st.error("Unexpected return from cv2.findContours")
    st.stop()

# Step 8: Apply contour approximation:
if apply_contour_approx and len(contours) > 0:
    approx_contours = []
    for cnt in contours:
        epsilon = approximation_epsilon * cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx_contours.append(approx)
    contours = approx_contours

# Step 9: Contour Filtering
filtered_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area == 0: continue #Avoid division by zero

    circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0

    valid = True
    if apply_shape_filter and circularity < circularity_min:
        valid = False
    if filter_contours_by_perimeter and perimeter < min_perimeter:
        valid = False
    if filter_contours_by_area and area < min_area:
        valid = False

    if valid:
        filtered_contours.append(cnt)



# Step 10: Calculate Areas
areas = [cv2.contourArea(cnt) for cnt in filtered_contours]
particle_count = len(filtered_contours)

# Step 11: Calculate Diameters (if enabled)
if use_diameter_distribution:
    diameters = [2 * math.sqrt(area / math.pi) * pixel_scale for area in areas]
    size_data = diameters
    size_label = "Particle Diameter (µm)"
else:
    size_data = areas
    size_label = "Particle Area (pixels²)"  # Correct units

# Step 12: Draw Contours
display_image = image_bgr.copy()
cv2.drawContours(display_image, filtered_contours, -1, (0, 0, 255), 2)

if use_hough and hough_circles is not None and show_hough:
    for i in hough_circles:
        cv2.circle(display_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(display_image, (i[0], i[1]), 2, (0, 0, 255), 3)
display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

# --- Visualization ---
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

# Subplot 1: Image with contours
ax[0].imshow(display_image_rgb)
title_particles = f"Particles Detected: {particle_count}"
if use_hough and hough_circles is not None:
    title_particles += f" | Hough Circles Detected: {len(hough_circles)}"
ax[0].set_title(title_particles)
ax[0].axis('off')

# Subplot 2: Histogram
if size_data:
    if use_custom_range:
        ax[1].hist(size_data, bins=num_bins, color='blue', edgecolor='black', range=(hist_min, hist_max))
    else:
        ax[1].hist(size_data, bins=num_bins, color='blue', edgecolor='black')
    ax[1].set_title("Particle Size Distribution")
    ax[1].set_xlabel(size_label)
    ax[1].set_ylabel("Frequency")
else:
    ax[1].text(0.5, 0.5, 'No particles detected.', horizontalalignment='center', verticalalignment='center')
    ax[1].set_title("Particle Size Distribution")
    ax[1].axis('off')

plt.tight_layout()
st.pyplot(fig)


# --- Intermediate Steps ---
if show_intermediate:
    st.markdown("### 📷 Intermediate Processing Steps")
    intermediate_images = [image_gray, blurred, enhanced, morphed, edges]
    captions = ["Original Grayscale", "After Noise Reduction", "After Contrast Enhancement",
                "After Morphological Operations", "After Thresholding/Edge Detection"]
    if edge_enhancement_method != "None":
        intermediate_images.append(edges) #append the enhanced edge
        captions.append("After Edge Enhancement")

    cols = st.columns(3)
    for i, (img, caption) in enumerate(zip(intermediate_images, captions)):
         with cols[i%3]:
            st.image(img, caption=caption, channels="GRAY")

# Display Hough Circle results separately
if use_hough and show_hough and hough_circles is not None:
        st.markdown("### 🔵 Hough Circle Transform Results")
        hough_display = image_bgr.copy()  # Create a copy for Hough circles
        for i in hough_circles:
            cv2.circle(hough_display, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(hough_display, (i[0], i[1]), 2, (0, 0, 255), 3)
        hough_display_rgb = cv2.cvtColor(hough_display, cv2.COLOR_BGR2RGB)
        st.image(hough_display_rgb, caption="Hough Circles Detected")


# --- Display Parameters ---
st.markdown("### 🎛️ Parameters Used")
st.write(f"**Noise Reduction Method:** {noise_reduction_type}")
if noise_reduction_type == "Gaussian Blur":
    st.write(f"  - Gaussian Kernel Size: {gaussian_kernel_size}")
    st.write(f"  - Gaussian Sigma: {gaussian_sigma}")
elif noise_reduction_type == "Median Blur":
    st.write(f"  - Median Kernel Size: {median_kernel_size}")
elif noise_reduction_type == "Bilateral Filter":
    st.write(f"  - Bilateral Diameter: {bilateral_diameter}")
    st.write(f"  - Bilateral Sigma Color: {bilateral_sigma_color}")
    st.write(f"  - Bilateral Sigma Space: {bilateral_sigma_space}")
st.write(f"**Contrast Enhancement:** {contrast_enhancement}")
if contrast_enhancement == "CLAHE":
    st.write(f"  - CLAHE Clip Limit: {clahe_clipLimit}")
    st.write(f"  - CLAHE Tile Grid Size: {tile_grid_size}")
st.write(f"**Morphological Operation:** {morph_operation}")
if morph_operation != "None":
    st.write(f"  - Morphology Kernel Size: {morph_kernel_size}")
st.write(f"**Thresholding/Edge Detection Method:** {thresholding_type}")
if thresholding_type == "Canny Edge Detection":
    st.write(f"  - Canny Thresholds: ({canny_threshold1}, {canny_threshold2})")
elif thresholding_type == "Adaptive Thresholding":
    st.write(f"  - Adaptive Method: {adaptive_method}")
    st.write(f"  - Adaptive Block Size: {adaptive_block_size}")
    st.write(f"  - Adaptive Constant (C): {adaptive_C}")
elif thresholding_type == "Sobel Edge Detection":
    st.write(f"  - Sobel Kernel Size: {sobel_kernel_size}")
st.write(f"**Edge Enhancement Method:** {edge_enhancement_method}")
if edge_enhancement_method == "Laplacian of Gaussian (LoG)":
    st.write(f"  - LoG Kernel Size: {log_kernel_size}")
    st.write(f"  - LoG Sigma: {log_sigma}")
st.write(f"**Hough Circle Transform Enabled:** {'Yes' if use_hough else 'No'}")
if use_hough:
    st.write(f"  - Hough Transform dp: {hough_dp}")
    st.write(f"  - Hough Transform Min Distance: {hough_minDist}")
    st.write(f"  - Hough Transform Param1: {hough_param1}")
    st.write(f"  - Hough Transform Param2: {hough_param2}")
    st.write(f"  - Hough Transform Min Radius: {hough_minRadius}")
    st.write(f"  - Hough Transform Max Radius: {hough_maxRadius}")
st.write(f"**Shape-Based Contour Filtering Applied:** {'Yes' if apply_shape_filter else 'No'}")
if apply_shape_filter:
    st.write(f"  - Minimum Circularity: {circularity_min}")
st.write(f"**Perimeter-Based Contour Filtering Applied:** {'Yes' if filter_contours_by_perimeter else 'No'}")
if filter_contours_by_perimeter:
    st.write(f"  - Minimum Perimeter: {min_perimeter}")
st.write(f"**Area-Based Contour Filtering Applied:** {'Yes' if filter_contours_by_area else 'No'}")
if filter_contours_by_area:
    st.write(f"  - Minimum Particle Area: {min_area}")

st.write(f"**Contour Approximation Applied**:{'Yes' if apply_contour_approx else 'No'}")
if apply_contour_approx:
    st.write(f"- Approximation Epsilon: {approximation_epsilon}")
st.write(f"**Use Diameter Distribution:** {'Yes' if use_diameter_distribution else 'No'}")
if use_diameter_distribution:
    st.write(f"  - Pixel to Micrometer Scale: {pixel_scale}")
st.write(f"**Total Particles Detected:** {particle_count}")



# --- Save Analysis Function ---

def save_analysis(output_dir, size_data, areas, display_image, intermediate_images, captions, hough_display, settings):
    """Saves analysis results, images, and settings."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_folder = os.path.join(output_dir, f"image_analysis_{timestamp}")
    os.makedirs(analysis_folder, exist_ok=True)

    # Save the main image with contours
    cv2.imwrite(os.path.join(analysis_folder, f"original_with_contours_{timestamp}.png"), display_image)

    # Save intermediate images
    if show_intermediate:
        for idx, (img, caption) in enumerate(zip(intermediate_images, captions)):
            filename = f"intermediate_{idx+1}_{caption.replace(' ', '_')}_{timestamp}.png"
            cv2.imwrite(os.path.join(analysis_folder, filename), img)

    # Save Hough circles image
    if use_hough and show_hough and hough_display is not None:
        cv2.imwrite(os.path.join(analysis_folder, f"hough_circles_{timestamp}.png"), hough_display)

    # Save the size distribution plot
    plot_buffer = io.BytesIO()
    fig.savefig(plot_buffer, format="svg", dpi=400)
    plot_buffer.seek(0)
    with open(os.path.join(analysis_folder, f"particle_size_distribution_{timestamp}.svg"), "wb") as f:
        f.write(plot_buffer.read())


    # Save particle sizes
    sorted_sizes = sorted(size_data)
    with open(os.path.join(analysis_folder, f"particle_sizes_{timestamp}.txt"), "w") as f:
        if use_diameter_distribution:
            f.write("Particle Diameters (µm):\n")
        else:
            f.write("Particle Areas (pixels²):\n")
        for size in sorted_sizes:
            f.write(f"{size}\n")


    # Save settings
    with open(os.path.join(analysis_folder, f"settings_{timestamp}.json"), "w") as f:
        json.dump(settings, f, indent=4)

    st.success(f"Analysis saved to {analysis_folder}!")


if save_button:
    # Prepare settings for saving
    settings_to_save = {
        "Noise Reduction": {
           "Method": noise_reduction_type,
           "Gaussian Kernel Size": gaussian_kernel_size if noise_reduction_type == "Gaussian Blur" else None,
           "Gaussian Sigma": gaussian_sigma if noise_reduction_type == "Gaussian Blur" else None,
           "Median Kernel Size": median_kernel_size if noise_reduction_type == "Median Blur" else None,
            "Bilateral Diameter": bilateral_diameter if noise_reduction_type == "Bilateral Filter" else None,
           "Bilateral Sigma Color": bilateral_sigma_color if noise_reduction_type == "Bilateral Filter" else None,
            "Bilateral Sigma Space": bilateral_sigma_space if noise_reduction_type == "Bilateral Filter" else None,
        },
        "Contrast Enhancement": {
            "Method": contrast_enhancement,
            "CLAHE Clip Limit": clahe_clipLimit if contrast_enhancement == "CLAHE" else None,
             "CLAHE Tile Grid Size": tile_grid_size if contrast_enhancement == "CLAHE" else None,
        },
        "Morphological Operations": {
            "Operation": morph_operation,
            "Kernel Size": morph_kernel_size if morph_operation != "None" else None
        },
        "Thresholding Edge Detection":{
            "Method": thresholding_type,
            "Canny Thresholds": (canny_threshold1, canny_threshold2) if thresholding_type == "Canny Edge Detection" else None,
            "Adaptive Method": adaptive_method if thresholding_type == "Adaptive Thresholding" else None,
            "Adaptive Block Size": adaptive_block_size if thresholding_type == "Adaptive Thresholding" else None,
            "Adaptive Constant": adaptive_C if thresholding_type == "Adaptive Thresholding" else None,
             "Sobel Kernel Size": sobel_kernel_size if thresholding_type == "Sobel Edge Detection" else None,
        },
        "Edge Enhancement":{
          "Method": edge_enhancement_method,
          "LoG Kernel Size": log_kernel_size if edge_enhancement_method == "Laplacian of Gaussian (LoG)" else None,
          "LoG Sigma": log_sigma if edge_enhancement_method == "Laplacian of Gaussian (LoG)" else None
        },
        "Hough Circle Transform":{
           "Enabled": use_hough,
           "dp": hough_dp if use_hough else None,
           "Min Distance": hough_minDist if use_hough else None,
           "Param1": hough_param1 if use_hough else None,
           "Param2": hough_param2 if use_hough else None,
           "Min Radius": hough_minRadius if use_hough else None,
           "Max Radius": hough_maxRadius if use_hough else None
        },
        "Contour Filtering":{
             "Shape-Based":{
                "Enabled": apply_shape_filter,
                "Min Circularity": circularity_min if apply_shape_filter else None
             },
             "Perimeter Based":{
               "Enabled": filter_contours_by_perimeter,
               "Min Perimeter": min_perimeter if filter_contours_by_perimeter else None,
            },
            "Area-Based": {
              "Enabled": filter_contours_by_area,
              "Min Area": min_area if filter_contours_by_area else None,
            },
         },
        "Scale Parameters":{
           "Use Diameter Distribution": use_diameter_distribution,
           "Pixel Scale": pixel_scale if use_diameter_distribution else None
         },
         "Histogram Settings": {
             "Number of Bins": num_bins,
              "Use Custom Range": use_custom_range,
              "Min Range": hist_min if use_custom_range else None,
              "Max Range": hist_max if use_custom_range else None
            },
        "Contour Approximation": {
            "Enabled": apply_contour_approx,
            "Epsilon": approximation_epsilon if apply_contour_approx else None
        },
        "Particle Count": particle_count
    }


    # Call save_analysis (make sure all necessary variables are available)
    save_analysis("output_files", size_data, areas, display_image, intermediate_images, captions, hough_display, settings_to_save)
