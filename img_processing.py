
import cv2
import numpy as np
import matplotlib.pyplot as plt

# func to detect bounding box and severity
def detect_flat_area_and_severity(image_path, reference_mm=100.0):

    # loading img
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

    # contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # image dimensions
    height, width = img.shape[:2]

    # getting central region
    central_region = {
        'x_min': width * 0.175,
        'x_max': width * 0.825,
        'y_min': height * 0.175,
        'y_max': height * 0.825
    }

    x_min_combined = width
    y_min_combined = height
    x_max_combined = 0
    y_max_combined = 0

    # central bounding box
    for contour in contours:
        # contour area
        area = cv2.contourArea(contour)

        if area >= 20:
            x, y, w, h = cv2.boundingRect(contour)

            center_x = x + w // 2
            center_y = y + h // 2

            # checking if the bounding box center is in central region
            if (central_region['x_min'] <= center_x <= central_region['x_max'] and
                    central_region['y_min'] <= center_y <= central_region['y_max']):
                x_min_combined = min(x_min_combined, x)
                y_min_combined = min(y_min_combined, y)
                x_max_combined = max(x_max_combined, x + w)
                y_max_combined = max(y_max_combined, y + h)

    if x_max_combined > x_min_combined and y_max_combined > y_min_combined:
        cv2.rectangle(img, (x_min_combined, y_min_combined), (x_max_combined, y_max_combined), (0, 255, 0), 2)

        # dimensions of bounding box
        combined_width_pixels = x_max_combined - x_min_combined
        combined_height_pixels = y_max_combined - y_min_combined

        # pixel to mm conversion
        # reference is in mm
        pixels_per_mm = width / reference_mm
        combined_width_mm = combined_width_pixels / pixels_per_mm
        combined_height_mm = combined_height_pixels / pixels_per_mm

        # flat area in pixels
        flat_area_pixels = np.sum(edges[y_min_combined:y_max_combined, x_min_combined:x_max_combined] == 255)

        # converting flat area to mm^2
        if combined_width_pixels > 0:
            flat_area_mm2 = (flat_area_pixels * reference_mm ** 2) / (combined_width_pixels * combined_height_pixels)
        else:
            flat_area_mm2 = 0

        # severity calculation
        severity = calculate_severity(flat_area_pixels)

        # impact analysis based on severity
        impact_analysis = perform_impact_analysis(severity)

        display_image_with_details(img, flat_area_mm2, severity, impact_analysis, combined_width_pixels,
                                   combined_height_pixels, combined_width_mm, combined_height_mm, flat_area_pixels,
                                   flat_area_mm2)

    else:
        print("No contours found in the central region.")

# severity calc function
def calculate_severity(flat_area_pixels):
    if flat_area_pixels < 200:
        return "Low Severity"
    elif flat_area_pixels < 700:
        return "Medium Severity"
    else:
        return "High Severity"


# impact analysis function
def perform_impact_analysis(severity):
    if severity == "Low Severity":
        return "Minimal impact, no immediate action required. Normal operation."
    elif severity == "Medium Severity":
        return "Potential impact on wheel performance, recommend further inspection and monitoring."
    elif severity == "High Severity":
        return "High risk of operational failure, urgent replacement or repair required."
    else:
        return "No Flat area detected. No impact."


# display
def display_image_with_details(image, flat_area_mm2, severity, impact_analysis, combined_width_pixels,
                               combined_height_pixels, combined_width_mm, combined_height_mm, flat_area_pixels,
                               flat_area_mm2_cvt):

    details = f"Combined Bounding Box - Width: {combined_width_pixels} px, {combined_width_mm:.2f} mm\n"
    details += f"Combined Bounding Box - Height: {combined_height_pixels} px, {combined_height_mm:.2f} mm\n"
    details += f"Flat Area in Pixels: {flat_area_pixels}\n"
    details += f"Flat Area in mm²: {flat_area_mm2_cvt:.2f} mm²\n"
    details += f"Severity Level: {severity}\n"
    details += f"Impact Analysis: {impact_analysis}"

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].axis('off')

    ax[1].axis('off')
    ax[1].text(0.1, 0.9, details, fontsize=12, wrap=True)

    plt.tight_layout()
    plt.show()

#image_path = 'C:/Users/mailv/OneDrive/Pictures/Saved Pictures/test7.png'
image_path = 'C:/Users/mailv/OneDrive/Pictures/Saved Pictures/test3.png'
reference_wheel_width_mm = 100.0

detect_flat_area_and_severity(image_path, reference_wheel_width_mm)
