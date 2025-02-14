
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

# function to get thresholds and reference area from config file
def load_config(config_path='C:/Users/mailv/Downloads/config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['low_severity_threshold'], config['medium_severity_threshold'], config['reference_max_area_mm2']

# function for FSI calculation 
def calculate_fsi(flat_area_mm2, reference_max_area_mm2):
    fsi = (flat_area_mm2 / reference_max_area_mm2) * 100 # fsi is expressed as a percentage of max area
    return fsi


# function to calculate impact parameter values
def calculate_impact_parameters(fsi):
    # speed reduction: for every 10% increase in FSI, reduce speed by 5 km/h
    speed_reduction = (fsi / 10) * 5

    # fuel consumption increase: for every 10% increase in FSI, fuel consumption increases by 1%
    fuel_increase = (fsi / 10) * 1

    # failure risk: base risk is 2%, and for every 10% increase in FSI, risk increases by 3%
    risk_of_failure = 2 + (fsi / 10) * 3

    return speed_reduction, fuel_increase, risk_of_failure


# impact analysis function based on severity level
def perform_impact_analysis(severity):
    if severity == "Low Severity":
        return "Minimal impact, no immediate action required. Normal operation."
    elif severity == "Medium Severity":
        return "Potential impact on wheel performance, recommend further inspection and monitoring."
    elif severity == "High Severity":
        return "High risk of operational failure, urgent replacement or repair required."
    else:
        return "No Flat area detected. No impact."

def display_impact_bar_graph(speed_reduction, fuel_increase, risk_of_failure):
    impact_values = [speed_reduction, fuel_increase, risk_of_failure]
    impact_labels = ['Speed Reduction (km/h)', 'Fuel Consumption Increase (%)', 'Risk of Failure (%)']

    # bar graph to display impact parameters
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4), dpi=100)

    ax_bar.barh(impact_labels, impact_values, color=['yellow', 'orange', 'red'])
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel('Percentage / Value')
    ax_bar.set_title('Impact Analysis')

    plt.tight_layout()
    plt.show()

# function to display image with analysis
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

# function for detection of flat area and severity from image
def detect_flat_area_and_severity(image_path, reference_mm=100.0, config_path='C:/Users/mailv/Downloads/config.json'):
    # loading the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # loading values from config file
    low_threshold, medium_threshold, reference_max_area_mm2 = load_config(config_path)

    # grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blurring
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

    # finding contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = img.shape[:2]

    # central region
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

    # iterating through contours to get bounding boxes
    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= 20:
            x, y, w, h = cv2.boundingRect(contour)

            center_x = x + w // 2
            center_y = y + h // 2

            # check whether bounding box is in central region
            if (central_region['x_min'] <= center_x <= central_region['x_max'] and
                    central_region['y_min'] <= center_y <= central_region['y_max']):
                x_min_combined = min(x_min_combined, x)
                y_min_combined = min(y_min_combined, y)
                x_max_combined = max(x_max_combined, x + w)
                y_max_combined = max(y_max_combined, y + h)

    if x_max_combined > x_min_combined and y_max_combined > y_min_combined:
        cv2.rectangle(img, (x_min_combined, y_min_combined), (x_max_combined, y_max_combined), (0, 255, 0), 2)

        combined_width_pixels = x_max_combined - x_min_combined
        combined_height_pixels = y_max_combined - y_min_combined

        # pixel to mm conversion
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

        # calculate FSI
        fsi = calculate_fsi(flat_area_mm2, reference_max_area_mm2)

        # determine severity based on thresholds
        if flat_area_pixels <= low_threshold:
            severity = "Low Severity"
        elif flat_area_pixels <= medium_threshold:
            severity = "Medium Severity"
        else:
            severity = "High Severity"

        # impact analysis based on severity
        impact_analysis = perform_impact_analysis(severity)

        # calculation of impact parameters
        speed_reduction, fuel_increase, risk_of_failure = calculate_impact_parameters(fsi)

        fig = plt.figure(figsize=(24, 10))
        gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 3])

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])

        # heatmap display
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        # original image with bounding box
        ax0.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax0.axis('off')

        # heatmap
        ax1.imshow(heatmap)
        ax1.axis('off')

        # text information
        ax2.axis('off')
        ax2.set_facecolor('white')

        warning_icon = "⚠"
        ax2.text(0.1, 0.95, warning_icon, fontsize=40, ha='left', va='center', color='red')

        ax2.text(0.2, 0.95, "Flat detected", fontsize=11, color='black')

        ax2.text(0.1, 0.88, f"236050 Wheel Number: Wheel 7", fontsize=11, color='black')

        ax2.text(0.1, 0.81, f"Flat Area in Pixels: {flat_area_pixels} px", fontsize=11, color='black')

        ax2.text(0.1, 0.74, f"Flat Area in mm²: {flat_area_mm2:.2f} mm²", fontsize=11, color='black')

        ax2.text(0.1, 0.67, f"Severity Level: {severity}", fontsize=11, color='black')

        ax2.text(0.1, 0.60, f"Impact Analysis: {impact_analysis}", fontsize=11, color='black')

        ax2.text(0.1, 0.53, f"FSI: {fsi:.2f}%", fontsize=11, color='black')

        ax2.text(0.1, 0.46, f"Speed Reduction: {speed_reduction:.2f} km/h", fontsize=11, color='black')

        ax2.text(0.1, 0.39, f"Fuel Increase: {fuel_increase:.2f}%", fontsize=11, color='black')

        ax2.text(0.1, 0.32, f"Risk of Failure: {risk_of_failure:.2f}%", fontsize=11, color='black')

        plt.subplots_adjust(wspace=0.05)
        plt.tight_layout()
        plt.show()

        display_impact_bar_graph(speed_reduction, fuel_increase, risk_of_failure)

    else:
        print("No contours found in the central region.")

# main execution function
def main(image_path, reference_wheel_width_mm=100.0, config_path='C:/Users/mailv/Downloads/config.json'):
    detect_flat_area_and_severity(image_path, reference_wheel_width_mm, config_path)

# example usage
image_path = 'C:/Users/mailv/OneDrive/Pictures/Saved Pictures/test3.png'
image_path = 'C:/Users/mailv/OneDrive/RailWheelDataset/flat/imgf40.jpg'

reference_wheel_width_mm = 100.0  # reference width for pixel level algorithm
config_path = 'C:/Users/mailv/Downloads/config.json'

main(image_path, reference_wheel_width_mm, config_path)
