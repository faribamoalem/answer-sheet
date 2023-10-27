
import cv2

# Load the image
image = cv2.imread('.\example/answer-sheet.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to convert to binary image
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform morphological operations to clean up the image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours based on area in ascending order
contours = sorted(contours, key=cv2.contourArea)

# Get the 50 smallest contours and sort them
smallest_contours = contours[:50]
smallest_contours = sorted(smallest_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

# Initialize left and right column contours
left_contours = []
right_contours = []

# Identify left and right column contours based on x-coordinate
for contour in smallest_contours:
    x, _, _, _ = cv2.boundingRect(contour)
    if x < image.shape[1] // 2:  # Assume left column if x < half of image width
        left_contours.append(contour)
    else:
        right_contours.append(contour)

# Sort left column contours based on y-coordinate in ascending order
left_contours = sorted(left_contours, key=lambda c: cv2.boundingRect(c)[1])

# Sort right column contours based on y-coordinate in ascending order
right_contours = sorted(right_contours, key=lambda c: cv2.boundingRect(c)[1])

# Combine left and right column contours
sorted_contours = left_contours + right_contours

# Loop over the smallest contours and draw rectangles around them
for i, contour in enumerate(sorted_contours):
    x, y, w, h = cv2.boundingRect(contour)
    enlarged_w = w * 15
    enlarged_h = h * 3
    x += 5 * w
    y += h - enlarged_h
    cv2.rectangle(image, (x, y), (x + enlarged_w, y + enlarged_h), (0, 255, 0), 2)

    # Extract the region of interest within the rectangle
    roi = binary[y:y + enlarged_h, x:x + enlarged_w]

    # Divide the region into 6 equal parts
    section_width = enlarged_w // 6

    # Initialize the detected filled option as none
    filled_option = ""

    # Loop over the sections and check for filled options
    for section in range(1, 6):
        section_roi = roi[:, section_width * section:section_width * (section + 1)]

        # Check if the section contains any contours
        section_contours, _ = cv2.findContours(section_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If contours are present, assume the option is filled
        if section_contours:
            filled_option = chr(ord('A') + section)
            for c in section_contours:
                area = cv2.contourArea(c)
                if area >= 40:  # Ignore areas with an area less than 40
                    print(f"Question {i + 1} area: {area}")
            break  # Break the loop if a filled option is found

    # Print the question number and filled option
    question_number = i + 1
    if filled_option:
        print(f"Question {question_number}: {filled_option}")
    else:
        print(f"Question {question_number}: EMPTY")

# Resize the image for display
resized_image = cv2.resize(image, (800, 900))

# Display the resized image with single-answer areas highlighted
cv2.imshow('Single-Answer Detection', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()