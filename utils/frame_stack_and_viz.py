import os
import re
from PIL import Image, ImageDraw, ImageFont

# The directory containing the images
directory = 'final_figures/'

# Regex pattern to match the files
pattern = re.compile(r'(\w+)_\d+_frame_select\.png')

# Collecting the right files with their labels
image_files = []
labels = []

# Search through the directory for files that match the pattern
for filename in sorted(os.listdir(directory)):
    if pattern.match(filename):
        image_files.append(os.path.join(directory, filename))
        # The label is the part before the first underscore
        label = filename.split('_')[0]
        labels.append(label)

# Load images to get their dimensions and calculate total height
images = []
for file in image_files:
    img = Image.open(file)
    images.append(img)

# Assuming all images are the same width
total_width = max(img.width for img in images)
label_height = 30  # Adjust label height as needed

# Calculate total height needed for the grid image (including labels)
total_height = sum(img.height for img in images) + label_height * len(images)

# Create a new blank image for the grid
grid_img = Image.new('RGB', (total_width, total_height), color='white')
draw = ImageDraw.Draw(grid_img)

# Set font for the labels
try:
    font = ImageFont.truetype("arial.ttf", label_height - 10)  # Adjust font size as needed
except IOError:
    font = ImageFont.load_default()

# Initial y_offset
y_offset = 50

# Place each image and its label onto the grid
for img, label in zip(images, labels):
    # Calculate text size
    # text_width, text_height = draw.textsize(label, font=font)
    
    # Draw label above the image
    # text_x = (total_width - text_width) // 2  # Center the text
    # draw.text((text_x, y_offset), label, fill="black", font=font)
    
    # Update y_offset for the image, leaving space for the label
    # y_offset += text_height
    
    # Paste the image onto the grid
    grid_img.paste(img, (0, y_offset))
    
    # Update y_offset for the next label, leaving space for the image
    y_offset += img.height if label != labels[-1] else 0


    # Close the image file
    img.close()

# Save the final grid image
output_path = 'property_stacked.png'
grid_img.save(output_path)
print(f"Image saved to {output_path}")
