import os
from PIL import Image

# Define the root directory of the tree structure
root_dir = './datasets'  # Replace with the correct path

# Function to check image size and delete if conditions are met
def check_and_delete_images(folder_path):
    image_details = []
    for root, dirs, files in os.walk(folder_path):
        print(root)
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                try:
                    with Image.open(os.path.join(root, file)) as img:
                        width, height = img.size
                        if min(width, height) < 100:
                            print(f"Image {file} is smaller than 100 pixels in {root}")
                            # answer = input("%s, %s\nY/N you want remove?"%(width*height, file))
                            answer = True
                            if answer:
                                os.remove(os.path.join(root, file))  # Uncomment to delete the file
                                print("remove %s"%file)
                            image_details.append((root, file, "Deleted"))
                        else:
                            image_details.append((root, file, "Kept"))
                except Exception as e:
                    print(f"Error opening image {file}: {e}")

    # Counting images in each folder
    folder_count = {folder: 0 for folder, _, _ in image_details}
    for folder, _, status in image_details:
        if status == "Kept":
            folder_count[folder] += 1

    for folder, count in folder_count.items():
        print(f"There are {count} images in the folder {folder}")

# Run the function starting from the root directory
check_and_delete_images(root_dir)