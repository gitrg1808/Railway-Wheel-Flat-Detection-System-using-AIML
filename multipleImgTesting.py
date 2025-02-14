import shutil
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('railcnn2.h5')
model.summary()

# func to classify flat images and move to folder
def classify_and_move_images(source_folder, flat_folder):
    # output directory
    os.makedirs(flat_folder, exist_ok=True)

    # looping through all images
    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(source_folder, filename)

            # predict whether image is flat or nonflat
            result = predict_image(image_path)  

            # move flat images to folder
            if result == 'flat':
                shutil.move(image_path, os.path.join(flat_folder, filename))
                print(f"Moved {filename} to flat images folder.")
            else:
                print(f"Image {filename} is classified as non-flat. Skipping.")


# prediction function to classify image as flat or nonflat
def predict_image(image_path):
    test_image = image.load_img(image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0  
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image)

    # returns flat or no flat based on prediction threshold
    if result[0] > 0.3:
        return 'flat'  
    else:
        return 'no flat'  


# calling func to classify and move images
source_folder = "C:/Users/mailv/OneDrive/Pictures/SourcePics" # source folder containing images to be classified
flat_folder = "C:/Users/mailv/OneDrive/Pictures/FlatResults"  

classify_and_move_images(source_folder, flat_folder)
