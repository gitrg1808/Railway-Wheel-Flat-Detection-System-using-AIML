from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.preprocessing import image

# load model
model = tf.keras.models.load_model('railcnn2.h5')
model.summary()

# function to make predictions on test images
def predict_image(image_path):
    test_image = image.load_img(image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0  # Normalize the image
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image)

    if result[0] > 0.3:
        print(f"The image {image_path} classified as flat")
    else:
        print(f"The image {image_path} classified as no flat")

# predict test images
predict_image("C:/Users/mailv/OneDrive/Pictures/Saved Pictures/test3.png")
predict_image("C:/Users/mailv/OneDrive/Pictures/Screenshots/Screenshot 2024-12-01 192742.png")
