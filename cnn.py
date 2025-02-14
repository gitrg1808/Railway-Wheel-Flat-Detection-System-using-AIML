
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, classification_report
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# resize images function
def resize_images(source_folder, target_size=(224, 224)):
    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(source_folder, filename)
            with Image.open(image_path) as img:
                img_resized = img.resize(target_size)
                img_resized.save(image_path)

# resizing dataset images
resize_images('C:/Users/mailv/OneDrive/RailWheelDataset/flat')
resize_images('C:/Users/mailv/OneDrive/RailWheelDataset/noflat')

# load and preprocess images function
def pixelsize(source_folder):
    image_arrays = []
    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(source_folder, filename)
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0  # image normalization
            image_arrays.append(img_array)
    return image_arrays

# loading and preprocessing
flataray = pixelsize('C:/Users/mailv/OneDrive/RailWheelDataset/flat')
noflatarray = pixelsize('C:/Users/mailv/OneDrive/RailWheelDataset/noflat')

train_datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# CNN model
cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
cnn.add(MaxPooling2D(2, 2))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(2, 2))
cnn.add(Conv2D(128, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(2, 2))

cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dropout(0.5)) # dropout layer to prevent overfitting
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))

# sigmoid activation output layer for binary classification
cnn.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))

cnn.summary()

# binary crossentropy loss function
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# training and validation generators
train_generator = train_datagen.flow_from_directory(
    'C:/Users/mailv/OneDrive/RailWheelDataset',
    target_size=(224, 224),
    batch_size=10,
    classes=['flat', 'noflat'],
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'C:/Users/mailv/OneDrive/RailWheelDataset',
    target_size=(224, 224),
    batch_size=10,
    classes=['flat', 'noflat'],
    class_mode='binary',
    subset='validation'
)

# early stop to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# weight training
class_weight = {0: 1.0, 1: 5.0}

# model training
history = cnn.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping],
    #class_weight=class_weight
)

# plotting training history
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

# save model
cnn.save('railcnn2.h5')

# loading trained model
model = tf.keras.models.load_model('railcnn2.h5')
model.summary()

# confusion matrix and performance report
y_true = validation_generator.classes
y_pred = (cnn.predict(validation_generator) > 0.3).astype("int32")

# confusion matrix and classification report
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

# function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(np.arange(len(cm)))
    ax.set_yticks(np.arange(len(cm)))
    ax.set_xticklabels(np.arange(1, len(cm) + 1))
    ax.set_yticklabels(np.arange(1, len(cm) + 1))

    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f'{value}', ha='center', va='center', color='red')

    root = tk.Tk()
    root.title('Confusion Matrix')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    root.mainloop()

# display confusion matrix
plot_confusion_matrix(y_true, y_pred)
