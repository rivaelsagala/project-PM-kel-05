import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG16

# Parameter
img_height, img_width = 150, 150
batch_size = 32

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, 
                                    rotation_range=20, 
                                    width_shift_range=0.2, 
                                    height_shift_range=0.2, 
                                    shear_range=0.2, 
                                    zoom_range=0.2, 
                                    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Model CNN dengan Transfer Learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze base model

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 kelas
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(train_generator, epochs=10)

# Menyimpan model
model.save('fish_classification_model.h5')
