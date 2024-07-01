import keras
import ssl
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
ssl._create_default_https_context = ssl._create_unverified_context
num_classes = 2
image_resize = 224
batch_size_training = 100
batch_size_validation = 100
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = data_generator.flow_from_directory('concrete_data_week4/train', target_size=(image_resize, image_resize), batch_size=batch_size_training, class_mode='categorical')
train_generator

