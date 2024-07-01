import keras
import ssl
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
ssl._create_default_https_context = ssl._create_unverified_context
test_img = ImageDataGenerator(preprocessing_function=preprocess_input)
num_classes = 2
image_resize = 224
batch_size_test = 100
test_generator = test_img.flow_from_directory('concrete_data_week4/test', target_size=(image_resize, image_resize), shuffle=False)
vgg16M = keras.saving.load_model('classifier_vgg_model.h5', custom_objects=None, compile=True, safe_mode=True)
resnetM = keras.saving.load_model('classifier_resnet_model.h5', custom_objects=None, compile=True, safe_mode=True)
vEval = vgg16M.evaluate(test_generator)
rEval = resnetM.evaluate(test_generator)
print(vEval)
print(rEval)
vPred = vgg16M.predict(test_generator)
print(vPred[0:4])