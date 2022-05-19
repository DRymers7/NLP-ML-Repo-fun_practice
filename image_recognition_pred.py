from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing import image
from pathlib import Path
import numpy as np

#Class labels
class_labels = {
    0: 'Plane',
    1: 'Car',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Boat',
    9: 'Truck'
}

#Load json file that contain's model's structure
f = Path('model_structure.json')
model_structure = f.read_text()

#Recreate keras model from json data
model = model_from_json(model_structure)

#Load trained weights
model.load_weights('model_weights.h5')

#Load an image file to test, resizing to 32x32
img = image.load_img('cat.png', target_size=(32,32))

#Convert the image to a numpy array
image_to_test = image.img_to_array(img) / 255

#Add fourth dimension to the image
list_of_images = np.expand_dims(image_to_test, axis=0)

#Make a prediction
results = model.predict(list_of_images)

#Only testing one image so only need to check first result
single_result = results[0]

#Get name of most likely class index
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]

#Name of most likely class
class_label = class_labels[most_likely_class_index]

#Print result
print('This image is a {} - likelihood: {}'.format(class_label, class_likelihood))