import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16

model = vgg16.VGG16()

img = image.load_img('bay.jpg', target_size=(224,224)) #image needs to match size of NN input nodes

#Convert to numpy array
x = image.img_to_array(img)

#Add fourth dimension (Keras expects list of images)
x = np.expand_dims(x, axis=0)

#Normalize input image pixel values to range used when training NN
x = vgg16.preprocess_input(x)

#Run the image through NN to make prediction
predictions = model.predict(x)

#Look up naems of predicted classes
predicted_classes = vgg16.decode_predictions(predictions, top=9)

print('Top predictions for this image: ')

for imagenet_id, name, likelihood in predicted_classes[0]:
    print('Prediction: {} - {}'.format(name, likelihood))