import sys
import tkinter

import cv2
import numpy as np
import re
import tensorflow as tf
from tkinter import filedialog



root = tkinter.Tk()  # esto se hace solo para eliminar la ventanita de Tkinter
root.withdraw()  # ahora se cierra
file_path = filedialog.askopenfilename()  # abre el explorador de archivos y guarda la seleccion en la variable!

# Ahora para guardar el directorio donde se encontraba el archivo seleccionado:
match = re.search(r'/.*\..+', file_path)  # matches name of file
file_position = file_path.find(match.group())  # defines position of filename in file path

dir_path = file_path[0: file_position + 1]  # extracts the saving path.

print(file_path)
# Primero le pasamos la ruta de la imagen
image_path = file_path
filename = dir_path +'/' +image_path
image_size=128
num_channels=3
images = []
# Reading the image using OpenCV
image = cv2.imread(filename)
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0)
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)

## Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('Venenosas-NoVen-model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()


# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 2))

### Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)

solucion = "Venenosas: " + str(result[0][0]) + " No Venenosas: " + str(result[0][1])
print(solucion)

