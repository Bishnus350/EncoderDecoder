# Giving the path of the directory
pwd = r'C:/Users/bishn/Desktop/Astro_Python/'

# importing necessary modules
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
import cv2

# loading and reading images
image = cv2.imread (pwd + 'Bp-RpNGC6366.png' )
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224,224))
image = np.array (image, dtype = np.float32)
image = image/255
print (image)
cv2.imshow('Image', image)
height, width, channel = image.shape
print (f" Image width:{width}, Image height: {height}, Image channel: {channel}")
image = np.expand_dims(image, axis=0)  # Remove the extra dimension
input_shape = (224, 224, 3)

## Architect of encoding and decoding model
def encoder_decoder_model():

    """
    Used to build Convolutional Autoencoder model architecture to get compressed image data which is easier to process.
    Returns:
    Auto encoder model
    """
    #Encoder 
    model = Sequential(name='Convolutional_AutoEncoder_Model')
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(224, 224, 3),padding='same', name='Encoding_Conv2D_1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='Encoding_MaxPooling2D_1'))
    model.add(Conv2D(128, kernel_size=(3, 3),strides=1,kernel_regularizer = tf.keras.regularizers.L2(0.001),activation='relu',padding='same', name='Encoding_Conv2D_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='Encoding_MaxPooling2D_2'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='Encoding_Conv2D_5'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

    #Decoder
    model.add(Conv2D(256, kernel_size=(3, 3), kernel_regularizer = tf.keras.regularizers.L2(0.001), activation='relu', padding='same',name='Decoding_Conv2D_3'))
    model.add(UpSampling2D((2, 2),name='Decoding_Upsamping2D_3'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer = tf.keras.regularizers.L2(0.001), padding='same',name='Decoding_Conv2D_4'))
    model.add(UpSampling2D((2, 2),name='Decoding_Upsamping2D_4'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer = tf.keras.regularizers.L2(0.001), padding='same',name='Decoding_Conv2D_5'))
    model.add(UpSampling2D((2, 2),name='Decoding_Upsamping2D_5'))
    model.add(Conv2D(3, kernel_size=(3, 3), padding='same',activation='sigmoid',name='Decoding_Output'))
    return model

model = encoder_decoder_model ()
model.summary()
print ('\n')
#tf.keras.utils.plot_model(model, to_file= pwd + 'model.png')
optimizer = Adam(learning_rate=0.001) 
model = encoder_decoder_model()
model.compile(optimizer=optimizer, loss='mse') 
model.fit(x=image, y=image,
          batch_size=30, epochs= 50)
#image_enc = np.expand_dims(image,axis=1)
encoder_image = model.predict(image)
decoder_image = encoder_image
#encoder_image = np.squeeze(encoder_image, axis=2)
## Visualize encoded and decoded image ##
plt.figure(figsize=(10,8))
plt.subplot(1,3,1)
plt.imshow (image[0])
plt.title("Original Image")
plt.axis('off')
# Encoded image
plt.subplot(1,3,2)
plt.imshow (encoder_image[0])
plt.title("Encoded Image")
plt.axis('off')
#Decoded image
plt.subplot (1, 3, 3)
plt.imshow(decoder_image[0])
plt.title('Decoded image')
plt.axis ('off')
plt.show()
plt.savefig('encoderdecoder.png')

 

