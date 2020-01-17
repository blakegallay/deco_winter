#!/usr/bin/env python

#------------------------------------------------------------------------#
#  Author: Miles Winter                                                  #
#  Date: 07-14-2017                                                      #
#  Project: DECO                                                         #
#  Desc: zoom in on brightest pixel and classify blobs with CNN          #
#  Note: Need the following installed:                                   #
#        $ pip install --user --upgrade h5py theano keras                #
#        Change keras backend to theano (default is tensorflow)          #
#        Importing keras generates a .json config file                   #
#        $ KERAS_BACKEND=theano python -c "from keras import backend"    #
#        Next, to change "backend": "tensorflow" -> "theano" type        #
#        $ sed -i 's/tensorflow/theano/g' $HOME/.keras/keras.json        #
#        Documentation at https://keras.io/backend/                      #
#------------------------------------------------------------------------#

try:
    import os
    import numpy as np
    import pandas as pd
    import keras
    from keras.models import load_model
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Cropping2D
    from keras.layers import Flatten, Dense, Dropout
    from keras.layers.advanced_activations import LeakyReLU
    from PIL import Image
    from collections import defaultdict
except ValueError:
    print('uhhh')

def build_model(n_classes, training=False):
    """Define model structure"""
    model = Sequential()

    if training:
        model.add(Cropping2D(cropping=18,input_shape=(100, 100, 1)))
        model.add(Conv2D(64, (3, 3), padding='same'))
    else:
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(64, 64, 1)))

    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    return model


def get_event_id(path):
    """Return the event ID for a given image file."""
    directory, filename = os.path.split(path)
    event_id, ext = os.path.splitext(filename)
    return event_id


def get_predicted_label(probs, track_thresh):
    """Returns predicted label. Track if prediction is > track_thresh """
    if probs[-1]>=track_thresh:
        return 'Track'
    else:
        return 'Other'


def get_crop_range(maxX, maxY, size=32):
    """define region of image to crop"""
    return maxX-size, maxX+size, maxY-size, maxY+size


def pass_edge_check(maxX, maxY, img_shape, crop_size=64):
    """checks if image is on the edge of the sensor"""
    x0, x1, y0, y1 = get_crop_range(maxX,maxY,size=crop_size/2)
    checks = np.array([x0>=0,x1<=img_shape[0],y0>=0,y1<=img_shape[1]])
    return checks.all()==True


def convert_image(img,dims=64,channels=1):
    """convert image to grayscale, normalize, and reshape"""
    img = np.array(img,dtype='float32')
    gray_norm_img = np.mean(img/255.,axis=-1)
    return gray_norm_img.reshape(1,dims,dims,channels)


def get_brightest_pixel(img):
    """get brightest image pixel indices"""
    img = np.array(img)
    summed_img = np.sum(img,axis=-1)
    return np.unravel_index(summed_img.argmax(), summed_img.shape)

def run_blob_classifier(arrs, n_classes, model, track_thresh=0.8):
    """Classify blobs with CNN"""
	# Build CNN model and load weights
    '''try:
        model = build_model(n_classes)
        model.load_weights(weights_file)
    except IOError:
        print('Weights could not be found ... check path')
        raise SystemExit'''
    
    class_labels = ['worm', 'spot', 'track', 'noise']
    data = defaultdict(list)
    
    # Loop through all images in the paths list
	
	# MODIFIED TO RUN ON NUMPY ARRAYS INSTEAD OF JPG IMAGE FILES
    #print(arrs)
    for index in arrs:
        arr = arrs[index]
        try:
            #print("HEY LOOK AT ME")
            image = Image.fromarray(arr).convert('RGB')
            #print(image)
            #print("HEY OVER HERE")
        except IOError:
            print('Could not open {}'.format(filename))
            return

        # Find the brightest pixel
        maxY, maxX = get_brightest_pixel(image)

        predicted_label = ''
        probability = 0.
        # Check if blob is near sensor edge
        if pass_edge_check(maxX, maxY, image.size)==True:
            # Crop image around the brightest pixel
            x0, x1, y0, y1 = get_crop_range(maxX,maxY)
            cropped_img = image.crop((x0,y0,x1,y1))
            
            # Convert to grayscale, normalize, and reshape
            gray_image = convert_image(cropped_img)

            # Predict image classification
            probability = model.predict(gray_image, batch_size=1, verbose=0)
            probability = probability.reshape(n_classes,)

            # Convert prediction probability to a single label
            #predicted_label = get_predicted_label(probability, track_thresh)
        else:
            probability = np.array([-1]*n_classes)
            predicted_label = 'Edge'
        
        # Add predicted class label and probabilities from model
        for idx, class_label in enumerate(class_labels):
            data['p_{}'.format(class_label)].append(probability[idx])
        data['x_coord'].append(maxX)
        data['y_coord'].append(maxY)
        data['event_id'].append(index)

    df_data = pd.DataFrame.from_records(data)
	
    del model
    return df_data
