import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  
from fine_tune_model import original_resnet152, original_resnet50
from keras.preprocessing.image import ImageDataGenerator
from load_data import load_full_data, load_task3_training_labels, load_training_data
import numpy as np
import tensorflow as tf
from keras import backend as K
from callback import config_cls_callbacks
from misc_utils.eval_utils import compute_class_weights
from misc_utils.print_utils import log_variable, Tee
from misc_utils.filename_utils import get_log_filename
import sys
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    img_rows, img_cols = 224, 224
    channels = 3
    num_classes = 7 
    batch_size = 16
    epochs = 250
    
    print("Loading Images...")
    x = load_full_data()
    y = load_task3_training_labels()
    #Replace with your own dataset.
    
    print("Splitting Data...")
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=42)
    
    print("x_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    print("x_valid shape: ", x_valid.shape)
    print("y_valid shape: ", y_valid.shape)
    """
    (x_train, y_train), (x_valid, y_valid), _ = load_training_data(task_idx=3,
                                                                   output_size=224,
                                                                   num_partitions=1,
                                                                    test_split = 0.8)"""

    class_wt_type = 'balanced'
    run_name = 'OriginalResNet152'

    num_classes = y_train.shape[1]
    
    class_weights = compute_class_weights(y, wt_type=class_wt_type)
    
    print(class_weights)
    
    callbacks = config_cls_callbacks(run_name)

    n_samples_train = x_train.shape[0]
    n_samples_valid = x_valid.shape[0]

    sys.stdout.flush()
    """#Size of training batch
    batch_train = x_train.shape[0]
    #Size of testing batch
    batch_test = x_test.shape[0]

    y_train = y_train.flatten()
    y_test  = y_test.flatten()

    #Uncomment this to reshape images to a size acceptable by the CNN

    x_train = x_train.reshape(batch_size, channels, img_rows, img_cols)
    x_test = x_test.reshape(batch_size, channels, img_rows, img_cols)


    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #-----------------------------------------------------------------------------
    #Initialized tensorflow session.
    '''
    sess = tf.InteractiveSession()
    K.set_session(sess)
    '''"""
    #-----------------------------------------------------------------------------

    #10 classes because dataset is Cifar-10
    model = original_resnet152(7)
    #model = original_resnet50(7)

    #-----------------------------------------------------------------------------
    #fit_generator is a method that fits the model on data that is 
    #processed in batches before being sent to the model 
    '''
    #create one-hot encodings of the true image classes
    y_train_one_hot = tf.one_hot(y_train, num_classes).eval()
    data_train_gen = data_generator(sess, x_train, y_train_one_hot)
    # Fit model on data using fit_generator
    model.fit_generator(data_train_gen(), epochs=batch_train/batch_size, verbose=1)
    '''
    #-------------------------------------------------------------------------------
    horizontal_flip = True
    vertical_flip = True
    rotation_angle = 180
    width_shift_range = 0.1
    height_shift_range = 0.1

    datagen = ImageDataGenerator(rotation_range=rotation_angle,
                                     horizontal_flip=horizontal_flip,
                                     vertical_flip=vertical_flip,
                                     width_shift_range=width_shift_range,
                                     height_shift_range=height_shift_range)
    
    model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // 16 * 2,
                            epochs=epochs,
                            initial_epoch=0,
                            verbose=1,
                            validation_data=(x_valid, y_valid),
                            callbacks=callbacks)
    #Fit model on the training and testing datasets
    '''model.fit(x=x_train,
              y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_valid, y_valid),
              class_weight=class_weights,
              shuffle=True,
              callbacks=callbacks
              )'''

    # Make predictions
    #predictions_valid = model.predict(x_test, batch_size=batch_size, verbose=1)
