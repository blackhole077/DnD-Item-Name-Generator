import tensorflow as tf

from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (LSTM, Activation, BatchNormalization,
                                     Convolution1D, Convolution2D, Dense,
                                     Dropout, Embedding, Flatten, Input,
                                     MaxPooling1D, MaxPooling2D, Reshape,
                                     ThresholdedReLU, TimeDistributed)
from tensorflow.keras.models import Model, Sequential, load_model

import dataset_utils as _dutils


def build_character_cnn(model_hyperparameters=None, verbose=None):
    """
        Create a language model based on the Crepe model in Zhang et. al.

        Creates a model that uses only Convolutional Layers (max pooling in between),
        and dense layers for prediction tasks. For this model, the final output layer
        will predict the next character in the sequence.

        Parameters
        ----------
        model_hyperparameters : dict
            A dictionary containing values necessary to build the model.
        verbose : bool
            A flag to print additional information.

        Returns
        -------
        model : Sequential
            A Keras Sequential that represents the model.
        Notes
        -----
        See https://gab41.lab41.org/deep-learning-sentiment-one-character-at-a-t-i-m-e-6cd96e4f780d for more information.
        See https://github.com/mhjabreel/CharCnn_Keras for the Keras code this is based on
        See https://github.com/zhangxiangxiao/Crepe for the original (Torch) Crepe code
    """
    if model_hyperparameters is None:
        model_hyperparameters = _dutils.load_dictionary('model_hyperparameters.json')
    '''
        Load hyperparameter-specific values from JSON file.
    '''
    #The size of the characater vocabulary
    vocabulary_size = model_hyperparameters.get("vocabulary_size")
    #The max length of the text. Set as 1014 in the original.
    text_length = model_hyperparameters.get("text_length")
    #Number of filters for each convolutional layer
    num_filters = model_hyperparameters.get("num_filters")
    #The threshold for the ReLU activation layers
    threshold = model_hyperparameters.get("relu_threshold")
    #Dropout probability for Dropout layers
    dropout_p = model_hyperparameters.get("dropout_percent")
    #Embedding output dimension. Implementation sets it equal to vocabulary_size
    embed_dim = model_hyperparameters.get("embedding_dimension")
    '''
        Values below specify the architecture.
        These aren't stored in the JSON file due to
        architectutre constraints with layers and
        kernel sizes.
    '''
    #The number of units for each dense layer minus output layer
    fully_connected_layers = [128,64]
    '''
        conv_layers is a list of pairs.
        First component refers to kernel size.
        Second component refers to the size of
        the MaxPooling1D layer (-1 indicates said layer is not present).
    '''
    conv_layers = [[7, 3], [3,-1], [3,-1], [3,-1], [3, 3]]
    #Input layer
    inputs = Input(shape=(text_length,), name='sent_input', dtype='int32')
    #Embedding layers
    x = Embedding(vocabulary_size + 1, embed_dim, input_length=text_length, mask_zero=True)(inputs)
    #Convolution layers
    '''
        First Conv1D layer + MaxPooling is separate in case
        changes are made upstream. Also it was used to test out
        TimeDistributed functionality.
    '''
    x = (Convolution1D(num_filters, 7))(x)
    x = (MaxPooling1D(3))(x)
    for cl in conv_layers:
        x = (Convolution1D(num_filters, cl[0]))(x)
        x = ThresholdedReLU(threshold)(x)
        if cl[1] != -1:
            x = (MaxPooling1D(cl[1]))(x)

    x = Flatten()(x)
    # #Fully connected layers
    for fl in fully_connected_layers:
        '''
            Original architecture did not use L2 regularization.
            However, empirical results show that, for my dataset
            it works well in handling overfitting.
        '''
        x = Dense(fl, kernel_regularizer=regularizers.l2(0.0001))(x)
        x = ThresholdedReLU(threshold)(x)
        '''
            Original architecture had dropout at 50%.
            This seemed to be too high for my dataset, and
            it resulted in underfitting.
        '''
        x = Dropout(dropout_p)(x)
    # #Output layer
    predictions = Dense(vocabulary_size, activation='softmax')(x)
    # Build and compile model
    model = Model(inputs=inputs, outputs=predictions)    
    if verbose:
        model.summary()
    return model

def loss(labels, logits):
    return tf.keras.losses.categorical_crossentropy(labels, logits)

def prepare_model(model_hyperparameters=None, load_weights=False, weight_directory=None, verbose=False):
    """
        Create a model to generate item names.

        Given a dictionary of hyperparameter values, this function builds a
        model according to the build_model function, then compiles it with
        Stochastic Gradient Descent and returns the model for training.

        Parameters
        ----------
        model_hyperparameters : dict
            A dictionary containing values necessary to build the model.
            Contains information on 
    """
    if model_hyperparameters is None:
        raise ValueError("Hyperparameters must be given to prepare a model.")
    model_to_train = build_character_cnn(model_hyperparameters, verbose=verbose)
    optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model_to_train.compile(optimizer=optimizer, loss=loss)
    if verbose:
        model_to_train.summary()
    if load_weights:
        if weight_directory is not None:
            model_to_train.load_weights(tf.train.latest_checkpoint(weight_directory))
        else:
            print("Error with directory provided to load weights from. \
                   Continuing with no weights loaded.")
    return model_to_train
