import argparse
import datetime
import os
import random
from random import random, seed

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model

import dataset_utils as _dutils
import model as _model

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, choices=['train', 'test'], help='Do you wish to train the model or test it.')
parser.add_argument('--num_samples', type=int, default=70000, help='How many samples do you want to take out of the full dataset?')
parser.add_argument('--num_generate', type=int, default=5, help='How many names do you wish to generate (test only)?')
parser.add_argument('--dataroot_path', type=str, default='item_names_lower.txt', help='The file path (including the file itself) of the raw data.')
parser.add_argument('--weight_directory', type=str, default='training_checkpoints', help='Where are the best weights going to be loaded/stored.')
args = parser.parse_args()


def generate_text(trained_model,
                  index_to_character_dict=None,
                  generator_hyperparameters=None,
                  generation_type=None):
    '''
        Generate an item name using a trained model.

        Given a trained model, the dictionary of indices to characters,
        the maximum length allowed for an item name, the number of
        characters present in the vocabulary, and a "text seed", generate
        a name character-by-character. The function ends if either the <END>
        or <PAD> tokens are generated, or the max_length is reached.

        This function also employs a simulated annealing approach, combined
        with an epsilon-greedy style of choosing between the "best" (i.e., most likely)
        character, versus a character at random (uniform distribution), with epsilon
        being annealed over time (can be tuned via hyperparameter 'temperature').

        Parameters
        ----------
        trained_model : tf.keras.Model
            A Tensorflow (Keras) Model that is already trained.
        index_to_character_dict: dict
            A dictionary mapping indices (keys) to characters (values).
            Example: include index_to_character_dict[3] -> '#'
        vocab_size : int
            The number of characters present in the vocabulary.
        generator_hyperparameters : dict
            A dictionary mapping various configuration values.

        Returns
        -------
        final_name : str
            The item name, stripped of all tokens (<START>,<END>,<UNK>,<PAD>).

        Notes
        -----
        If the model's input dimensions change, then this function will have to
        prepare the text_seed to match that input.
    '''
    if generator_hyperparameters is None:
        text_seed = '*'
        offset = len(text_seed)
        max_length = 150
        # Converting our start string to numbers (vectorizing)
        generated_name = list(text_seed + '#' * (150 - offset))
        #Controls the scaling of the simulated annealing.
        temperature = 1.0
        epsilon = 0.3
        probability = (1 - epsilon)
        seed(1)
    else:
        text_seed = generator_hyperparameters.get('text_seed', '*')
        offset = len(text_seed)
        max_length = generator_hyperparameters.get('max_length')
        generated_name = list(text_seed + '#' * (max_length - offset))
        temperature = generator_hyperparameters.get('temperature')
        epsilon = generator_hyperparameters.get('epsilon')
        probability = (1 - epsilon)
        vocab_size = generator_hyperparameters.get('vocab_size')
        seed(generator_hyperparameters.get('seed'))
    if index_to_character_dict is None:
        index_to_character_dict = _dutils.load_dictionary('dictionary/idx2char_dict.json')
    if vocab_size is None:
        vocab_size = len(index_to_character_dict)
    # Generate new character from current sequence
    for i in range(max_length):
        #Temperature to deal with epsilon value change over time
        anneal_temperature = ((i+1)/max_length)*temperature
        # print("Anneal Temperature: {}\n".format(anneal_temperature))
        # Convert current sequence to one hot vector
        x_ohe = _dutils.encode_entry(generated_name)
        x_cat = np.array(x_ohe)
        #Add a dimension so that it matches what the model wants to see.
        #NOTE: Expected dimensions are (batch_size, text_length)
        x_cat = np.expand_dims(x_cat, 0)

        # Predict new character probabilities
        # Actually this output a list of probabilities for each character
        character_probabilities = trained_model.predict(x_cat, batch_size=1)
        
        #Exploit vs. Explore value generated
        value = random()
        #If the choice is exploit, get the character with the highest probability
        if value > probability:
            # Extract the best character (and its probability)
            try:
                next_character = index_to_character_dict[np.argmax(character_probabilities)]
            except KeyError:#KeyError occurs if you load from JSON, as keys become strings instead of int
                next_character = index_to_character_dict[str(np.argmax(character_probabilities))]
            # next_character_probability = np.max(character_probabilities)
        #If the choice is explore, then select a character at random.
        else:
            # Choose a random character index according to their probabilities (and its probability)
            next_character_index = np.random.choice(range(vocab_size), p=character_probabilities.ravel())
            # new_char_prob = character_probabilities[0][next_character_index]
            try:
                next_character = index_to_character_dict[next_character_index]
            except KeyError:#KeyError occurs if you load from JSON, as keys become strings instead of int
                next_character = index_to_character_dict[str(next_character_index)]
        # Update the generated name with the new character
        generated_name[i+offset] = next_character
        # Update counters
        epsilon *= (1- anneal_temperature)
        probability = (1 - epsilon)

        #If the next character is the <END> or <PAD> token, respectively
        if (next_character == '%' or next_character == '#'):
            break

    # Clean the generated name
    final_name = ''.join(generated_name)
    final_name = _dutils.remove_characters(final_name)
    if generation_type == 'encode':
        return _dutils.encode_entry(final_name)
    else:
        return final_name

def train(dataroot_path=None, weights_directory=None, num_samples=None):
    # print(tf.__version__)
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # tf.debugging.set_log_device_placement(True)
    text = _dutils.open_file('item_names_lower.txt')

    idx2char, char2idx = _dutils.create_dictionaries(text, True)
    dictionary_size = len(idx2char.values())
    '''
        Load necessary hyperparameters from JSON files.
    '''
    general_hyperparameters = _dutils.load_dictionary('hyperparameters/hyperparameters.json')
    #If, for some reason the JSON value is not the same, update the values
    if general_hyperparameters.get('vocab_size') != dictionary_size:
        general_hyperparameters['vocab_size'] = dictionary_size
    model_hyperparameters = _dutils.load_dictionary('hyperparameters/model_hyperparameters.json')
    '''
        Set the necessary logging directories
    '''
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir,
                                    "weights-improvement-{epoch:02d}")
    '''
        Set callback functions here.
        Tensorboard callback doesn't seem to
        work well, so I take it out.
    '''
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.8, patience=2, min_lr=0.0001)
    # Name of the checkpoint files
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                            save_best_only=True, mode='max',
                                                            save_weights_only=True)

    '''
        Prepare the model for training.
    '''
    model = _model.prepare_model(model_hyperparameters, load_weights=False, weight_directory=checkpoint_dir, verbose=False)
    '''
        Prepare the dataset for training.
    '''
    padded_encoded_text = _dutils.preprocess_dataset(text)
    text_train, text_test = _dutils.create_train_test_data(padded_encoded_text, general_hyperparameters['seq_len'], num_samples=num_samples, rand_state=420, verbose=True)
    train_dataset = _dutils.create_dataset(text_train, dictionary_size, idx2char, char2idx, True, None)
    test_dataset = _dutils.create_dataset(text_test, dictionary_size, idx2char, char2idx, True, None)
    dset = train_dataset.shuffle(general_hyperparameters['buffer_size']).batch(general_hyperparameters['batch_size'])
    t_dset = test_dataset.shuffle(general_hyperparameters['buffer_size']).batch(general_hyperparameters['batch_size'])
    # Begin training the model.
    history = model.fit(dset,
                        epochs=general_hyperparameters['epochs'],
                        validation_data=t_dset,
                        validation_freq=1, verbose=1, callbacks=[reduce_lr, checkpoint_callback])
    history_json_file = "history_{}.json".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    _dutils.save_dictionary(history, history_json_file)

def test_model(num_instances=None):
    """
        Test the model by generating item names.
    """
    idx2char = _dutils.load_dictionary('dictionary/idx2char_dict.json')
    generation_hyperparameters = _dutils.load_dictionary('hyperparameters/generation_hyperparameters.json')
    checkpoint_dir = './training_checkpoints'
    final_model_path = os.path.join(checkpoint_dir, 'final_model.h5')
    generated_names = np.zeros(shape=(num_instances, generation_hyperparameters.get("max_length")))
    model = load_model(final_model_path, custom_objects={'loss': _model.loss})
    for instance in range(num_instances):
        generated_name = generate_text(model, idx2char, generation_hyperparameters)
        generated_names[instance] = generated_name
    return generated_names

def name_generator_function(model, num_instances=None):
    idx2char = _dutils.load_dictionary('dictionary/idx2char_dict.json')
    generation_hyperparameters = _dutils.load_dictionary('hyperparameters/generation_hyperparameters.json')
    generated_names = np.zeros(shape=(num_instances, generation_hyperparameters.get("max_length")), dtype=int)
    for instance in range(num_instances):
        generated_name = generate_text(model, idx2char, generation_hyperparameters, 'encode')
        generated_name.extend([0] * (generation_hyperparameters.get("max_length") - len(generated_name)))
        # print(_dutils.decode_entry(generated_name))
        generated_names[instance] = generated_name
    return generated_names


if __name__ == '__main__':
    print("\n\n\nUsing arguments: {}\n\n\n".format(args))
    if args.task == 'train':
        train(args.dataroot_path, args.weight_directory, args.num_samples)
    elif args.task == 'test':
        test_model(args.num_generate)
    else:
        assert False, 'unkown task'
