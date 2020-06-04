import json
from random import random, seed
from unicodedata import normalize

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_dictionary(file_name=None):
    if file_name:
        return json.load(open(file_name, 'r'))
    else:
        raise ValueError("No file_name used.")

def save_dictionary(dictionary_to_save=None, file_name=None):
    """
        Save a dictionary as a JSON file.
    """
    if dictionary_to_save is None:
        raise ValueError("Dictionary must have values present.")
    elif not isinstance(dictionary_to_save, dict):
        raise TypeError("Expected type dictionary to save. Got {}".format(str(type(dictionary_to_save))))
    with open(file_name, 'w') as _file:
        json.dump(dictionary_to_save, _file)

def decode_entry(enc_text=None, idx2char_dict=None):
    """
        Decode a string of encoded text.
        
        The string, in this case, does not use One Hot Encoding
        (i.e., zero-vector with index of character set to 1).
        Instead, characters are encoded with unique integer ID.
        
        Parameters:
        -----------
        text_to_decode : str
            The text to be decoded.
        idx2char_dict : dict(int:char)
            Dictionary mapping integers to characters

        Returns:
        --------
        decoded_text : list(str) OR str
            A list corresponding to the decoded version
            of each integer present in text_to_decode.
            If only one character is given, then it returns
            the character.
    """
    if idx2char_dict is None:
        idx2char_dict = load_dictionary('dictionary/idx2char_dict.json')
    if isinstance(enc_text, np.ndarray):
        try:
            return [idx2char_dict[enc_char] for enc_char in enc_text]
        except KeyError:
            return [idx2char_dict[str(enc_char)] for enc_char in enc_text]
    elif isinstance(enc_text, np.int64):
        try:
            return idx2char_dict[enc_text]
        except KeyError:
            return idx2char_dict[str(enc_text)]
    else:
        raise ValueError("Expected enc_text to be either numpy ndarray or numpy int64. Got {}".format(type(enc_text)))

def encode_entry(text_to_encode=None, char2idx_dict=None):
    """
        Encode a piece of text to integer format, character-by-character.

        Encoding in this case is not One Hot Encoding
        (i.e., zero-vector with index of character set to 1).
        Instead, characters are encoded with unique integer ID.

        Parameters:
        -----------
        text_to_encode : str
            The text to be encoded.
        char2idx_dict : dict(char:int)
            Dictionary mapping characters to integers

        Returns:
        --------
        encoded_text : list(int)
            A list corresponding to the encoded version
            of each character present in text_to_encode.
    """
    if char2idx_dict is None:
        char2idx_dict = load_dictionary('dictionary/char2idx_dict.json')
    if text_to_encode:
        return [char2idx_dict[c] for c in text_to_encode]
    else:
        return

def decode_categorical_entry(cat_entry):
    result = []
    if not isinstance(cat_entry, np.ndarray):
        cat_entry = cat_entry.numpy().transpose()
        indices = np.argmax(cat_entry, axis=(cat_entry.ndim-1))
    else:
        indices = np.argmax(cat_entry, axis=(1))
    result = decode_entry(indices)
    return result

def open_file(filename, verbose=False):
    """
        Open a text file and create a list of entries.

        Opens a text file and removes extraneous characters
        and forces the text to be in unicode format.
        If verbosity is toggled, then the number of entries
        and the number of unique entries is listed.

        Parameters
        ----------
        filename : str
            The name of the file to open. If the data file
            is not located in the same directory, an absolute
            filepath will be needed.
        verbose : boolean
            A flag incidating if auxiliary information will be
            provided or not.
        
        Returns
        -------
        name_list : list(str)
            A list of all entries in the text file.
    """
    name_list = []
    with open(filename, 'r') as _file:
        for line in _file:
            line = line.replace('\\ufeff', '')
            line = normalize('NFKD', line)
            line = line.rstrip() #remove all trailing whitespace including newline
            line = remove_characters(line)
            name_list.append(line)
    if verbose:
        print('Number of entries: {} entries'.format(len(name_list)))
        print('Number of unique entries: {} entries'.format(len(list(set(name_list)))))
    return name_list
    
def remove_characters(text, characters_to_remove=None):
    """
        Remove various auxiliary characters from a string.

        This function uses a hard-coded string of 'undesirable'
        characters (if no such string is provided),
        and removes them from the text provided.

        Parameters:
        -----------
        text : str
            A piece of text to remove characters from.
        characters_to_remove : str
            A string of 'undesirable' characters to remove from the text.

        Returns:
        --------
        text : str
            A piece of text with undesired characters removed.
    """
    # chars = "\\`*_{}[]()<>#+-.!$%@"
    if characters_to_remove is None:
        characters_to_remove = "\\`*_{}[]()<>#+!$%@"
    for c in characters_to_remove:
        if c in text:
            text = text.replace(c, '')
    return text

def preprocess_dataset(input_dataset=None, pad_length=150):
    """
        Pre-process the dataset.

        Given a list of item names, prepend the <START> token,
        then append the <END> token and <PAD> until the sequence
        length is reached. The assumed length is 150 to fit the
        short-form text presented in the Crepe architecture.

        Parameters
        ----------
        input_dataset : list(str)
            The list of all item names
        
        Returns
        -------
        padded_sequences : list(int)
            The list of all item names, with <START> and <END> tokens.
            Furthermore, all entries are now uniformly 150 characters.
        
        Raises
        ------
        ValueError
            If no dataset is provided, this error is raised.
    """
    if input_dataset is None:
        raise ValueError("No input_dataset provided.")
    #Prepend and Append each datum with the start and end tokens.
    encoded_input_dataset = ['*' + input_datum + '%' + '#' * (pad_length - len(input_datum) - 2) for input_datum in input_dataset]
    return encoded_input_dataset

def create_dictionaries(input_text, verbose=False, save_data=None):
    '''
        Create the dictionaries mapping characters to integers.

        Given an input text, it converts the text into a
        lexicographically sorted list, and outputs two dictionaries.

        Parameters
        ----------
        input_text : list(str)
            The list of all item names.
        verbose : boolean
            Flag toggling printing of information.
        save_data : boolean
            Flag toggling if dictionaries should be saved.
        Returns
        -------
        idx2char_dict : dict(int:char)
            Dictionary mapping integers to characters
        char2idx_dict : dict(char:int)
            Dictionary mapping characters to integers
    '''
    vocab = sorted({l for word in input_text for l in word})
    if verbose:
        print(vocab)
        print('Vocab has {} unique characters'.format(len(vocab)))
    #Create the dictionary over the vocabulary, starting at the fourth entry.
    idx2char_dict = dict(enumerate(vocab, 4))
    #Insert the key-value pairs for the tokens.
    idx2char_dict[0] = '#'  # <PAD>
    idx2char_dict[1] = '@'  # <UNK>
    idx2char_dict[2] = '*'  # <START>
    idx2char_dict[3] = '%'  # <END>
    #Create the character-to-index dictionary by reversing the previous dictionary.
    char2idx_dict = {value: key for key, value in idx2char_dict.items()}
    if verbose:
        print(list(zip(idx2char_dict.keys(), idx2char_dict.values())))
        print('Index to Character has {} entries'.format(len(idx2char_dict.values())))

        print(list(zip(char2idx_dict.keys(), char2idx_dict.values())))
        print('Character to Index has {} entries'.format(len(char2idx_dict.values())))
    if save_data:
        save_dictionary(idx2char_dict, 'dictionary/idx2char_dict.json')
        save_dictionary(char2idx_dict, 'dictionary/char2idx_dict.json')
    
    return idx2char_dict, char2idx_dict

'''
    General sequence of events for creating a dataset goes as follows:
        1. create_train_test_data
        2. create_dataset
        3. create_example_label_data (called during create_dataset)
'''

def create_example_label_data(input_text=None, idx2char=None, char2idx=None, save_data=None):
    """
        Create OHE sequences for all instances, along with their character labels.

        Given the input text, for each instance present in the input text,
        begin with a sequence of only the first character (likely the <START> token),
        and the label being the second character. Subsequent iterations will append the
        previous label onto the instance, and the new label will be the next character
        present. If the previous label is an <END> or <PAD> token, then the label will
        always be the <END> token. The input sequence is re-padded to 150 characters,
        and both are encoded.
        
        Example: <START>A Bloodstained Envelope<END><PAD>xN
        
        Input: OHE array of "<START><END><PAD>x98"
        Label: "A" (encoded to be integer)

        Input: OHE matrix of "<START>A<END><PAD>x97"
        Label: <SPACE> (encoded to be integer)

        This process continues until the sequence length is reached.
        <PAD> characters and <END> characters will always have the <END> label attached.
        
        The OHE matrix should have shape (sequence_lengthxcharacter_set_size)

        Parameters
        ----------
        input_text : list(str)
            The list of all instances to process.
            Example: "a bloodstained letter"
        idx2char : dict (int : str)
            A dictionary mapping the encoded integer to its character.
        char2idx : dict (str : int)
            A dictionary mapping the non-encoded character to its encoding.
        save_data : str
            Toggles whether the results should be saved. If so, they are saved
            as the value of save_data.
        
        Returns
        -------
        X_numpy : Numpy Array
            A Numpy array containing the encoded instances.
            Serves as the 'X' portion of a dataset.
        Y_numpy : Numpy Array
            A Numpy array containing the encoded label (e.g., 3) for an instance.
            Serves as the 'y' portion of a dataset.
        
        Note
        ----
        This function uses TQDM for convenience in seeing dataset creation progress.
    """
    encoded_sequences = []
    labels = []
    for datum in tqdm(input_text):
        input_chunk = []
        for split_index in range(len(datum)):
            #Set up the base case
            if split_index < 1:
                input_chunk.append(datum[0])
                label = datum[1]
            #If we hit the last character, or the previous label was a PAD or END character, the label is END
            elif split_index == len(datum)-1 or datum[split_index] == idx2char[3] or datum[split_index] == idx2char[0]:
                label = idx2char[3] #Corresponds to the END token.
            else:
                #Update the chunk window
                input_chunk.append(datum[split_index])
                label = datum[split_index+1]
            #Grab the first chunk and convert the characters into numbers.
            padded_sequence = str(''.join(input_chunk) + '#' * (150 - len(input_chunk)))
            encoded_sequence = [char2idx[c] for c in padded_sequence]
            cat_label = char2idx[label]
            encoded_sequences.append(encoded_sequence)
            labels.append(cat_label)
    
    X_numpy = np.array(encoded_sequences)
    Y_numpy = np.array(labels)

    if save_data:
        np.save(save_data, X_numpy)
        np.save(save_data+"_labels", Y_numpy)
    return X_numpy, Y_numpy

def create_train_test_data(input_text, length_of_sequence, num_samples=None, rand_state=None, verbose=False):
    """
        Create a train/test split of the dataset. Can return a sample of dataset.

        Given a list of text (input_text) and the length of a given sequence, create a training and
        validation/testing set that is padded accordingly. Additional arguments allow for the use of
        a sample of the dataset (recommended for now), along with the number of samples to pull, as
        well as a flag for verbosity.
    """
    if rand_state:
        np.random.seed(rand_state)
    if num_samples:
        input_text = np.random.choice(input_text, num_samples, replace=False)
        if verbose:
            print('Number of entries sampled: {} entries'.format(len(input_text)))
            print('Example - First entry: {}'.format(input_text[0]))
    training_data, testing_data = train_test_split(input_text, test_size=0.1,
                                                   random_state=rand_state)
    if verbose:
        print('Number of entries in training data: {} entries'.format(len(training_data)))
        print('Example - First entry: {}'.format(training_data[0]))
        print('Number of entries in testing data: {} entries'.format(len(testing_data)))
        print('Example - First entry: {}'.format(testing_data[0]))
    return training_data, testing_data

def create_dataset(input_text, vocab_size, idx2char, char2idx, verbose=False, save_data=None):
    """
        Given the input_text, or an output of the create_train_test_data, the size of the vocabulary
        and the length of the sequences, we create a Tensorflow Dataset object to use for training.
        Additionally, it returns the sample_instances.shape[0] as the secondary output.
    """

    sample_instances, sample_labels = create_example_label_data(input_text, idx2char, char2idx, save_data)

    dataset = tf.data.Dataset.from_tensor_slices((sample_instances, sample_labels))
    if verbose:
        for instance, label in dataset.take(5):
            print("Type of i: {}, Type of j: {}\n".format(type(instance), type(label)))
            print("Shape of i: {}, Shape of j: {}\n".format(instance.shape, label.shape))
            print("Input data: {}".format(decode_categorical_entry(instance)))
            print("Target data: {}".format(decode_categorical_entry(label)))
    return dataset
