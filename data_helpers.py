import numpy as np
import re
import pandas as pd
import itertools
from collections import Counter

## Check whther need to do more cleaning
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(data_file):
    """
    Loads twwets data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    data_file = pd.read_csv(data_file)

    # Split by words
    x_text = np.array(data_file['text'])
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    y = [[0, 1] if label.strip() == 'neg' else [1, 0] for label in data_file['label']]
    y = np.array(y)
    return [np.array(x_text), y]

def load_data_and_labels_forsvm(data_file):
    """
    Loads twwets data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    data_file = pd.read_csv(data_file)

    # Split by words
    x_text = np.array(data_file['text'])
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    y = [0 if label.strip() == 'neg' else 1 for label in data_file['label']]
    y = np.array(y)
    return [np.array(x_text), y]

def load_embedding_vectors_word2vec(vocabulary, filepath):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filepath, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        binary_len = np.dtype('float32').itemsize * vector_size
        for line_no in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    break
                if ch == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                if ch != b'\n':
                    word.append(ch)
            word = str(b''.join(word), encoding=encoding, errors='strict')
            idx = vocabulary.get(word)
            if idx != 0:
                embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.seek(binary_len, 1)
        f.close()
        return embedding_vectors

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    if data_size < batch_size: batch_size = data_size
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
