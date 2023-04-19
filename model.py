import string

import numpy as np

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import io
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalAveragePooling1D, Flatten, Dropout, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv1D, MaxPool1D
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics.pairwise import cosine_similarity
import zipfile


class model(object):
    def __init__(self):

        self.tokenizer = None
        self.model = None
        self.vocabulary = None
        self.features = None
        self.labels = None
        self.output_length = None
        self.model_trained = False

    def process_data(self, dataset):
        dataset['inputs'] = dataset['inputs'].apply(lambda sequence:
                                                    [ltrs.lower() for ltrs in sequence if
                                                     ltrs not in string.punctuation])
        dataset['inputs'] = dataset['inputs'].apply(lambda wrd: ''.join(wrd))
        self.tokenizer = Tokenizer(num_words=1000)
        self.tokenizer.fit_on_texts(dataset['inputs'])
        train = self.tokenizer.texts_to_sequences(dataset['inputs'])
        self.features = pad_sequences(train)
        self.input_shape = self.features.shape[1]
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(dataset['tags'])
        self.le.fit_transform(dataset['tags'])

        self.vocabulary = len(self.tokenizer.word_index)

        self.output_length = self.le.classes_.shape[0]

    def tain_model(self, dataset, epochs, batch_size):
        self.process_data(dataset)

        # LSTM MODEL

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(self.vocabulary + 1, 300),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.2)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.output_length, activation='softmax')
        ])


        glove_zip = "glove.6B.300d.txt.zip"
        embeddings_index = {}

        with zipfile.ZipFile(glove_zip, "r") as zip_ref:
            with zip_ref.open("glove.6B.300d.txt") as file_:
                for line in file_:
                    arr = line.split()
                    single_word = arr[0].decode('utf-8')
                    w = np.asarray(arr[1:], dtype='float32')
                    embeddings_index[single_word] = w



        max_words = self.vocabulary + 1
        word_index = self.tokenizer.word_index
        embedding_matrix = np.zeros((max_words, 300)).astype(object)
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        self.model.layers[0].set_weights([embedding_matrix])
        self.model.layers[0].trainable = True

        self.model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

        earlyStopping = EarlyStopping(monitor='loss', patience=200, mode='min', restore_best_weights=True)
        print(self.model.summary())

        history_training = self.model.fit(self.features, self.labels, epochs=epochs, batch_size=batch_size,
                                          callbacks=[earlyStopping])
        self.model_trained = True

    def predict(self, query):
        texts = []
        pred_input = query
        pred_input = [letters.lower() for letters in pred_input if letters not in string.punctuation]
        pred_input = ''.join(pred_input)
        texts.append(pred_input)
        pred_input = self.tokenizer.texts_to_sequences(texts)
        pred_input = np.array(pred_input).reshape(-1)
        pred_input = pad_sequences([pred_input], self.input_shape)
        output = self.model.predict(pred_input)

        # compute the cosine similarity between the input sequence and the sequence of each label
        label_embeddings = self.model.layers[0].get_weights()[0]
        input_embedding = np.mean([label_embeddings[i] for i in pred_input[0]], axis=0)
        sim_scores = cosine_similarity([input_embedding], label_embeddings)[0]

        # find the label with the highest cosine similarity
        max_sim_idx = np.argmax(sim_scores)
        max_sim_score = sim_scores[max_sim_idx]

        # if the maximum cosine similarity is less than 0.60, return "INVALID"
        print(query, max_sim_score)

        pred_value = output[0][output.argmax()]
        print("Pred ", pred_value)
        if pred_value < 0.50 or max_sim_score < 0.50:
            return "INVALID"
        output = output.argmax()
        response_tag = self.le.inverse_transform([output])[0]

        return response_tag
