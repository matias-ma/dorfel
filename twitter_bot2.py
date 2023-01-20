#!/usr/local/bin/python3.9

import tweepy
import random
from datetime import datetime, timedelta
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
import numpy as np

# Enter your Twitter API credentials
consumer_key = "2f4BZdqeSZ4RMh6sjOxOVwn5O"
consumer_secret = "sppCMBwhaelEOiE4v3GXeIdHHzoNsgg9RAhDddZ9YtUe1Gq9Fc"
access_token = "1475748533064151041-8l5piBAY7DlRFl6RKHyLACMB9UiXph"
access_token_secret = "PdmPOF5RapPxvd8lOOqIBlOGiKbxSV1OmQw9UQxXaAAlY"

author_list = ['Murakami', 'Nabokov', 'Calvino', 'Hemingway', 'Many']
i=0

with open('/Users/student/Downloads/many_murakami.txt') as f:
    text0 = f.read()

with open('/Users/student/Downloads/many_nabokov.txt') as f:
    text1 = f.read()

with open('/Users/student/Downloads/many_calvino.txt') as f:
    text2 = f.read()

with open('/Users/student/Downloads/many_hemingway.txt') as f:
    text3 = f.read()

with open('/Users/student/Downloads/combined.txt') as f:
    text4 = f.read()

while True:
    try:
        auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
        api = tweepy.API(auth)
        author = random.randint(0,4)

        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        start_word = random.choice(alphabet)

        char_count = random.randint(200,270)

        if author == 0:
            text = text0

        elif author == 1:
            text = text1

        elif author == 2:
            text = text2

        elif author == 3:
            text = text3

        elif author == 4:
            text = text4

        vocab = sorted(set(text))
        example_texts = ['abcdefg', 'xyz']

        chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
        ids_from_chars = tf.keras.layers.StringLookup(
            vocabulary=list(vocab), mask_token=None)
    
        ids = ids_from_chars(chars)
        chars_from_ids = tf.keras.layers.StringLookup(
            vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

        chars = chars_from_ids(ids)
        tf.strings.reduce_join(chars, axis=-1).numpy()
        def text_from_ids(ids):
          return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

        all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)


        seq_length = 100

        sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

        BATCH_SIZE = 2048
        BUFFER_SIZE = 10000

        def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)


        dataset = (
            dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

        # Length of the vocabulary in StringLookup Layer
        vocab_size = len(ids_from_chars.get_vocabulary())

        # The embedding dimension
        embedding_dim = 256

        # Number of RNN units
        rnn_units = 1024

        class MyModel(tf.keras.Model):
          def __init__(self, vocab_size, embedding_dim, rnn_units):
            super().__init__(self)
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(rnn_units,
                                           return_sequences=True,
                                           return_state=True)
            self.dense = tf.keras.layers.Dense(vocab_size)

          def call(self, inputs, states=None, return_state=False, training=False):
            x = inputs
            x = self.embedding(x, training=training)
            if states is None:
              states = self.gru.get_initial_state(x)
            x, states = self.gru(x, initial_state=states, training=training)
            x = self.dense(x, training=training)

            if return_state:
              return x, states
            else:
              return x

        model = MyModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            rnn_units=rnn_units)

        for input_example_batch, target_example_batch in dataset.take(1):
            example_batch_predictions = model(input_example_batch)


        class OneStep(tf.keras.Model):
          def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
            super().__init__()
            self.temperature = temperature
            self.model = model
            self.chars_from_ids = chars_from_ids
            self.ids_from_chars = ids_from_chars

            # Create a mask to prevent "[UNK]" from being generated.
            skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
            sparse_mask = tf.SparseTensor(
                # Put a -inf at each bad index.
                values=[-float('inf')]*len(skip_ids),
                indices=skip_ids,
                # Match the shape to the vocabulary
                dense_shape=[len(ids_from_chars.get_vocabulary())])
            self.prediction_mask = tf.sparse.to_dense(sparse_mask)

          @tf.function
          def generate_one_step(self, inputs, states=None):
            # Convert strings to token IDs.
            input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
            input_ids = self.ids_from_chars(input_chars).to_tensor()

            # Run the model.
            # predicted_logits.shape is [batch, char, next_char_logits]
            predicted_logits, states = self.model(inputs=input_ids, states=states,
                                                  return_state=True)
            # Only use the last prediction.
            predicted_logits = predicted_logits[:, -1, :]
            predicted_logits = predicted_logits/self.temperature
            # Apply the prediction mask: prevent "[UNK]" from being generated.
            predicted_logits = predicted_logits + self.prediction_mask

            # Sample the output logits to generate token IDs.
            predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
            predicted_ids = tf.squeeze(predicted_ids, axis=-1)

            # Convert from token ids to characters
            predicted_chars = self.chars_from_ids(predicted_ids)

            # Return the characters and model state.
            return predicted_chars, states

        one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

        if author == 0:
            model.load_weights('/Users/student/Desktop/tf_models/many_murakami.h5')

        elif author == 1:
            model.load_weights('/Users/student/Desktop/tf_models/many_nabokov.h5')

        elif author == 2:
            model.load_weights('/Users/student/Desktop/tf_models/many_calvino.h5')

        elif author == 3:
            model.load_weights('/Users/student/Desktop/tf_models/many_hemingway.h5')

        elif author == 4:
            model.load_weights('/Users/student/Desktop/tf_models/gutenberg3.h5')

        states = None
        next_char = tf.constant([start_word])
        result = [next_char]

        for n in range(int(char_count)):
          next_char, states = one_step_model.generate_one_step(next_char, states=states)
          result.append(next_char)

        result = tf.strings.join(result)
        tweet = author_list[author]+':\n'+result[0].numpy().decode('utf-8')
        api.update_status(tweet)
        i+=1
        print(i)

    except:
        pass

# with open('/Users/student/Desktop/ai_calvino_novel.txt', 'a+') as f:
#     f.write(result[0].numpy().decode('utf-8'))