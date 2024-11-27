import json

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer


class MelodyPreprocessor:

    def __init__(self, dataset_path, batch_size=32):

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.tokenizer = Tokenizer(filters="", lower=False, split=",")
        self.max_melody_length = None
        self.number_of_tokens = None

    @property
    def number_of_tokens_with_padding(self):

        return self.number_of_tokens + 1

    def create_training_dataset(self):
        dataset = self._load_dataset()
        parsed_melodies = [self._parse_melody(melody) for melody in dataset]
        tokenized_melodies = self._tokenize_and_encode_melodies(
            parsed_melodies
        )
        self._set_max_melody_length(tokenized_melodies)
        self._set_number_of_tokens()
        input_sequences, target_sequences = self._create_sequence_pairs(
            tokenized_melodies
        )
        tf_training_dataset = self._convert_to_tf_dataset(
            input_sequences, target_sequences
        )
        return tf_training_dataset

    def _load_dataset(self):
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def _parse_melody(self, melody_str):
        return melody_str.split(", ")

    def _tokenize_and_encode_melodies(self, melodies):
        self.tokenizer.fit_on_texts(melodies)
        tokenized_melodies = self.tokenizer.texts_to_sequences(melodies)
        return tokenized_melodies

    def _set_max_melody_length(self, melodies):
        self.max_melody_length = max([len(melody) for melody in melodies])

    def _set_number_of_tokens(self):
        self.number_of_tokens = len(self.tokenizer.word_index)

    def _create_sequence_pairs(self, melodies):
        input_sequences, target_sequences = [], []
        for melody in melodies:
            for i in range(1, len(melody)):
                input_seq = melody[:i]
                target_seq = melody[1 : i + 1]  # Shifted by one time step
                padded_input_seq = self._pad_sequence(input_seq)
                padded_target_seq = self._pad_sequence(target_seq)
                input_sequences.append(padded_input_seq)
                target_sequences.append(padded_target_seq)
        return np.array(input_sequences), np.array(target_sequences)

    def _pad_sequence(self, sequence):
        return sequence + [0] * (self.max_melody_length - len(sequence))

    def _convert_to_tf_dataset(self, input_sequences, target_sequences):
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_sequences, target_sequences)
        )
        shuffled_dataset = dataset.shuffle(buffer_size=1000)
        batched_dataset = shuffled_dataset.batch(self.batch_size)
        return batched_dataset


if __name__ == "__main__":
    # Usage example
    preprocessor = MelodyPreprocessor("dataset.json", batch_size=32)
    training_dataset = preprocessor.create_training_dataset()