import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
import keras.optimizers
from melodypreprocessor import MelodyPreprocessor
from melodygenerator import MelodyGenerator
from transformer import Transformer

from play_melody import play_melody

# Global parameters
EPOCHS = 32
BATCH_SIZE = 32
DATA_PATH = "dataset2.json"
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 1500

# Loss function and optimizer
sparse_categorical_crossentropy = SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)
optimizer = keras.optimizers.adam_v2.Adam()


def train(train_dataset, transformer, epochs):
    """
    Trains the Transformer model on a given dataset for a specified number of epochs.

    Parameters:
        train_dataset (tf.data.Dataset): The training dataset.
        transformer (Transformer): The Transformer model instance.
        epochs (int): The number of epochs to train the model.
    """
    print("Training the model...")
    for epoch in range(epochs):
        total_loss = 0
        # Iterate over each batch in the training dataset
        for (batch, (input, target)) in enumerate(train_dataset):
            # Perform a single training step
            batch_loss = _train_step(input, target, transformer)
            total_loss += batch_loss
            print(
                f"Epoch {epoch + 1} Batch {batch + 1} Loss {batch_loss.numpy()}"
            )


def create_look_ahead_mask(seq_len):
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_padding_mask(sequence):
    mask = tf.cast(tf.math.equal(sequence, 0), tf.float32)  # 填充值为 0
    return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

@tf.function
def _train_step(input, target, transformer):
    """
    Performs a single training step for the Transformer model.

    Parameters:
        input (tf.Tensor): The input sequences.
        target (tf.Tensor): The target sequences.
        transformer (Transformer): The Transformer model instance.

    Returns:
        tf.Tensor: The loss value for the training step.
    """
    # Prepare the target input and real output for the decoder
    # Pad the sequences on the right by one position
    target_input = _right_pad_sequence_once(target[:, :-1])
    target_real = _right_pad_sequence_once(target[:, 1:])

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation
    with tf.GradientTape() as tape:
        # Forward pass through the transformer model
        # for decoder
        enc_padding_mask = create_padding_mask(input)
        dec_padding_mask = create_padding_mask(target_input)
        look_ahead_mask = create_look_ahead_mask(tf.shape(target_input)[1])

        predictions = transformer(input, target_input, True, enc_padding_mask, look_ahead_mask, dec_padding_mask)

        # Compute loss between the real output and the predictions
        loss = _calculate_loss_with_repetition_penalty(target_real, predictions)

    # Calculate gradients with respect to the model's trainable variables
    gradients = tape.gradient(loss, transformer.trainable_variables)

    # Apply gradients to update the model's parameters
    gradient_variable_pairs = zip(gradients, transformer.trainable_variables)
    optimizer.apply_gradients(gradient_variable_pairs)

    # Return the computed loss for this training step
    return loss

def _calculate_loss(real, pred):
    """
    Computes the loss between the real and predicted sequences.

    Parameters:
        real (tf.Tensor): The actual target sequences.
        pred (tf.Tensor): The predicted sequences by the model.

    Returns:
        average_loss (tf.Tensor): The computed loss value.
    """

    # Compute loss using the Sparse Categorical Crossentropy
    loss_ = sparse_categorical_crossentropy(real, pred)

    # Create a mask to filter out zeros (padded values) in the real sequences
    boolean_mask = tf.math.equal(real, 0)
    mask = tf.math.logical_not(boolean_mask)

    # Convert mask to the same dtype as the loss for multiplication
    mask = tf.cast(mask, dtype=loss_.dtype)

    # Apply the mask to the loss, ignoring losses on padded positions
    loss_ *= mask

    # Calculate average loss, excluding the padded positions
    total_loss = tf.reduce_sum(loss_)
    number_of_non_padded_elements = tf.reduce_sum(mask)
    average_loss = total_loss / number_of_non_padded_elements

    return average_loss
def _calculate_loss_with_repetition_penalty(real, pred):
    """
    计算带有重复惩罚的损失。

    Parameters:
        real (tf.Tensor): 实际的目标序列。
        pred (tf.Tensor): 模型的预测序列。

    Returns:
        tf.Tensor: 平均损失值（标量）。
    """
    # 基础损失计算
    loss_ = sparse_categorical_crossentropy(real, pred)  # 每个时间步的损失
    mask = tf.cast(tf.math.not_equal(real, 0), dtype=loss_.dtype)  # 忽略填充部分
    loss_ *= mask  # 只计算非填充部分的损失

    # 计算重复惩罚
    pred_ids = tf.argmax(pred, axis=-1, output_type=tf.int32)  # 确保输出为 int32
    unique_counts = tf.map_fn(
        lambda seq: tf.math.bincount(seq, minlength=vocab_size), pred_ids, dtype=tf.int32
    )
    repetition_penalty = tf.reduce_sum(unique_counts * unique_counts, axis=-1)
    repetition_penalty = tf.cast(repetition_penalty, dtype=loss_.dtype)

    # 归一化基础损失
    total_loss = tf.reduce_sum(loss_)  # 所有非填充部分的损失总和
    num_non_padding = tf.reduce_sum(mask)  # 非填充部分的数量
    average_loss = total_loss / num_non_padding  # 计算平均损失

    # 加入重复惩罚（标量形式）
    total_loss_with_penalty = average_loss + 0.1 * tf.reduce_mean(repetition_penalty)

    return total_loss_with_penalty


def _right_pad_sequence_once(sequence):
    """
    Pads a sequence with a single zero at the end.

    Parameters:
        sequence (tf.Tensor): The sequence to be padded.

    Returns:
        tf.Tensor: The padded sequence.
    """
    return tf.pad(sequence, [[0, 0], [0, 1]], "CONSTANT")


if __name__ == "__main__":
    melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    train_dataset = melody_preprocessor.create_training_dataset()
    vocab_size = melody_preprocessor.number_of_tokens_with_padding

    transformer_model = Transformer(
        num_layers=6,
        d_model=64,
        num_heads=8,
        d_feedforward=128,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        max_num_positions_in_pe_encoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        max_num_positions_in_pe_decoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        dropout_rate=0.3,
    )

    train(train_dataset, transformer_model, EPOCHS)

    print("Generating a melody...")
    melody_generator = MelodyGenerator(
        transformer_model, melody_preprocessor.tokenizer,200
    )
    start_sequence = ["C4-1.0"]
    new_melody = melody_generator.generate(start_sequence)
    print(new_melody)
    play_melody(new_melody.split(' '))

    print("Generating a melody...")
    start_sequence = ["A4-1.0"]
    new_melody = melody_generator.generate(start_sequence)
    print(new_melody)
    play_melody(new_melody.split(' '))

