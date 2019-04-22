import os
import re
import time
import pickle
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
import tensorflow_datasets as tfds

tf.compat.v1.logging.set_verbosity('ERROR')

# Download and extract dataset
path_to_zip = tf.keras.utils.get_file(
    'cornell_movie_dialogs.zip',
    origin=
    'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
    extract=True)

path_to_dataset = os.path.join(
    os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
path_to_movie_conversations = os.path.join(path_to_dataset,
                                           'movie_conversations.txt')

NUM_SAMPLES = 25000


def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  # adding a start and an end token to the sentence
  return sentence


def load_conversations():
  # dictionary of line id to text
  id2line = {}
  with open(path_to_movie_lines, errors='ignore') as file:
    lines = file.readlines()
  for line in lines:
    parts = line.replace('\n', '').split(' +++$+++ ')
    id2line[parts[0]] = parts[4]

  inputs, outputs = [], []
  with open(path_to_movie_conversations, 'r') as file:
    lines = file.readlines()
  for line in lines:
    parts = line.replace('\n', '').split(' +++$+++ ')
    # get conversation in a list of line ID
    conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
    for i in range(len(conversation) - 1):
      inputs.append(preprocess_sentence(id2line[conversation[i]]))
      outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
      if len(inputs) >= NUM_SAMPLES:
        return inputs, outputs
  return inputs, outputs


questions, answers = load_conversations()

print('Sample question: {}'.format(questions[0]))
print('Sample answer: {}'.format(answers[0]))

# Build tokenizer using tfds for both questions and answers
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2

# Maximum sentence length
MAX_LENGTH = 40


# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []
  for (input, output) in zip(inputs, outputs):
    # tokenize sentence
    tokenized_input = START_TOKEN + tokenizer.encode(input) + END_TOKEN
    tekenized_output = START_TOKEN + tokenizer.encode(output) + END_TOKEN
    # check tokenized sentence max length
    if len(tokenized_input) <= MAX_LENGTH and len(
        tekenized_output) <= MAX_LENGTH:
      tokenized_inputs.append(tokenized_input)
      tokenized_outputs.append(tekenized_output)
  # pad tokenized sentences
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  return tokenized_inputs, tokenized_outputs


questions, answers = tokenize_and_filter(questions, answers)

print('Train set size: {}'.format(len(questions)))
print('Vocab size: {}'.format(tokenizer.vocab_size))

BATCH_SIZE = 64
BUFFER_SIZE = 20000

train_ds = tf.data.Dataset.from_tensor_slices((questions, answers))
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(BUFFER_SIZE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)


def scaled_dot_product_attention(query, key, value, mask):
  """Calculate the attention weights. """
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k)
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, inputs, key, value, mask):
    batch_size = tf.shape(inputs)[0]

    # linear layers
    query = self.query_dense(inputs)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # scaled dot-product attention
    scaled_attention, attention_weights = scaled_dot_product_attention(
        query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # concatenation of heads
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # final linear layer
    outputs = self.dense(concat_attention)

    return outputs, attention_weights


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(
      np.arange(position)[:, np.newaxis],
      np.arange(d_model)[np.newaxis, :], d_model)

  # apply sin to even indices in the array; 2i
  sines = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  cosines = np.cos(angle_rads[:, 1::2])

  pos_encoding = np.concatenate([sines, cosines], axis=-1)

  pos_encoding = pos_encoding[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


# Mask all the pad tokens (value `0`) in the batch to ensure the model does not
# treat padding as input.
def create_padding_mask(sequence):
  sequence = tf.cast(tf.math.equal(sequence, 0), tf.float32)
  return sequence[:, tf.newaxis, tf.newaxis, :]


# Look-ahead mask to mask the future tokens in a sequence.
# i.e. To predict the third word, only the first and second word will be used
def create_look_ahead_mask(size):
  return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


def create_masks(inputs, outputs):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inputs)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inputs)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(outputs)[1])
  dec_target_padding_mask = create_padding_mask(outputs)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask


# Define encoder layer
def get_encoder_layer(max_length, units, d_model, num_heads, dropout):
  inputs = tf.keras.Input(shape=(max_length, d_model), name="inputs")
  mask = tf.keras.Input(shape=(1, 1, max_length), name="mask")

  attention, _ = MultiHeadAttention(
      d_model, num_heads)(inputs=[inputs, inputs, inputs, mask])
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)

  layer_norm_1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)
  outputs = tf.keras.layers.Dense(units=units, activation='relu')(layer_norm_1)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(layer_norm_1 + outputs)

  encoder_layer = tf.keras.Model(
      inputs=[inputs, mask], outputs=outputs, name='encoder_layer')

  return encoder_layer


NUM_LAYERS = 4
D_MODEL = 128
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    units=UNITS,
    vocab_size=VOCAB_SIZE,
    dropout=DROPOUT)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')


@tf.function
def train_step(inputs, targets):
  # use teacher forcing, decoder use the previous target as input
  decoder_inputs = targets[:, :-1]
  # remove START_TOKEN from targets
  cropped_targets = targets[:, 1:]

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
      inputs, decoder_inputs)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(
        inputs,
        dec_inputs=decoder_inputs,
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=combined_mask,
        dec_padding_mask=dec_padding_mask,
        training=True)
    loss = loss_function(cropped_targets, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(cropped_targets, predictions)


CKPT_PATH = "runs/"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, CKPT_PATH, max_to_keep=3)
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Restored checkpoint {}'.format(ckpt_manager.latest_checkpoint))

EPOCHS = 20
# Number of batches per epoch
NUM_BATCH = int(np.ceil(len(questions) / BATCH_SIZE))

for epoch in range(EPOCHS):
  # reset metrics
  train_loss.reset_states()
  train_accuracy.reset_states()

  print('Epoch {}'.format(epoch + 1))

  start = time.time()

  with tqdm(total=NUM_BATCH) as pbar:
    for inputs, targets in train_ds:
      train_step(inputs, targets)
      pbar.update(1)

  end = time.time()

  print('Loss {:.4f} Accuracy {:.2f} Time {:.2f}s'.format(
      train_loss.result(),
      train_accuracy.result() * 100,
      end - start,
  ))

  if epoch % 2 == 0:
    ckpt_save_path = ckpt_manager.save()
    print('Saved checkpoint {}'.format(ckpt_save_path))

  print('')


def evaluate(sentence):
  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        sentence, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(
        sentence,
        dec_inputs=output,
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=combined_mask,
        dec_padding_mask=dec_padding_mask,
        training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, tokenizer.vocab_size + 1):
      return tf.squeeze(output, axis=0), attention_weights

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights


def predict(sentence):
  prediction, attention_weights = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  print('Input: {}'.format(sentence))
  print('Output: {}'.format(predicted_sentence))

  return predicted_sentence


predict('Where have you been?')
print('')

predict("It's a trap")
print('')

# test the model with its previous output as input
sentence = 'I am not crazy, my mother had me tested.'
for _ in range(5):
  sentence = predict(sentence)
  print('')
