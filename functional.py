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

NUM_SAMPLES = 200


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
DATASET_SIZE = len(questions)

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

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
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

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query)
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


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="{}/inputs".format(name))
  padding_mask = tf.keras.Input(
      shape=(1, 1, None), name="{}/padding_mask".format(name))

  attention, _ = MultiHeadAttention(
      d_model, num_heads, name="{}/attn1".format(name))({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


sample_encoder_layer = encoder_layer(
    units=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name="sample_encoder_layer")

tf.keras.utils.plot_model(
    sample_encoder_layer, to_file='encoder_layer.png', show_shapes=True)


def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="{}/inputs".format(name))
  padding_mask = tf.keras.Input(
      shape=(1, 1, None), name="{}/padding_mask".format(name))
  pos_encoding = tf.keras.Input(
      shape=(None, d_model), name="{}/pos_encoding".format(name))

  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings += pos_encoding

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = encoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="encoder_layer_{}".format(i),
    )(inputs=[outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask, pos_encoding], outputs=outputs, name=name)


sample_encoder = encoder(
    vocab_size=8192,
    num_layers=4,
    units=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name="sample_encoder")

tf.keras.utils.plot_model(
    sample_encoder, to_file='encoder.png', show_shapes=True)


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="{}/inputs".format(name))
  enc_outputs = tf.keras.Input(
      shape=(None, d_model), name="{}/encoder_outputs".format(name))
  look_ahead_mask = tf.keras.Input(
      shape=(1, 1, None), name="{}/look_ahead_mask".format(name))
  padding_mask = tf.keras.Input(
      shape=(1, 1, None), name='{}/padding_mask'.format(name))

  attention1, _ = MultiHeadAttention(
      d_model, num_heads, name="{}/attn1".format(name))({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })
  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

  attention2, _ = MultiHeadAttention(
      d_model, num_heads, name="{}/attn2".format(name))({
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


sample_decoder_layer = decoder_layer(
    units=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name="sample_decoder_layer")

tf.keras.utils.plot_model(
    sample_decoder_layer, to_file='decoder_layer.png', show_shapes=True)


def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="decoder"):
  inputs = tf.keras.Input(shape=(None,), name="{}/inputs".format(name))
  enc_outputs = tf.keras.Input(
      shape=(None, d_model), name="{}/encoder_outputs".format(name))
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="{}/look_ahead_mask".format(name))
  padding_mask = tf.keras.Input(
      shape=(1, 1, None), name="{}/padding_mask".format(name))
  pos_encoding = tf.keras.Input(
      shape=(None, d_model), name="{}/pos_encoding".format(name))

  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings += pos_encoding

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = decoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="decoder_layer_{}".format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask, pos_encoding],
      outputs=outputs,
      name=name)


sample_decoder = decoder(
    vocab_size=8192,
    num_layers=4,
    units=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name="sample_decoder")

tf.keras.utils.plot_model(
    sample_decoder, to_file='decoder.png', show_shapes=True)


def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")
  enc_padding_mask = tf.keras.Input(shape=(1, 1, None), name="enc_padding_mask")
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  dec_padding_mask = tf.keras.Input(shape=(1, 1, None), name="dec_padding_mask")
  enc_pos_encoding = tf.keras.Input(
      shape=(None, d_model), name="enc_pos_encoding")
  dec_pos_encoding = tf.keras.Input(
      shape=(None, d_model), name="dec_pos_encoding")

  enc_outputs = encoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[inputs, enc_padding_mask, enc_pos_encoding])

  dec_outputs = decoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[
      dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask,
      dec_pos_encoding
  ])

  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(
      inputs=[
          inputs, dec_inputs, enc_padding_mask, look_ahead_mask,
          dec_padding_mask, enc_pos_encoding, dec_pos_encoding
      ],
      outputs=outputs,
      name=name)


sample_transformer = transformer(
    vocab_size=8192,
    num_layers=4,
    units=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name="sample_transformer")

tf.keras.utils.plot_model(
    sample_transformer, to_file='transformer.png', show_shapes=True)

tf.keras.backend.clear_session()

# Hyper-parameters
NUM_LAYERS = 4
D_MODEL = 128
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1
EPOCHS = 20

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
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


def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)


model.compile(optimizer=optimizer, loss=loss_function)

# use teacher forcing, decoder use the previous target as input
decoder_inputs = answers[:, :-1]
# remove START_TOKEN from targets
cropped_targets = answers[:, 1:]
encoder_padding_mask, combined_mask, decoder_padding_mask = create_masks(
    questions, decoder_inputs)
encoding = positional_encoding(position=VOCAB_SIZE, d_model=D_MODEL)
encoder_positional_encoding = encoding[:, :tf.shape(questions)[1], :]
decoder_positional_encoding = encoding[:, :tf.shape(decoder_inputs)[1], :]

# Calculate the number of step per epoch to crop off batch that has size
# smaller than BATCH_SIZE
steps_per_epoch = len(questions) // BATCH_SIZE

model.fit(
    x=[
        questions, decoder_inputs, encoder_padding_mask, combined_mask,
        decoder_padding_mask, encoder_positional_encoding,
        decoder_positional_encoding
    ],
    y=cropped_targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch)


def evaluate(sentence):
  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  sentence = tf.keras.preprocessing.sequence.pad_sequences(
      sentence, maxlen=MAX_LENGTH, padding='post')

  output = tf.expand_dims(START_TOKEN, 0)
  output = tf.keras.preprocessing.sequence.pad_sequences(
      output, maxlen=MAX_LENGTH - 1, padding='post')

  encoding = positional_encoding(position=VOCAB_SIZE, d_model=D_MODEL)

  for i in range(MAX_LENGTH):
    encoder_pos_encoding = encoding[:, :tf.shape(sentence)[1], :]
    decoder_pos_encoding = encoding[:, :tf.shape(output)[1], :]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        sentence, output)

    predictions = model(
        inputs=[
            sentence, output, enc_padding_mask, combined_mask, dec_padding_mask,
            encoder_pos_encoding, decoder_pos_encoding
        ],
        training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, tokenizer.vocab_size + 1):
      return tf.squeeze(output, axis=0)

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def predict(sentence):
  prediction = evaluate(sentence)

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
