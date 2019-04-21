import os
import re
import time
import pickle
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

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

NUM_SAMPLES = 10000


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

tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)

MAX_LENGTH = 40
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]


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
VOCAB_SIZE = tokenizer.vocab_size + 2

train_ds = tf.data.Dataset.from_tensor_slices((questions, answers))
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(BUFFER_SIZE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)


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
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  return seq[:, tf.newaxis, tf.newaxis, :]


# Look-ahead mask to mask the future tokens in a sequence.
# i.e. To predict the third word, only the first and second word will be used
def create_look_ahead_mask(size):
  return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


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

    self.wq = tf.keras.layers.Dense(units=d_model)
    self.wk = tf.keras.layers.Dense(units=d_model)
    self.wv = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)
    v = self.split_heads(v, batch_size)

    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    output = self.dense(concat_attention)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, units):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(units=units, activation='relu'),
      tf.keras.layers.Dense(units=d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, units, dropout=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, units)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate=dropout)
    self.dropout2 = tf.keras.layers.Dropout(rate=dropout)

  def call(self, x, training, mask):
    # (batch_size, input_seq_len, d_model)
    attn_output, _ = self.mha(x, x, x, mask)
    attn_output = self.dropout1(attn_output, training=training)
    # (batch_size, input_seq_len, d_model)
    out1 = self.layernorm1(x + attn_output)

    # (batch_size, input_seq_len, d_model)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    # (batch_size, input_seq_len, d_model)
    out2 = self.layernorm2(out1 + ffn_output)

    return out2


class Encoder(tf.keras.layers.Layer):

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               units,
               vocab_size,
               dropout=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
    self.pos_encoding = positional_encoding(vocab_size, self.d_model)

    self.enc_layers = [
        EncoderLayer(d_model, num_heads, units, dropout)
        for _ in range(num_layers)
    ]

    self.dropout = tf.keras.layers.Dropout(rate=dropout)

  def call(self, x, training, mask):
    seq_len = tf.shape(x)[1]

    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    return x


class DecoderLayer(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, units, dropout=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, units)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate=dropout)
    self.dropout2 = tf.keras.layers.Dropout(rate=dropout)
    self.dropout3 = tf.keras.layers.Dropout(rate=dropout)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1,
                                           padding_mask)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)

    ffn_output = self.ffn(out2)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)

    return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               units,
               vocab_size,
               dropout=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
    self.pos_encoding = positional_encoding(vocab_size, self.d_model)

    self.dec_layers = [
        DecoderLayer(d_model, num_heads, units, dropout)
        for _ in range(num_layers)
    ]
    self.dropout = tf.keras.layers.Dropout(rate=dropout)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


class Transformer(tf.keras.Model):

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               units,
               vocab_size,
               dropout=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, units, vocab_size,
                           dropout)

    self.decoder = Decoder(num_layers, d_model, num_heads, units, vocab_size,
                           dropout)

    self.final_layer = tf.keras.layers.Dense(units=vocab_size)

  def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask,
           dec_padding_mask):
    # (batch_size, inp_seq_len, d_model)
    enc_output = self.encoder(inp, training, enc_padding_mask)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)

    # (batch_size, tar_seq_len, target_vocab_size)
    final_output = self.final_layer(dec_output)

    return final_output, attention_weights


NUM_LAYERS = 4
D_MODEL = 256
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


# Metrics
train_loss = tf.keras.metrics.Mean(name='loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

CKPT_PATH = "runs/"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, CKPT_PATH, max_to_keep=3)
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Restored checkpoint {}'.format(ckpt_manager.latest_checkpoint))


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


@tf.function
def train_step(inputs, targets):
  # use teacher forcing, decoder use the previous target as input
  decoder_inputs = targets[:, :-1]
  # remove START_TOKEN from targets
  cropped_targets = targets[:, 1:]

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
      questions, decoder_inputs)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(inputs, decoder_inputs, True, enc_padding_mask,
                                 combined_mask, dec_padding_mask)
    loss = loss_function(cropped_targets, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(cropped_targets, predictions)


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
        sentence, output, False, enc_padding_mask, combined_mask,
        dec_padding_mask)

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

predict('How are you?')
print('')

# test the model with its previous output as input
sentence = 'I am not crazy, my mother had me tested.'
for _ in range(5):
  sentence = predict(sentence)
  print('')
