import os
import re
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

START_TAG, END_TAG = '<start>', '<end>'

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


def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
  sentence = sentence.strip()
  # adding a start and an end token to the sentence
  return START_TAG + ' ' + sentence + ' ' + END_TAG


def load_conversations(num_samples=None):
  # dictionary of line id to text
  id2line = {}
  with open(path_to_movie_lines, errors='ignore') as file:
    for line in file:
      parts = line.replace('\n', '').split(' +++$+++ ')
      id2line[parts[0]] = parts[4]

  inputs, targets = [], []
  count = 0
  with open(path_to_movie_conversations, 'r') as file:
    for line in file:
      parts = line.replace('\n', '').split(' +++$+++ ')
      # get conversation in a list of line ID
      conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
      for i in range(len(conversation) - 1):
        inputs.append(preprocess_sentence(id2line[conversation[i]]))
        targets.append(preprocess_sentence(id2line[conversation[i + 1]]))
        count += 1
        if num_samples is not None and count >= num_samples:
          return inputs, targets
  return inputs, targets


def tokenize(sentences):
  tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  tokenizer.fit_on_texts(sentences)
  tensor = tokenizer.texts_to_sequences(sentences)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
  return tensor, tokenizer


def load_dataset(num_samples=None):
  # creating cleaned input, output pairs
  inputs, targets = load_conversations(num_samples)
  print('Sample input: {}'.format(inputs[10]))
  print('Sample output: {}'.format(targets[10]))
  input_tensor, input_tokenizer = tokenize(inputs)
  target_tensor, target_tokenizer = tokenize(targets)
  return input_tensor, target_tensor, input_tokenizer, target_tokenizer


NUM_SAMPLES = 100000

# load and tokenize dataset
input_tensor, target_tensor, input_tokenizer, target_tokenizer = load_dataset(
    num_samples=NUM_SAMPLES)

# split into train and evaluation sets using 80-20 split
input_train, input_eval, target_train, target_eval = train_test_split(
    input_tensor, target_tensor, test_size=0.2, shuffle=True)

print('Train set size: {}'.format(len(input_train)))
print('Evaluation set size: {}'.format(len(input_eval)))

BUFFER_SIZE = 1024
BATCH_SIZE = 32
embedding_dim = 256
units = 1024
max_input_len = input_tensor.shape[-1]
max_target_len = target_tensor.shape[-1]
input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

print("Input vocab size: {}".format(input_vocab_size))
print("Target vocab size: {}".format(target_vocab_size))

# train dataset
train_ds = tf.data.Dataset.from_tensor_slices((input_train, target_train))
train_ds = train_ds.shuffle(buffer_size=BUFFER_SIZE)
train_ds = train_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)

# eval dataset
eval_ds = tf.data.Dataset.from_tensor_slices((input_eval, target_eval))
eval_ds = eval_ds.shuffle(buffer_size=BUFFER_SIZE)
eval_ds = eval_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)

print("Train set: {}".format(train_ds))
print("Evaluation set: {}".format(eval_ds))


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


def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions so that we can add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(
      scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(
        q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(
        k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(
        v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(
        scaled_attention, perm=[0, 2, 1,
                                3])  # (batch_size, seq_len_v, num_heads, depth)

    concat_attention = tf.reshape(
        scaled_attention,
        (batch_size, -1, self.d_model))  # (batch_size, seq_len_v, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff,
                            activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.experimental.LayerNormalization(
        epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.experimental.LayerNormalization(
        epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

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


class DecoderLayer(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.experimental.LayerNormalization(
        epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.experimental.LayerNormalization(
        epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.experimental.LayerNormalization(
        epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    # (batch_size, target_seq_len, d_model)
    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    # (batch_size, target_seq_len, d_model)
    attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1,
                                           padding_mask)
    attn2 = self.dropout2(attn2, training=training)
    # (batch_size, target_seq_len, d_model)
    out2 = self.layernorm2(attn2 + out1)

    # (batch_size, target_seq_len, d_model)
    ffn_output = self.ffn(out2)
    ffn_output = self.dropout3(ffn_output, training=training)
    # (batch_size, target_seq_len, d_model)
    out3 = self.layernorm3(ffn_output + out2)

    return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               input_vocab_size,
               rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

    self.enc_layers = [
        EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
    ]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):
    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    # (batch_size, input_seq_len, d_model)
    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    # (batch_size, input_seq_len, d_model)
    return x


class Decoder(tf.keras.layers.Layer):

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               target_vocab_size,
               rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(target_vocab_size, self.d_model)

    self.dec_layers = [
        DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
    ]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    seq_len = tf.shape(x)[1]
    attention_weights = {}

    # (batch_size, target_seq_len, d_model)
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
               dff,
               input_vocab_size,
               target_vocab_size,
               rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                           input_vocab_size, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                           target_vocab_size, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

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


EPOCHS = 5
MAX_LENGTH = 40
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1


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


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size,
                          target_vocab_size, dropout_rate)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')


@tf.function
def train_step(inputs, targets):
  # decoder input begins with START_TAG
  decoder_inputs = targets[:, :-1]
  # target outputs begins after START_TAG
  targets = targets[:, 1:]

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
      inputs, decoder_inputs)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(inputs, decoder_inputs, True, enc_padding_mask,
                                 combined_mask, dec_padding_mask)
    loss = loss_function(targets, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  return loss, predictions


@tf.function
def eval_step(inputs, targets):
  decoder_inputs = targets[:, :-1]
  targets = targets[:, 1:]

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
      inputs, decoder_inputs)

  predictions, _ = transformer(inputs, decoder_inputs, True, enc_padding_mask,
                               combined_mask, dec_padding_mask)
  loss = loss_function(targets, predictions)

  return loss, predictions


def train_and_evaluate(epochs):
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')

  eval_loss = tf.keras.metrics.Mean(name='eval_loss')
  eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='eval_accuracy')

  for epoch in range(epochs):

    # reset metrics
    train_loss.reset_states()
    train_accuracy.reset_states()
    eval_loss.reset_states()
    eval_accuracy.reset_states()

    start = time.time()

    # train
    for (batch, (inputs, targets)) in enumerate(train_ds):
      loss, predictions = train_step(inputs, targets)

      train_loss(loss)
      train_accuracy(targets[:, 1:], predictions)

      if batch % 200 == 0:
        print('Target: {}'.format(targets[0]))
        print('Predictions: {}'.format(predictions[0]))
        print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.2f}'.format(
            epoch + 1, batch, train_loss.result(),
            train_accuracy.result() * 100))

    end = time.time()

    # evaluate
    for (batch, (inputs, targets)) in enumerate(eval_ds):
      loss, predictions = eval_step(inputs, targets)

      eval_loss(loss)
      eval_accuracy(targets[:, 1:], predictions)

    print('Epoch {} Loss {:.4f} Accuracy {:.2f} Eval Loss {:.4f} '
          'Eval Accuracy {:.2f} Time {:.2f}s\n'.format(
              epoch + 1,
              train_loss.result(),
              train_accuracy.result() * 100,
              eval_loss.result(),
              eval_accuracy.result() * 100,
              end - start,
          ))

    # save checkpoint every 2 epochs
    if epoch % 2 == 0:
      ckpt_save_path = ckpt_manager.save()
      print('Saving checkpoint for epoch {} at {}'.format(
          epoch + 1, ckpt_save_path))


train_and_evaluate(epochs=EPOCHS)


def evaluate(sentence):
  sentence = preprocess_sentence(sentence)

  print('Preprocessed sentence: {}'.format(sentence))

  # tokenize sentence
  sentence = [
      input_tokenizer.word_index[t]
      for t in sentence.split(' ')
      if t in input_tokenizer.word_index
  ]

  print('Tokenized sentence: {}'.format(sentence))

  # pad tokens to fixed length
  encoder_input = tf.keras.preprocessing.sequence.pad_sequences(
      [sentence], maxlen=max_input_len, padding='post')

  decoder_input = [target_tokenizer.word_index[START_TAG]]

  output = tf.expand_dims(decoder_input, axis=0)

  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(
        encoder_input,
        output,
        False,
        enc_padding_mask,
        combined_mask,
        dec_padding_mask,
    )

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, target_tokenizer.word_index[END_TAG]):
      return tf.squeeze(output, axis=0), attention_weights

    # concatenate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights


def translate(sentence):
  result, attention_weights = evaluate(sentence)

  print('Tokenized output: {}'.format(result))
  # convert from tokens to words
  predicted_sentence = [
      target_tokenizer.index_word[int(t)]
      for t in result
      if int(t) in target_tokenizer.index_word
  ]

  print('Input: {}'.format(sentence))
  print('Output: {}'.format(' '.join(predicted_sentence)))


translate('This is my head')
