import argparse
import tensorflow as tf

tf.random.set_seed(1234)

from transformer.model import transformer
from transformer.dataset import get_dataset, preprocess_sentence


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, hparams, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = tf.cast(hparams.d_model, dtype=tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * self.warmup_steps**-1.5
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def inference(hparams, model, tokenizer, sentence):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      hparams.start_token + tokenizer.encode(sentence) + hparams.end_token,
      axis=0)

  output = tf.expand_dims(hparams.start_token, 0)

  for i in range(hparams.max_length):
    predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, hparams.end_token[0]):
      break

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def predict(hparams, model, tokenizer, sentence):
  prediction = inference(hparams, model, tokenizer, sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  return predicted_sentence


def evaluate(hparams, model, tokenizer):
  print('\nEvaluate')
  sentence = 'where have you been?'
  output = predict(hparams, model, tokenizer, sentence)
  print('input: {}\noutput: {}'.format(sentence, output))

  sentence = "it's a trap!"
  output = predict(hparams, model, tokenizer, sentence)
  print('\ninput: {}\noutput: {}'.format(sentence, output))

  sentence = 'I am not crazy, my mother had me tested'
  for _ in range(5):
    output = predict(hparams, model, tokenizer, sentence)
    print('\ninput: {}\noutput: {}'.format(sentence, output))
    sentence = output


def main(hparams):
  dataset, tokenizer = get_dataset(hparams)

  model = transformer(hparams)

  optimizer = tf.keras.optimizers.Adam(
      CustomSchedule(hparams), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

  cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')

  def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, hparams.max_length - 1))
    loss = cross_entropy(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)

  def accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, hparams.max_length - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

  model.compile(optimizer, loss=loss_function, metrics=[accuracy])

  model.fit(dataset, epochs=hparams.epochs)

  evaluate(hparams, model, tokenizer)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--max_samples',
      default=25000,
      type=int,
      help='maximum number of conversation pairs to use')
  parser.add_argument(
      '--max_length', default=40, type=int, help='maximum sentence length')
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--num_layers', default=2, type=int)
  parser.add_argument('--num_units', default=512, type=int)
  parser.add_argument('--d_model', default=256, type=int)
  parser.add_argument('--num_heads', default=8, type=int)
  parser.add_argument('--dropout', default=0.1, type=float)
  parser.add_argument('--activation', default='relu', type=str)
  parser.add_argument('--epochs', default=20, type=int)

  hparams = parser.parse_args()
  main(hparams)
