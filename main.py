import os
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split

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
  sentence = sentence.rstrip().strip()
  # adding a start and an end token to the sentence
  return '<start> ' + sentence + ' <end>'


def create_dataset():
  # dictionary of line id to text
  id2line = {}
  with open(path_to_movie_lines, errors='ignore') as file:
    for line in file.read().splitlines():
      parts = line.split(' +++$+++ ')
      id2line[parts[0]] = parts[4]

  inputs, targets = [], []
  with open(path_to_movie_conversations, 'r') as file:
    for line in file.read().splitlines():
      parts = line.split(' +++$+++ ')
      # get conversation in a list of line ID
      conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
      for i in range(len(conversation) - 1):
        inputs.append(preprocess_sentence(id2line[conversation[i]]))
        targets.append(preprocess_sentence(id2line[conversation[i + 1]]))

  return inputs, targets


def tokenize(sentences):
  tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  tokenizer.fit_on_texts(sentences)
  tensor = tokenizer.texts_to_sequences(sentences)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
  return tensor, tokenizer


def load_dataset():
  # creating cleaned input, output pairs
  inputs, targets = create_dataset()
  input_tensor, input_tokenizer = tokenize(inputs)
  target_tensor, target_tokenizer = tokenize(targets)
  return input_tensor, target_tensor, input_tokenizer, target_tokenizer


# load and tokenize dataset
input_tensor, target_tensor, input_tokenizer, target_tokenizer = load_dataset()

# get max_length of the inputs and target tensors
max_input_len, max_target_len = input_tensor.shape[-1], target_tensor.shape[-1]

# split into train and evaluation sets using 80-20 split
input_train, input_eval, target_train, target_eval = train_test_split(
    input_tensor, target_tensor, test_size=0.2, shuffle=True)

print('train set size: %d\ntest set size: %d' % (len(input_train),
                                                 len(input_eval)))


def convert(tokenizer, tensor):
  for t in tensor:
    if t != 0:
      print("%d ----> %s" % (t, tokenizer.index_word[t]))


print("Input; index to word mapping")
convert(input_tokenizer, input_train[0])
print()
print("Target; index to word mapping")
convert(target_tokenizer, target_train[0])

BUFFER_SIZE = len(input_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(input_tokenizer.word_index) + 1
vocab_tar_size = len(target_tokenizer.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices(
    (input_train, target_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)
