import os
import tensorflow as tf

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
SEPARATOR = ' +++$+++ '


def get_conversation_pairs():
  # dictionary of line id to text
  id2line = {}
  with open(path_to_movie_lines, encoding='utf-8', errors='ignore') as file:
    for line in file.read().splitlines():
      parts = line.split(SEPARATOR)
      id2line[parts[0]] = parts[4]

  questions, answers = [], []
  with open(path_to_movie_conversations, 'r') as file:
    for line in file.read().splitlines():
      parts = line.split(SEPARATOR)
      if len(parts) == 4:
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conversation) - 1):
          questions.append(id2line[conversation[i]])
          answers.append(id2line[conversation[i + 1]])
      else:
        print(parts)

  return questions, answers


questions, answers = get_conversation_pairs()
print('Question: %s\nAnswer: %s' % (questions[0], answers[0]))
