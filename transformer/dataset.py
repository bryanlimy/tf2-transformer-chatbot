import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence


def load_conversations(hparams, lines_filename, conversations_filename):
    # dictionary of line id to text
    id2line = {}
    with open(lines_filename, errors="ignore") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        id2line[parts[0]] = parts[4]

    questions, answers = [], []
    with open(conversations_filename, "r") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(", ")]
        for i in range(len(conversation) - 1):
            questions.append(preprocess_sentence(id2line[conversation[i]]))
            answers.append(preprocess_sentence(id2line[conversation[i + 1]]))
            if len(questions) >= hparams.max_samples:
                return questions, answers
    return questions, answers


def tokenize_and_filter(hparams, tokenizer, questions, answers):
    tokenized_questions, tokenized_answers = [], []

    for (question, answer) in zip(questions, answers):
        # tokenize sentence
        sentence1 = hparams.start_token + tokenizer.encode(question) + hparams.end_token
        sentence2 = hparams.start_token + tokenizer.encode(answer) + hparams.end_token

        # check tokenize sentence length
        if (
            len(sentence1) <= hparams.max_length
            and len(sentence2) <= hparams.max_length
        ):
            tokenized_questions.append(sentence1)
            tokenized_answers.append(sentence2)

    # pad tokenized sentences
    tokenized_questions = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_questions, maxlen=hparams.max_length, padding="post"
    )
    tokenized_answers = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_answers, maxlen=hparams.max_length, padding="post"
    )

    return tokenized_questions, tokenized_answers


def get_dataset(hparams):
    # download corpus
    path_to_zip = tf.keras.utils.get_file(
        "cornell_movie_dialogs.zip",
        origin="http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
        extract=True,
    )

    path_to_dataset = os.path.join(
        os.path.dirname(path_to_zip), "cornell movie-dialogs corpus"
    )

    # get movie_lines.txt and movive_conversations.txt
    lines_filename = os.path.join(path_to_dataset, "movie_lines.txt")
    conversations_filename = os.path.join(path_to_dataset, "movie_conversations.txt")

    questions, answers = load_conversations(
        hparams, lines_filename, conversations_filename
    )

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=2**13
    )

    hparams.start_token = [tokenizer.vocab_size]
    hparams.end_token = [tokenizer.vocab_size + 1]
    hparams.vocab_size = tokenizer.vocab_size + 2

    questions, answers = tokenize_and_filter(hparams, tokenizer, questions, answers)

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"inputs": questions, "dec_inputs": answers[:, :-1]}, answers[:, 1:])
    )
    dataset = dataset.cache()
    dataset = dataset.shuffle(len(questions))
    dataset = dataset.batch(hparams.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, tokenizer
