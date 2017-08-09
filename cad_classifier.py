
import numpy
# Set random seed to produce repeatable results
numpy.random.seed(7)

from keras.models import Sequential
import random
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import Counter, defaultdict
from itertools import count
import nltk
import mmap
import os
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from keras.layers import Dropout


class Vocab:  # Storing the vocabulary and word-2-id mappings
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())


def ExtractAlphanumeric(ins):
    from string import ascii_letters, digits, whitespace, punctuation
    return "".join([ch for ch in ins if ch in (ascii_letters + whitespace + punctuation)])


def get_padded_sentences_tokens_list(text, mark=""):
    tokens = []
    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        sent_tokens = nltk.word_tokenize(sent)
        new_tokens = [token + mark for token in sent_tokens]
        tokens += ["<sentence_start>"] + new_tokens + ["<sentence_stop>"]

    return tokens


class CorpusReader:
    def __init__(self, positive_file_path, negative_file_path):
        self.positive_file_path = positive_file_path
        self.negative_file_path = negative_file_path
        self.common_words_filename = "/Users/macbook/Desktop/corpora/MIMIC/MIMIC_50k_vocab.txt"
        self.positive_number = 0
        self.negative_number = 0

    def __iter__(self):  # Yields one instance as a list of words

        word_length = 6
        words_number = 100
        substring_length = word_length * words_number

        vocab_list = []
        with open(self.common_words_filename) as vocab_file:
            for line in vocab_file:
                line = line.lower().strip()
                vocab_list.append(line)

        with open(self.positive_file_path) as pos_file:
            for line in pos_file:
                self.positive_number += 1
                line = line[0:substring_length].lower()
                line = ExtractAlphanumeric(line)
                tokens = get_padded_sentences_tokens_list(line)
                clean_tokens = [token for token in tokens if token in vocab_list]
                tokens_list = ["<start>"] + clean_tokens + ["<stop>"]
                yield tokens_list
        print "Finished positive yield"
        with open(self.negative_file_path) as neg_file:
            for line in neg_file:
                self.negative_number += 1
                line = line[0:substring_length].lower()
                line = ExtractAlphanumeric(line)
                tokens = get_padded_sentences_tokens_list(line)
                clean_tokens = [token for token in tokens if token in vocab_list]
                tokens_list = ["<start>"] + clean_tokens + ["<stop>"]
                yield tokens_list


mimic_cad_positive_summaries_filename = "/Users/macbook/Desktop/corpora/MIMIC/cad_summaries_pos.txt"
mimic_cad_negative_summaries_filename = "/Users/macbook/Desktop/corpora/MIMIC/cad_summaries_neg.txt"

cad_positive_test_filename = "/Users/macbook/Desktop/corpora/MIMIC/test_cad_positive.txt"
cad_negative_test_filename = "/Users/macbook/Desktop/corpora/MIMIC/test_cad_negative.txt"


print "Read Training folder..."
train = CorpusReader(mimic_cad_positive_summaries_filename, mimic_cad_negative_summaries_filename)
print "Read Testing folder..."
test = CorpusReader(cad_positive_test_filename, cad_negative_test_filename)
print "Creating train vocab..."
vocab = Vocab.from_corpus(train)
# Save now the train labels size, before turning to list which yields again the content
positive_train_num = train.positive_number
negative_train_num = train.negative_number

print "Train Vocabulary size:", vocab.size()
print "Vocab content:", vocab.w2i

train_list = list(train)

test_list = list(test)  # This line will yield all the counts from test

print "TEST SIZE:", len(test_list)

positive_test_num = test.positive_number
negative_test_num = test.negative_number

print "Positive test N", positive_test_num
print "Negative test N", negative_test_num




# Creating the train set - labels
Ys = []
for i in range(0, positive_train_num):
    Ys.append(1)
for i in range(0, negative_train_num):
    Ys.append(0)

# We need to shuffle the train with its labels accordingly
c = list(zip(train_list, Ys))
random.shuffle(c)
train_list, arr_Ys = zip(*c) # The results are arrays

# Convert back into list, so we can use it in training
Ys = []
for item in arr_Ys:
    Ys.append(item)

test_Ys = []

for i in range(0, positive_test_num):
    test_Ys.append(1)
for i in range(0, negative_test_num):
    test_Ys.append(0)

# Shuffle the train instances and labels together
c = list(zip(test_list, test_Ys))
random.shuffle(c)
test_list, test_Ys = zip(*c)

int_train = []

for item in train_list:
    int_item = [vocab.w2i[w] for w in item]
    int_train.append(int_item)

int_test = []

for item in test_list:
    int_item = [vocab.w2i[w] for w in item if w in vocab.w2i.keys()]
    int_test.append(int_item)


max_text_length = 100  # words
embedding_vector_length = 200
memory_size = 200  # The size of LSTM memory cell
WORDS_NUM = vocab.size()

X_train = int_train
Y_train = Ys
X_train = sequence.pad_sequences(X_train, maxlen=max_text_length)

X_test = int_test
Y_test = test_Ys
X_test = sequence.pad_sequences(X_test, maxlen=max_text_length)

model = Sequential()
model.add(Embedding(WORDS_NUM, embedding_vector_length, input_length=max_text_length))
model.add(LSTM(memory_size))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print "Fitting the model - Train start"

model.fit(X_train, Y_train, epochs=5, batch_size=100)

predictions = model.predict(X_test)

rounded = []
for prediction_value in predictions:
    if prediction_value > 0.5:
        rounded.append(1)
    else:
        rounded.append(0)

print "Positive Recall:", recall_score(Y_test, rounded, pos_label=1)
print "Negative Recall:", recall_score(Y_test, rounded, pos_label=0)
print "AUC:", roc_auc_score(Y_test, predictions)

