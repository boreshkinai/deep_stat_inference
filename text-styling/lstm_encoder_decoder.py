'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, Input, Flatten, Masking, merge, Reshape, Lambda
from keras.layers.merge import Concatenate, Add
from keras import backend as K
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils.data_utils import get_file
from keras.models import Model
import numpy as np
import random
import sys
import re

EXAMPLES_PER_SENTENCE = 2
SENTENCE_BATCH_SIZE = 8
LSTM_WIDTH = 256
SENTENCE_START = '#'
SENTENCE_END = '_'


#import nltk.data

caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    #if "\'" in text: text = text.replace("\'", " ")
    text = text.replace(". ","."+SENTENCE_END+"<stop> ")
    text = text.replace("? ","?"+SENTENCE_END+"<stop> ")
    text = text.replace("! ","!"+SENTENCE_END+"<stop> ")
    text = text.replace("<prd> ",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    out = []
    for s in sentences:
        if (len(s) > 30) and (len(s) < 500):
            out.append(SENTENCE_START+s)
    return out

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read()

#text = "The basic idea—that software can simulate the neocortex’s large array of neurons in an artificial “neural network”—is decades old, and it has led to as many disappointments as breakthroughs. But because of improvements in mathematical formulas and increasingly powerful computers, computer scientists can now model many more layers of virtual neurons than ever before. With this greater depth, they are producing remarkable advances in speech and image recognition. Last June, a Google deep-learning system that had been shown 10 million images from YouTube videos proved almost twice as good as any previous image recognition effort at identifying objects such as cats. Google also used the technology to cut the error rate on speech recognition in its latest Android mobile software. In October, Microsoft chief research officer Rick Rashid wowed attendees at a lecture in China with a demonstration of speech software that transcribed his spoken words into English text with an error rate of 7 percent, translated them into Chinese-language text, and then simulated his own voice uttering them in Mandarin. That same month, a team of three graduate students and two professors won a contest held by Merck to identify molecules that could lead to new drugs. The group used deep learning to zero in on the molecules most likely to bind to their targets. Google in particular has become a magnet for deep learning and related AI talent. In March the company bought a startup cofounded by Geoffrey Hinton, a University of Toronto computer science professor who was part of the team that won the Merck contest. Hinton, who will split his time between the university and Google, says he plans to “take ideas out of this field and apply them to real problems” such as image recognition, search, and natural-language understanding, he says."
text = text.lower().replace('\n', ' ').replace('=', ' ').replace(r"\\'", " ")
print('corpus length:', len(text))

# nltk.download()
#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#tokenized = tokenizer.tokenize(text)

sentences = np.array(split_into_sentences(text))
sentences = sorted(sentences, key=len)
chars = sorted(list(set("".join(sentences))))

#chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def text_generator():



    batches = []
    for i in range(0, len(sentences), SENTENCE_BATCH_SIZE): # len(sentences)
        # print('Preparing batch: ', i)
        sentence_batch = sentences[i:i + SENTENCE_BATCH_SIZE]
        maxlen_batch = len(max(sentence_batch, key=len))

        X = np.zeros((EXAMPLES_PER_SENTENCE*SENTENCE_BATCH_SIZE, maxlen_batch, len(chars)), dtype=np.int32)
        x = np.zeros((EXAMPLES_PER_SENTENCE*SENTENCE_BATCH_SIZE, maxlen_batch, len(chars)), dtype=np.int32)
        y = np.zeros((EXAMPLES_PER_SENTENCE*SENTENCE_BATCH_SIZE, len(chars)), dtype=np.int32)

        for i, sentence in enumerate(sentence_batch):

            for t, char in enumerate(sentence):
                X[EXAMPLES_PER_SENTENCE*i:EXAMPLES_PER_SENTENCE*(i+1), t, char_indices[char]] = 1

            example_positions = np.random.randint(0, len(sentence)-2, EXAMPLES_PER_SENTENCE)
            for t in range(EXAMPLES_PER_SENTENCE):
                taget_pos = example_positions[t]+1
                x[EXAMPLES_PER_SENTENCE*i+t, 0:taget_pos, :] = X[EXAMPLES_PER_SENTENCE*i, 0:taget_pos, :]
                y[EXAMPLES_PER_SENTENCE*i+t, char_indices[sentence[taget_pos]]] = 1
            # This is to learn predicting the first symbol in the sequence
            x[EXAMPLES_PER_SENTENCE * i, :, :] = 0
            x[EXAMPLES_PER_SENTENCE * i, 0, char_indices[sentence[0]]] = 1
            y[EXAMPLES_PER_SENTENCE * i, :] = 0
            y[EXAMPLES_PER_SENTENCE * i, char_indices[sentence[1]]] = 1
            # This is to learn predicting the last symbol in the sequence
            x[EXAMPLES_PER_SENTENCE * i + EXAMPLES_PER_SENTENCE-1, :, :] = 0
            x[EXAMPLES_PER_SENTENCE * i + EXAMPLES_PER_SENTENCE-1, 0:len(sentence)-1, :] = X[EXAMPLES_PER_SENTENCE*i, 0:len(sentence)-1, :]
            y[EXAMPLES_PER_SENTENCE * i + EXAMPLES_PER_SENTENCE-1, :] = 0
            y[EXAMPLES_PER_SENTENCE * i + EXAMPLES_PER_SENTENCE-1, char_indices[sentence[-1]]] = 1

        batches.append(([X, x], y))

    cum_count=0
    while 1:
        count=0
        cum_count += 1
        for batch in batches:
            print('batch number: ', count, ', cumulative batch number: ', cum_count)
            count += 1
            yield batch



## cut the text in semi-redundant sequences of maxlen characters
#maxlen = 40
#step = 3
#sentences = []
#next_chars = []
#for i in range(0, len(text) - maxlen, step):
#    sentences.append(text[i: i + maxlen])
#    next_chars.append(text[i + maxlen])
#print('nb sequences:', len(sentences))

#print('Vectorization...')
#X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
#y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
#for i, sentence in enumerate(sentences):
#    for t, char in enumerate(sentence):
#        X[i, t, char_indices[char]] = 1
#    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
#model = Sequential()
#model.add(LSTM(128, input_shape=(maxlen, len(chars)))) # return_sequences
#model.add(Dense(len(chars)))
#model.add(Activation('softmax'))



def concat_context(inputs):
    seq = inputs[0]
    c = inputs[1]
    c_tiled = K.tile(K.reshape(c, [-1, 1, LSTM_WIDTH]), (1,K.shape(seq)[1],1) )
    out = K.concatenate([seq, c_tiled], axis=2)

    boolean_mask = K.any(K.not_equal(seq, 0), axis=-1, keepdims=True)

    return out * K.cast(boolean_mask, K.floatx())


#def get_output_shape_for_concat_context(input_shape):
#    return (None, None, input_shape[0][2] + input_shape[1][1])

context_input = Input(shape=(None, len(chars)))
x = Masking(mask_value=0)(context_input)
x = LSTM(LSTM_WIDTH, return_sequences=True, go_backwards=True, dropout=0.0)(x)
x = LSTM(LSTM_WIDTH, return_sequences=True, dropout=0.0)(x)
encoder_output = LSTM(LSTM_WIDTH, return_sequences=False, dropout=0.0)(x)

teacher_input = Input(shape=(None, len(chars)))
decoder_input = Masking(mask_value=0)(teacher_input)

context_layer = Lambda(concat_context) # , output_shape=get_output_shape_for_concat_context
decoder_input_c = context_layer([decoder_input, encoder_output])

#encoder_output = Reshape((1, LSTM_WIDTH))(encoder_output)
#decoder_input_c = Concatenate(axis=2)([decoder_input, encoder_output])

y1 = LSTM(LSTM_WIDTH, return_sequences=True, dropout=0.0, )(decoder_input_c)
y2 = LSTM(LSTM_WIDTH, return_sequences=True, dropout=0.0)(y1)
#y2 = Add()([y1, y2])
y3 = LSTM(LSTM_WIDTH, return_sequences=False, dropout=0.0)(y2)
#y3 = Add()([y2[:,-1,:], y3])

#y = Dense(LSTM_WIDTH, activation='relu')(encoder_output)
#y = Dense(LSTM_WIDTH, activation='relu')(y)
decoder_appended = Concatenate()([encoder_output, y3])
decoder_appended = Dense(LSTM_WIDTH, activation='relu')(decoder_appended)

decoder_output = Dense(len(chars), activation='softmax')(decoder_appended)

model = Model(inputs=[context_input, teacher_input], outputs=[decoder_output])

optimizer = Adam(clipnorm=5.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration

baseline_gen = text_generator()

for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    #model.fit(X, y,
    #          batch_size=512,
    #          nb_epoch=1)

    model.fit_generator(baseline_gen, steps_per_epoch=2700/SENTENCE_BATCH_SIZE, epochs=1, verbose=1, workers=1)

# Some simple test of model prediction performance
test_set = baseline_gen.__next__()
prediction = model.predict_on_batch(test_set[0])
for i in range(len(test_set[1])):

    sentence = np.argmax(test_set[0][0][i], axis=1)
    sentence_decode = ''
    for t in range(len(sentence)):
        sentence_decode += indices_char[sentence[t]]

    frag = np.argmax(test_set[0][1][i], axis=1)
    frag_decode = ''
    for t in range(len(frag)):
        frag_decode += indices_char[frag[t]]

    predicted_symbol = indices_char[np.argmax(prediction[i])]
    true_symbol = indices_char[np.argmax(test_set[1][i])]
    print('Sentence: ', sentence_decode)
    print('Frag: ', frag_decode, ' Predicted symbol: \"%s\"' % (predicted_symbol),
          ' True symbol: \"%s\"' % (true_symbol))



# Some simple test of model prediction performance
test_set = baseline_gen.__next__()
prediction = model.predict_on_batch(test_set[0])
for i in range(len(test_set[1])):

    sentence = np.argmax(test_set[0][0][i], axis=1)
    sentence_decode = ''
    for t in range(len(sentence)):
        sentence_decode += indices_char[sentence[t]]

    frag = np.argmax(test_set[0][1][i], axis=1)
    frag_decode = ''
    for t in range(len(frag)):
        frag_decode += indices_char[frag[t]]

    predicted_symbol = indices_char[np.argmax(prediction[i])]
    true_symbol = indices_char[np.argmax(test_set[1][i])]
    print('Sentence: ', sentence_decode)
    print('Frag: ', frag_decode, ' Predicted symbol: \"%s\"' % (predicted_symbol),
          ' True symbol: \"%s\"' % (true_symbol))

