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
from keras import backend as K
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils.data_utils import get_file
from keras.models import Model
import numpy as np
import random
import sys
import re

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
    text = text.replace(". ","._<stop> ")
    text = text.replace("? ","?_<stop> ")
    text = text.replace("! ","!_<stop> ")
    text = text.replace("<prd> ",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    out = []
    for s in sentences:
        if (len(s) > 30) and (len(s) < 500):
            out.append(s)
    return out

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower().replace('\n', ' ').replace('=', ' ').replace(r"\\'", " ")
print('corpus length:', len(text))

# nltk.download()
#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#tokenized = tokenizer.tokenize(text)

def text_generator():
    sentences = np.array(split_into_sentences(text))
    sentences = sorted(sentences, key=len)
    chars = sorted(list(set("".join(sentences))))

    EXAMPLES_PER_SENTENCE = 1
    SENTENCE_BATCH_SIZE = 8
    batches = []
    for i in range(0, len(sentences), SENTENCE_BATCH_SIZE): # len(sentences)
        print('Preparing batch: ', i)
        sentence_batch = sentences[i:i + SENTENCE_BATCH_SIZE]
        maxlen_batch = len(max(sentence_batch, key=len))

        X = np.zeros((EXAMPLES_PER_SENTENCE*SENTENCE_BATCH_SIZE, maxlen_batch, len(chars)), dtype=np.int32)
        x = np.zeros((EXAMPLES_PER_SENTENCE*SENTENCE_BATCH_SIZE, maxlen_batch, len(chars)), dtype=np.int32)
        y = np.zeros((EXAMPLES_PER_SENTENCE*SENTENCE_BATCH_SIZE, len(chars)), dtype=np.int32)

        for i, sentence in enumerate(sentence_batch):

            for t, char in enumerate(sentence):
                X[EXAMPLES_PER_SENTENCE*i:EXAMPLES_PER_SENTENCE*(i+1), t, char_indices[char]] = 1

            example_positions = np.random.random_integers(0, len(sentence)-2, EXAMPLES_PER_SENTENCE)
            for t in range(EXAMPLES_PER_SENTENCE):
                taget_pos = example_positions[t]+1
                x[EXAMPLES_PER_SENTENCE*i+t, 0:taget_pos, :] = X[EXAMPLES_PER_SENTENCE*i, 0:taget_pos, :]
                y[EXAMPLES_PER_SENTENCE*i+t, char_indices[sentence[taget_pos]]] = 1
            # This is to learn predicting the first symbol in the sequence
            x[EXAMPLES_PER_SENTENCE * i, :, :] = 0
            y[EXAMPLES_PER_SENTENCE * i, char_indices[sentence[0]]] = 1

        batches.append(([X, x], y))

    cum_count=0
    while 1:
        count=0
        cum_count += 1
        for batch in batches:
            print('batch number: ', count, ', cumulative batch number: ', cum_count)
            count += 1
            yield batch

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

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

LSTM_WIDTH = 256

def concat_context(inputs):
    seq = inputs[0]
    c = inputs[1]
    c_tiled = K.tile(K.reshape(c, [-1, 1, LSTM_WIDTH]), (1,K.shape(seq)[1],1) )
    out = K.concatenate([seq, c_tiled], axis=2)
    return out


context_input = Input(shape=(None, len(chars)))
x = Masking(mask_value=0)(context_input)
x = LSTM(LSTM_WIDTH, return_sequences=True, dropout_W=0.2)(x)
x = LSTM(LSTM_WIDTH, return_sequences=True, dropout_W=0.2)(x)
encoder_output = LSTM(LSTM_WIDTH, return_sequences=True, dropout_W=0.2)(x)

teacher_input = Input(shape=(None, len(chars)))
decoder_input = Masking(mask_value=0)(teacher_input)

context_layer = Lambda(concat_context)
context_layer.build((-1,-1,-1))
C = context_layer([decoder_input, encoder_output[:,-1,:]])
decoder_input_c = merge([decoder_input, C], mode='concat', concat_axis=2)
#a = K.concatenate([decoder_input, K.reshape(encoder_output[:,-1,:], (-1, -1, LSTM_WIDTH))], axis=2)
#decoder_input = merge([teacher_input, encoder_output], mode='concat', concat_axis=1)

y = GRU(LSTM_WIDTH, return_sequences=True, go_backwards=True, dropout_W=0.2, )(decoder_input_c)
y = GRU(LSTM_WIDTH, return_sequences=True, dropout_W=0.2)(y)
y = GRU(LSTM_WIDTH, return_sequences=False, dropout_W=0.2)(y)
decoder_output = Dense(len(chars), activation='softmax')(y)

model = Model(input=[context_input, teacher_input], output=[decoder_output])

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

    model.fit_generator(baseline_gen, samples_per_epoch=2700*10, nb_epoch=1, verbose=1, nb_worker=1)



#    test_sentence = 'eventually--the memory yields._'
#    tes_frag = 'eventually--the m'
#    prediction = model.predict_on_batch(test_set[0])


    # 'eventually--the memory yields._', 'whom i thank when in my bliss?_', 'or even wagner\'s "tannhauser"!_', 'the eternal, fatal "too late"!_', '_must_ we not be dupers also"?_', '17   metaphysical explanation._', 'this is the age of comparison!_', 'history of the moral feelings._', '43   inhuman men as survivals._', '65   whither honesty may lead._'


#    start_index = random.randint(0, len(text) - maxlen - 1)

#    if (iteration % 10 == 0):
#        for diversity in [0.2, 0.5, 1.0, 1.2]:
#            print()
#            print('----- diversity:', diversity)#

#            generated = ''
#            sentence = text[start_index: start_index + maxlen]
#            generated += sentence
#            print('----- Generating with seed: "' + sentence + '"')
#            #sys.stdout.write(generated)
#
#            for i in range(400):
#                x = np.zeros((1, maxlen, len(chars)))
#                for t, char in enumerate(sentence):
#                    x[0, t, char_indices[char]] = 1.
#
#                preds = model.predict(x, verbose=0)[0]
#                next_index = sample(preds, diversity)
#                next_char = indices_char[next_index]
#
#                generated += next_char
#                sentence = sentence[1:] + next_char#
#
#            #    sys.stdout.write(next_char)
#            #    sys.stdout.flush()
#            print(generated)


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
    print('Sentence: ', sentence_decode)
    print('Frag: ', frag_decode, ' Predicted symbol: \"%s\"' %(predicted_symbol) )