import numpy as np
import codecs
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import TimeDistributedDense, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
import keras
from collections import Counter

from sklearn.preprocessing import LabelBinarizer
import gc
import nltk
import sys
import os
#### READ GLOVE ####

# Read GloVe Embedding
BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/GLOVE/'
#EMBEDDING_DIM = 100 # Embedding dimensions

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split() #Tokenizer
    word = values[0] # word form
    coefs = np.asarray(values[1:], dtype='float32') # embedding coefficients
    embeddings_index[word] = coefs
f.close()

print('Number of embeddings:', len(embeddings_index))
print('Ten first element from `the` Embedding Vector :', embeddings_index["the"][:10])

#### SEQUENCE SEGMENTATION ####
#def chunker(seq, size):
#    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
def chunker(seq, size, is_y=False):
    size = size-1
    if is_y:
        return (["O"] + seq[pos:pos + size] for pos in range(0, len(seq), size))
    else:
        return (seq[pos:pos + size] + ["<EOS>"] for pos in range(0, len(seq), size))

def score(yh, pr):
    coords = [np.where(yhh >= 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result

chunk_size     =   int(sys.argv[1])
vocabulary_size=   int(sys.argv[2])
embedding_size =   int(sys.argv[3])
hidden_size    =   int(sys.argv[4])
batch_size     =   int(sys.argv[5])
no_question    =   int(sys.argv[6])
optimizer_     =   sys.argv[7]
patience_      =   int(sys.argv[8])

#chunk_size     =   200
#vocabulary_size=   20000
#embedding_size =   100
#hidden_size    =   256
#batch_size     =   200
#no_question    =   0
#optimizer		= 	"adam"
#patience_		=	2

# In[4]:
#### READ TRAIN ####
if no_question==1:
	raw = codecs.open('train2012_nq', 'r', "ISO-8859-1").readlines()
else:
	raw = codecs.open('train2012', 'r', "ISO-8859-1").readlines()
env = []
for line in raw:
    stripped_line = line.split()
    #print (stripped_line)
    env.append(stripped_line)

all_x=[]

for i in(range(len(env))):
    if (len(env[i])==2):
        all_x.append(env[i])

X = [x[0] for x in all_x]
y = [x[1] for x in all_x]

all_text = [c for x in X for c in x]

x_chunks=[]
y_chunks=[]
for group in chunker(X, chunk_size):
    x_chunks.append(group)
for group in chunker(y, chunk_size, is_y=True):
    y_chunks.append(group)

### CREATE VOCABULARY ###

# Count the word frequencies
word_freq = nltk.FreqDist(X)
print ("Found %d unique words tokens." % len(word_freq.items()))

words = word_freq.most_common(vocabulary_size)


vocab=["<PAD>"]+[x[0] for x in words]
vocab.append("UNKNOWN")
vocab.append("<EOS>")

### TRANSFORM TEXTS #
word2ind = {word: index for index, word in enumerate(vocab)}
ind2word = {index: word for index, word in enumerate(word2ind)}



labels = ['O','COMMA','PERIOD']
if no_question==0:
    labels.append('QUESTION')
label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}

print (label2ind)

print ('Vocabulary size:', len(word2ind), len(label2ind))

maxlen = max([len(x) for x in x_chunks])
print ('Maximum sequence length:', maxlen)

unknown_token="UNKNOWN"

for i, sent in enumerate(x_chunks):
    x_chunks[i] = [w if w in word2ind else unknown_token for w in sent]

X_enc = [[word2ind[c] for c in x] for x in x_chunks]
max_label = max(label2ind.values()) + 1
y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y_chunks]


y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]
y_enci =[]
for i in (range(len(y_enc))):
    a=y_enc[i]
    v=np.array(a)
    v=np.delete(v, 0, 1)
    y_enci.append(v)


X_enc = pad_sequences(X_enc, maxlen=maxlen, padding="post", value=word2ind["<PAD>"] )
y_enc = pad_sequences(y_enci, maxlen=maxlen, padding="post")

train_len=int(len(X_enc))

X_train = X_enc[:train_len]
y_train = y_enc[:train_len]



### READ DEV ###
if no_question==1:
	raw = codecs.open('dev2012_nq', 'r', "ISO-8859-1").readlines()
else:
	raw = codecs.open('dev2012', 'r', "ISO-8859-1").readlines()

print (raw[1])
env = []
for line in raw:
    stripped_line = line.split()
    #print (stripped_line)
    env.append(stripped_line)

all_x_dev=[]
for i in(range(len(env))):
    if (len(env[i])==2):
        all_x_dev.append(env[i])
    else:
        print (env[i])

X = [x[0] for x in all_x_dev]
y = [x[1] for x in all_x_dev]

all_text = [c for x in X for c in x]

x_dev=[]
y_dev=[]
for group in chunker(X, chunk_size):
    x_dev.append(group)
for group in chunker(y, chunk_size, is_y=True):
    y_dev.append(group)


print ('Vocabulary size:', len(word2ind), len(label2ind))

maxlen_ = max([len(x) for x in x_dev])
print ('Maximum sequence length:', maxlen)
### TRANSFORM TEXTS #

for i, sent in enumerate(x_dev):
    x_dev[i] = [w if w in word2ind else unknown_token for w in sent]

X_enc = [[word2ind[c] for c in x] for x in x_dev]
max_label = max(label2ind.values())+1
y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y_dev]

y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

y_enci =[]
for i in (range(len(y_enc))):
    a=y_enc[i]
    v=np.array(a)
    v=np.delete(v, 0, 1)
    y_enci.append(v)

print (len(y_enci))
print (y_enci[0].shape)

X_enc = pad_sequences(X_enc, maxlen=maxlen, padding="post", value=word2ind["<PAD>"] )
y_enc = pad_sequences(y_enci, maxlen=maxlen, padding="post")

val_len=int(len(X_enc)*1.0)
X_val = X_enc[:val_len]
y_val = y_enc[:val_len]


### READ TEST ###
if no_question==1:
	raw = codecs.open('test2011_nq', 'r', "ISO-8859-1").readlines()
else:
	raw = codecs.open('test2011', 'r', "ISO-8859-1").readlines()

print (raw[1])
all_x_test = []
for line in raw:
    stripped_line = line.split()
    #print (stripped_line)
    all_x_test.append(stripped_line)

X = [x[0] for x in all_x_test]
y = [x[1] for x in all_x_test]

all_text = [c for x in X for c in x]

x_test=[]
y_test=[]
for group in chunker(X, chunk_size):
    x_test.append(group)
for group in chunker(y, chunk_size, is_y=True):
    y_test.append(group)

### TRANSFORM TEXTS #
for i, sent in enumerate(x_test):
    x_test[i] = [w if w in word2ind else unknown_token for w in sent]

X_enc = [[word2ind[c] for c in x] for x in x_test]

y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y_test]

y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

y_enci =[]
for i in (range(len(y_enc))):
    a=y_enc[i]
    v=np.array(a)
    v=np.delete(v, 0, 1)
    y_enci.append(v)

X_enc = pad_sequences(X_enc, maxlen=maxlen, padding="post", value=word2ind["<PAD>"] )
y_enc = pad_sequences(y_enci, maxlen=maxlen, padding="post")

test_len=int(len(X_enc)*1.0)

X_test = X_enc[:test_len]
y_test = y_enc[:test_len]

### READ TEST ASR###
if no_question==1:
	raw = codecs.open('test2011asr_nq', 'r', "ISO-8859-1").readlines()
else:
	raw = codecs.open('test2011asr', 'r', "ISO-8859-1").readlines()

all_x_test_asr = []
for line in raw:
    stripped_line = line.split()
    all_x_test_asr.append(stripped_line)

X = [x[0] for x in all_x_test_asr]
y = [x[1] for x in all_x_test_asr]

all_text = [c for x in X for c in x]

x_test_asr=[]
y_test_asr=[]
for group in chunker(X, chunk_size):
    x_test_asr.append(group)
for group in chunker(y, chunk_size, is_y=True):
    y_test_asr.append(group)

### TRANSFORM TEXTS #
for i, sent in enumerate(x_test_asr):
    x_test_asr[i] = [w if w in word2ind else unknown_token for w in sent]

X_enc = [[word2ind[c] for c in x] for x in x_test_asr]
max_label = max(label2ind.values())+1
y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y_test_asr]

y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]
y_enci =[]
for i in (range(len(y_enc))):
    a=y_enc[i]
    v=np.array(a)
    v=np.delete(v, 0, 1)
    y_enci.append(v)

X_enc = pad_sequences(X_enc, maxlen=maxlen, padding="post", value=word2ind["<PAD>"] )
y_enc = pad_sequences(y_enci, maxlen=maxlen)

test_asr_len=int(len(X_enc)*1.0)

X_test_asr = X_enc[:test_asr_len]
y_test_asr = y_enc[:test_asr_len]
print (len(X_test_asr))



max_features = len(word2ind)
out_size = len(label2ind)

### CREATE EMBEDDING ###
nb_words = min(vocabulary_size, len(word2ind))
print (nb_words)
print (embedding_size)

embedding_matrix = np.zeros((nb_words + 3, embedding_size))
for word, i in word2ind.items():
    if i > nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: # Initialized embedding_vector
        embedding_matrix[i] = embedding_vector

print ('Number of remaining lines from embedding matrix:', len(embedding_matrix))
### CREATE RNN model with LSTM ###
model = Sequential()
model.add(Embedding(nb_words+3, embedding_size, input_length=maxlen,weights=[embedding_matrix],mask_zero=True))
model.add(LSTM(hidden_size, return_sequences=True,init='glorot_normal'))
model.add(TimeDistributedDense(out_size))
model.add(Activation('softmax'))

earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience_,verbose=0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer_, metrics=['accuracy','precision','recall','fmeasure'])
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=100, validation_data=(X_val, y_val),callbacks=[earlyStopping], shuffle=True)
score_model = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Raw test score:', score_model)


### EVALUATION TRAIN ###
pr = model.predict_classes(X_train)
yh = y_train.argmax(2)
fyh, fpr = score(yh, pr)
print ('Training accuracy:', accuracy_score(fyh, fpr))
print ('Training confusion matrix:')
print (confusion_matrix(fyh, fpr))
print (classification_report(fyh,fpr))

### EVALUATION DEV ###
pr = model.predict_classes(X_val)
yh = y_val.argmax(2)
fyh, fpr = score(yh, pr)
print ('Validation accuracy:', accuracy_score(fyh, fpr))
print ('Validation confusion matrix:')
print (confusion_matrix(fyh, fpr))
print (classification_report(fyh,fpr))

### EVALUATION TEST ###
# In[51]:

pr =model.predict_classes(X_test)
yh = y_test.argmax(2)
fyh, fpr = score(yh, pr)
print ('Testing accuracy:', accuracy_score(fyh, fpr))
print ('Testing confusion matrix:')
print (confusion_matrix(fyh, fpr))
print (classification_report(fyh,fpr))


### EVALUATION TEST ASR###
pr =model.predict_classes(X_test_asr)
yh = y_test_asr.argmax(2)
fyh, fpr = score(yh, pr)
print ('Testing ASR accuracy:', accuracy_score(fyh, fpr))
print ('Testing ASR confusion matrix:')
print (confusion_matrix(fyh, fpr))
print (classification_report(fyh,fpr))
