import numpy as np
import codecs
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dense
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import Merge,Bidirectional, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
import keras
from collections import Counter

from sklearn.preprocessing import LabelBinarizer
import gc
import nltk
import sys
import os

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
#no_question    =   1
#optimizer_     =   'adam'
#patience_		=	2

# In[4]:
#### READ TRAIN ####
if no_question==1:
	raw = open('merged_train_punct_word_lc.txt', 'r').readlines()
else:
	raw = open('merged_train_punct_word_lc_four.txt', 'r').readlines()

all_x = []
point = []
for line in raw:
    stripped_line = line.split()
    #print (stripped_line)
    all_x.append(stripped_line)

for i in (range(len(all_x))):
    if 3==(len(all_x[i])):
        #print (all_x[i])
        del all_x[i][1]
        #print (all_x[i])


# Read Hungarian Embedding
import os
BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/HUNVEC/'
#EMBEDDING_DIM = 600 # Embedding dimensions

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'hunglove.txt'))
for line in f:
    values = line.split() #Tokenizer
    word = values[0] # word form
    coefs = np.asarray(values[1:], dtype='float32') # embedding coefficients
    embeddings_index[word] = coefs
f.close()

print('Number of embeddings:', len(embeddings_index))
print('Ten first element from `és` Embedding Vector :', embeddings_index["és"][:10])

### CHUNKING TEXTS #
lengths = [len(x) for x in all_x]
print ('Input sequence length range: ', max(lengths), min(lengths))

X = [x[0] for x in all_x]
y = [x[1].upper() for x in all_x]

all_text = [c for x in X for c in x]


x_chunks=[]
y_chunks=[]
for group in chunker(X, chunk_size):
    x_chunks.append(group)
for group in chunker(y, chunk_size, is_y=True):
    y_chunks.append(group)

### CREATE VOCABULARY ###
a=len(x_chunks)
for_voca=x_chunks[:int(len(x_chunks)*0.8)]


total = []
for i in for_voca:
    total += i

words = list(total)
words = [x for x in words if x != '<EOS>']

# Count the word frequencies
word_freq = nltk.FreqDist(words)
print ("Found %d unique words tokens." % len(word_freq.items()))

words = word_freq.most_common(vocabulary_size)


vocab=["<PAD>"]+[x[0] for x in words]
vocab.append("UNKNOWN")
vocab.append("<EOS>")

### TRANSFORM TEXTS #
word2ind = {word: index for index, word in enumerate(vocab)}
ind2word = {index: word for index, word in enumerate(word2ind)}


#FULLSTOP=PERIOD
labels = ['O','COMMA','FULLSTOP']
if no_question==0:
	labels.append('QUESTION','EXCLAMATION')
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


train_len=int(len(X_enc)*0.8)
val_len=int(len(X_enc)*0.2)

X_train = X_enc[:train_len]
y_train = y_enc[:train_len]

X_val = X_enc[train_len:len(X_enc)]
y_val = y_enc[train_len:len(X_enc)]


print (train_len)
print (val_len)


out_size = len(label2ind)
oov=0

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
    else:
        oov+=1

print ('Number of remaining lines from embedding matrix:', len(embedding_matrix))
print (oov)


### CREATE RNN Model with BiLSTM ###
model = Sequential()
model.add(Embedding(nb_words+3, embedding_size, input_length=maxlen,weights=[embedding_matrix],mask_zero=True))
model.add(Bidirectional(LSTM(hidden_size,return_sequences=True)))

model.add(TimeDistributed(Dense(out_size)))
model.add(Activation('softmax'))

earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience_,verbose=0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer_, metrics=['accuracy','precision','recall','fmeasure'])

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, validation_data=(X_val, y_val), callbacks=[earlyStopping],shuffle=True)

### EVALUATION TRAIN ###
#pr = model.predict_classes(X_train)
#yh = y_train.argmax(2)
#fyh, fpr = score(yh, pr)
#print ('Training accuracy:', accuracy_score(fyh, fpr))
#print ('Training confusion matrix:')
#print (confusion_matrix(fyh, fpr))
#print (classification_report(fyh,fpr))

### EVALUATION DEV ###
#pr = model.predict_classes([X_val_f])
#yh = y_val.argmax(2)
#fyh, fpr = score(yh, pr)
#print ('Validation accuracy:', accuracy_score(fyh, fpr))
#print ('Validation confusion matrix:')
#print (confusion_matrix(fyh, fpr))
#print (classification_report(fyh,fpr))


### READ TEST ###
if no_question==1:
	raw = open('TESZTV7_word_lc.txt', 'r').readlines()
else:
	raw = open('tesztfour.txt').readlines()

all_x_test = []
for line in raw:
    stripped_line = line.split()
    #print (stripped_line)
    all_x_test.append(stripped_line)

for i in (range(len(all_x_test))):
    if 3==(len(all_x_test[i])):
        #print (all_x[i])
        del all_x_test[i][1]
        #print (all_x[i])

for i in (range(len(all_x_test))):
    if 1==(len(all_x_test[i])):
        print (all_x_test[i])

X = [x[0] for x in all_x_test]
all_text = [c for x in X for c in x]

x_test=[]
y_test=[]
for group in chunker(X, chunk_size):
    x_test.append(group)

from copy import deepcopy
x_test_orig=deepcopy(x_test)


### TRANSFORM TEXTS #
for i, sent in enumerate(x_test):
    x_test[i] = [w if w in word2ind else unknown_token for w in sent]

X_enc = [[word2ind[c] for c in x] for x in x_test]


X_enc_f = pad_sequences(X_enc, maxlen=maxlen, padding="post", value=word2ind["<PAD>"] )
test_len=int(len(X_enc_f)*1.0)

X_test_f = X_enc_f[:test_len]

### PUNCTUATION PREDICTION FOR REFERENCE TRANSCRIPTS #
pr = model.predict_classes([X_test_f])

from datetime import datetime
filename1 = datetime.now().strftime("%Y%m%d-%H%M%S")


predict=open("prediction_ref_"+filename1+".txt",'w')

def decode(x):
    n=[]
    for i in range(1,(len(x))):
        if x[i]==0:
            n.append("O")
        elif x[i]==1:
            n.append("COMMA")
        elif x[i]==2:
            n.append("FULLSTOP")
        elif x[i]==3:
            n.append("QUESTION")
        elif x[i]==4:
            n.append("EXCLAMATION")
    return n

for i in range(len(pr)):
    a=decode(pr[i])
    for j in range(0,len(x_test_orig[i])-1):
        #print x_test[i][j]
        predict.write((x_test_orig[i][j]+"\t"+a[j]+"\n"))

predict.close()


### READ TEST ASR###
if no_question==1:
	raw = open('asr_output_to_tag_nn.txt', 'r').readlines()
else:
	raw = open('asr_output_to_tag_nn.txt', 'r').readlines()

all_x_test_asr = []
for line in raw:
    stripped_line = line.split()
    all_x_test_asr.append(stripped_line)

X = [x[0] for x in all_x_test_asr]
all_text = [c for x in X for c in x]

x_test_asr=[]

#CHUNKING#
for group in chunker(X, chunk_size):
    x_test_asr.append(group)

from copy import deepcopy
x_test_orig_asr=deepcopy(x_test_asr)

### TRANSFORM TEXTS #
for i, sent in enumerate(x_test_asr):
    x_test_asr[i] = [w if w in word2ind else unknown_token for w in sent]

X_enc = [[word2ind[c] for c in x] for x in x_test_asr]

max_label = max(label2ind.values())+1
X_enc = pad_sequences(X_enc, maxlen=maxlen,padding='post')
test_asr_len=int(len(X_enc)*1.0)

X_test_asr = X_enc[:test_asr_len]

### PUNCTUATION PREDICTION FOR ASR TRANSCRIPTS #
pr = model.predict_classes([X_test_asr])

from datetime import datetime
filename1 = datetime.now().strftime("%Y%m%d-%H%M%S")

predict=open("prediction"+filename1+".txt",'w')

def decode(x):
    n=[]
    for i in range(1,(len(x))):
        if x[i]==0:
            n.append("O")
        elif x[i]==1:
            n.append("COMMA")
        elif x[i]==2:
            n.append("FULLSTOP")
        elif x[i]==3:
            n.append("QUESTION")
        elif x[i]==4:
            n.append("EXCLAMATION")
    return n

for i in range(len(pr)):
    a=decode(pr[i])
    for j in range(0,len(x_test_orig_asr[i])-1):
        #print x_test[i][j]
        predict.write((x_test_orig_asr[i][j]+"\t"+a[j]+"\n"))

predict.close()

import sys
sys.exit()