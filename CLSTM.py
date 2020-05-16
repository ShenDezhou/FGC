import codecs
import os
import numpy
from keras import regularizers
from keras.initializers import Constant
from keras.layers import Dense, Embedding, CuDNNLSTM, SpatialDropout1D, Input, Bidirectional, Dropout, \
    BatchNormalization, Lambda, concatenate, Conv1D, MaxPooling1D
from keras.models import Model
from keras.models import load_model
from scipy import sparse
from sklearn_crfsuite import metrics

#               precision    recall  f1-score   support
#
#            A     0.8333    0.1154    0.2027       130
#            B     0.5248    0.9769    0.6828       130
#
#    micro avg     0.5462    0.5462    0.5462       260
#    macro avg     0.6791    0.5462    0.4427       260
# weighted avg     0.6791    0.5462    0.4427       260


words = []
with codecs.open('dataset/actor_dic.utf8', 'r', encoding='utf8') as fa:
    lines = fa.readlines()
    lines = [line.strip() for line in lines]
    words.extend(lines)

rxwdict = dict(zip(words, range(1, 1 + len(words))))
rxwdict['\n'] = 0

rydict = dict(zip(list("AB"), range(len("AB"))))
ytick = [0, 263.5, 244001]


def getYClass(y):
    r = 0
    for i in range(len(ytick) - 1):
        if int(y) >= ytick[i] and int(y) <= ytick[i + 1]:
            return r
        r += 1
    assert r < len(ytick), (y, r)
    return r


batch_size = 100
nFeatures = 5
seqlen = 225
totallen = nFeatures + seqlen
word_size = 11
actors_size = 8380
nfilters = 150
kernelSize = 3
Hidden = 150
Regularization = 1e-4
Dropoutrate = 0.2
learningrate = 0.2
Marginlossdiscount = 0.2
STATES = list("AB")
nState = 2
EPOCHS = 100

modelfile = os.path.basename(__file__).split(".")[0]

loss = "squared_hinge"
optimizer = "nadam"

sequence = Input(shape=(totallen,))
seqsa = Lambda(lambda x: x[:, 0:5])(sequence)
seqsb = Lambda(lambda x: x[:, 5:])(sequence)
seqsc = Lambda(lambda x: x[:, 5:])(sequence)

network_emb = sparse.load_npz("embedding/weibo_wembedding.npz").todense()
embedded = Embedding(len(words) + 1, word_size, embeddings_initializer=Constant(network_emb), input_length=seqlen,
                     mask_zero=False, trainable=True)(seqsb)

networkcore_emb = sparse.load_npz("embedding/weibo_coreembedding.npz").todense()
embeddedc = Embedding(len(words) + 1, actors_size, embeddings_initializer=Constant(networkcore_emb),
                      input_length=seqlen, mask_zero=False, trainable=True)(seqsc)

dropout = Dropout(rate=Dropoutrate)(seqsa)
middle = Dense(Hidden, activation='relu', kernel_regularizer=regularizers.l2(Regularization))(dropout)
batchNorm = BatchNormalization()(middle)

dropoutb = SpatialDropout1D(rate=Dropoutrate)(embedded)
blstm = Bidirectional(CuDNNLSTM(Hidden, return_sequences=False), merge_mode='sum')(dropoutb)
batchNormb = BatchNormalization()(blstm)

dropoutc = SpatialDropout1D(rate=Dropoutrate)(embeddedc)
conv = Conv1D(filters=nfilters, kernel_size=kernelSize)(dropoutc)
mpool = MaxPooling1D()(conv)
blstmc = Bidirectional(CuDNNLSTM(Hidden, return_sequences=False), merge_mode='sum')(mpool)
batchNormc = BatchNormalization()(blstmc)

concat = concatenate([batchNorm, batchNormb, batchNormc])

dense = Dense(nState, activation='softmax', kernel_regularizer=regularizers.l2(Regularization))(concat)
model = Model(input=sequence, output=dense)
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

model.summary()


MODE = 1

if MODE == 1:
    with codecs.open('dataset/fgc_training.utf8', 'r', encoding='utf8') as fx:
        with codecs.open('dataset/fgc_training_states.utf8', 'r', encoding='utf8') as fy:
            xlines = fx.readlines()
            ylines = fy.readlines()
            assert len(xlines) == len(ylines)
            X = []
            print('process X list.')
            counter = 0
            for i in range(len(xlines)):
                line = xlines[i].strip()
                segs = line.split(",")
                item = []
                sents = [float(s) for s in segs[0:5]]
                item.extend(sents)
                anames = segs[5:]
                item.extend([0] * (totallen - len(item) - len(anames)))
                item.extend([rxwdict.get(name, 0) for name in anames])
                assert len(item) == totallen, (len(item))
                X.append(item)
                if counter % 1000 == 0 and counter != 0:
                    print('.')
            X = numpy.array(X)
            print(X.shape)

            y = []
            print('process y list.')
            for line in ylines:
                line = line.strip()
                yi = numpy.zeros((len(STATES)), dtype=int)
                yi[getYClass(line)] = 1
                y.append(yi)
            y = numpy.array(y)
            print(y.shape)

            history = model.fit(X, y, batch_size=batch_size, nb_epoch=EPOCHS, verbose=1)
            model.save("model/%s.h5" % modelfile)
            print('FIN')

    # if MODE == 2:
    with codecs.open('dataset/fgc_test.utf8', 'r', encoding='utf8') as fx:
        with codecs.open('dataset/fgc_test_states.utf8', 'r', encoding='utf8') as fy:
            with codecs.open('output/fgc_test_%s_states.utf8' % modelfile, 'w', encoding='utf8') as fp:
                model = load_model("model/%s.h5" % modelfile)
                model.summary()

                xlines = fx.readlines()
                X = []
                print('process X list.')
                counter = 0
                for i in range(len(xlines)):
                    line = xlines[i].strip()
                    segs = line.split(",")
                    item = []
                    sents = [float(s) for s in segs[0:5]]
                    item.extend(sents)
                    anames = segs[5:]
                    item.extend([0] * (totallen - len(item) - len(anames)))
                    item.extend([rxwdict.get(name, 0) for name in anames])
                    # pad right '\n'
                    # print(len(item))
                    assert len(item) == totallen, (len(item))
                    X.append(item)
                    if counter % 1000 == 0 and counter != 0:
                        print('.')
                    counter += 1
                X = numpy.array(X)
                print(X.shape)

                yp = model.predict(X)
                print(yp.shape)
                for i in range(yp.shape[0]):
                    i = numpy.argmax(yp[i])
                    fp.write(STATES[i])
                    fp.write('\n')
                print('FIN')

    GOLD = 'dataset/fgc_test_states_gold.utf8'
    with codecs.open('output/fgc_test_%s_states.utf8' % modelfile, 'r', encoding='utf8') as fj:
        with codecs.open(GOLD, 'r', encoding='utf8') as fg:
            jstates = fj.readlines()
            states = fg.readlines()
            y = []
            for state in states:
                state = state.strip()
                y.append(list(state))
            yp = []
            for jstate in jstates:
                jstate = jstate.strip()
                yp.append(list(jstate))

            assert len(yp) == len(y)
            m = metrics.flat_classification_report(
                y, yp, labels=list("AB"), digits=4
            )
            print(m)
            print('FIN')
