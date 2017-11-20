import cntk as C
import RNN
import Config
import  numpy as np

class LM_Model:
    def __init__(self):
        RNNCell = RNN.GRUN if not Config.UseLSTM else RNN.LSTM
        self.Emb = C.layers.Embedding(Config.EmbeddingSize, init=Config.defaultInit())
        self.Decoder = RNNCell(Config.EmbeddingSize, Config.TrgHiddenSize)
        self.Wt = C.parameter(shape=(Config.TrgHiddenSize, Config.TrgVocabSize), init=Config.defaultInit())
        self.Wtb = C.parameter(shape=(Config.TrgVocabSize), init=Config.defaultInit())
        self.firstHidden = C.constant(0, shape=(1, Config.BatchSize, Config.TrgHiddenSize))
        self.inputMatrixTrg = C.input_variable(shape=(Config.TrgMaxLength * Config.BatchSize, Config.TrgVocabSize), is_sparse=True)
        self.maskMatrixTrg = C.input_variable(shape=(Config.TrgMaxLength, Config.BatchSize))
        self.Parameters = [self.Emb.E, self.Wt, self.Wtb]
        self.Parameters.extend(self.Decoder.Parameters)

    def createNetwork(self, length):
        networkHiddenTrg = {}
        networkMemTrg = {}
        inputTrg = C.reshape(self.inputMatrixTrg, shape=(Config.TrgMaxLength, Config.BatchSize, Config.TrgVocabSize))
        tce = 0
        for i in range(0, length - 1, 1):
            if (i == 0):
                networkHiddenTrg[i] = self.firstHidden
                networkMemTrg[i] = networkHiddenTrg[i]
            else:
                curWord = self.Emb(inputTrg[i])
                (networkHiddenTrg[i], networkMemTrg[i]) = self.Decoder.createNetwork(curWord, networkHiddenTrg[i - 1], networkMemTrg[i-1])

            preSoftmax = C.times(networkHiddenTrg[i], self.Wt) + self.Wtb
            ce = C.cross_entropy_with_softmax(preSoftmax, inputTrg[i + 1], 2)
            tce += C.times(C.reshape(ce, shape=(1, Config.BatchSize)),
                           C.reshape(self.maskMatrixTrg[i], shape=(Config.BatchSize, 1)))
        return tce

    def saveModel(self, filename):
        print("Saving model " + filename)
        f = open(filename, "wb")
        for parameter in self.Parameters:
            pValue = parameter.value
            np.save(f, pValue)
        f.close()

    def loadModel(self, filename):
        print("Loading model " + filename)
        f = open(filename, "rb")
        for parameter in self.Parameters:
            pValue = np.load(f)
            parameter.value = pValue
        f.close()