import cntk as C
import GRU
import Config
import  numpy as np

class GRU_NMT_Model:

    def __init__(self):
        self.EmbSrc = C.layers.Embedding(Config.EmbeddingSize, init=Config.defaultInit())
        self.EmbTrg = C.layers.Embedding(Config.EmbeddingSize, init=Config.defaultInit())
        self.EncoderL2R = GRU.GRU(Config.EmbeddingSize, Config.SrcHiddenSize)
        self.EncoderR2L = GRU.GRU(Config.EmbeddingSize, Config.SrcHiddenSize)
        self.Decoder = GRU.GRU(Config.EmbeddingSize + Config.SrcHiddenSize*2, Config.TrgHiddenSize)
        self.Wt = C.parameter(shape=(Config.TrgHiddenSize, Config.TrgVocabSize), init=Config.defaultInit())
        self.Wtb = C.parameter(shape=(Config.TrgVocabSize), init=Config.defaultInit())
        self.WI = C.parameter(shape=(Config.SrcHiddenSize, Config.TrgHiddenSize), init=Config.defaultInit())
        self.WIb = C.parameter(shape=(Config.TrgHiddenSize), init=Config.defaultInit())
        self.Was = C.parameter(shape=(Config.SrcHiddenSize*2, Config.TrgHiddenSize), init=Config.defaultInit())
        self.Wat = C.parameter(shape=(Config.TrgHiddenSize, Config.TrgHiddenSize), init=Config.defaultInit())
        self.Wav = C.parameter(shape=(Config.TrgHiddenSize, 1), init=Config.defaultInit())
        self.firstHidden = C.constant(0, shape=(Config.BatchSize, Config.SrcHiddenSize))
        self.inputMatrixSrc = C.input_variable(shape=(Config.SrcMaxLength * Config.BatchSize, Config.SrcVocabSize), is_sparse=True)
        self.inputMatrixTrg = C.input_variable(shape=(Config.TrgMaxLength * Config.BatchSize, Config.TrgVocabSize), is_sparse=True)
        self.maskMatrixSrc = C.input_variable(shape=(Config.SrcMaxLength, Config.BatchSize))
        self.maskMatrixTrg = C.input_variable(shape=(Config.TrgMaxLength, Config.BatchSize))
        self.Parameters = [self.EmbSrc.E, self.EmbTrg.E, self.Wt, self.Wtb, self.WI, self.WIb, self.Was, self.Wat, self.Wav]
        self.Parameters.extend(self.EncoderL2R.Parameters)
        self.Parameters.extend(self.EncoderR2L.Parameters)
        self.Parameters.extend(self.Decoder.Parameters)

    def createEncoderNetwork(self, srcLength):
        networkHiddenSrcL2R = {}
        networkHiddenSrcR2L = {}
        inputSrc = C.reshape(self.inputMatrixSrc, shape=(Config.SrcMaxLength, Config.BatchSize, Config.SrcVocabSize))
        for i in range(0, srcLength, 1):
            networkHiddenSrcL2R[i] = self.EncoderL2R.createNetwork(self.EmbSrc(inputSrc[i]),
                                                          self.firstHidden if i == 0 else networkHiddenSrcL2R[i - 1])
            networkHiddenSrcR2L[srcLength - i - 1] = self.EncoderR2L.createNetwork(self.EmbSrc(inputSrc[srcLength - i - 1]),
                                                          self.firstHidden if i == 0 else networkHiddenSrcR2L[srcLength - i])
        networkHiddenSrc = []
        for i in range(0, srcLength, 1):
            networkHiddenSrc.append(C.splice(networkHiddenSrcL2R[i], networkHiddenSrcR2L[i], axis=2))
        if(srcLength > 1):
            sourceHidden = C.splice(networkHiddenSrc[0], networkHiddenSrc[1], axis=0)
            for i in range(2, srcLength, 1):
                sourceHidden = C.splice(sourceHidden, networkHiddenSrc[i], axis=0)
        else:
            sourceHidden = C.reshape(networkHiddenSrc[0], shape=(1, Config.BatchSize, Config.SrcHiddenSize*2))
        return (sourceHidden, networkHiddenSrcR2L[0] + networkHiddenSrcL2R[srcLength-1])

    def createDecoderInitNetwork(self, initTrgHidden):
        WIS = C.times(initTrgHidden, self.WI) + self.WIb
        return C.tanh(WIS)

    def createAttentionNet(self, hiddenSrc, curHiddenTrg, srcLength):
        srcHiddenSize = Config.SrcHiddenSize*2
        hsw = C.times(hiddenSrc, self.Was)
        htw = C.times(curHiddenTrg, self.Wat)
        hst = C.reshape(hsw, shape=(srcLength, Config.BatchSize * Config.TrgHiddenSize)) + C.reshape(htw, shape=(1, Config.BatchSize * Config.TrgHiddenSize))
        hstT = C.reshape(C.tanh(hst), shape=(srcLength * Config.BatchSize, Config.TrgHiddenSize))
        attScore = C.reshape(C.times(hstT, self.Wav), shape=(srcLength, Config.BatchSize))
        maskOut = (C.slice(self.maskMatrixSrc, 0, 0, srcLength) -1)*999999
        nAttScore = attScore + maskOut
        attProb = C.reshape(C.softmax(nAttScore, axis=0),shape=(srcLength, Config.BatchSize, 1))
        attVector =hiddenSrc*attProb
        contextVector =C.reduce_sum(C.reshape(attVector, shape=(srcLength, Config.BatchSize * srcHiddenSize)), axis=0)
        return C.reshape(contextVector, shape=(1, Config.BatchSize, srcHiddenSize))

    def createDecoderNetwork(self, networkHiddenSrc, initTrgHidden, srcLength, trgLength):
        networkHiddenTrg = {}
        inputTrg = C.reshape(self.inputMatrixTrg, shape=(Config.TrgMaxLength, Config.BatchSize, Config.TrgVocabSize))
        tce = 0
        for i in range(0, trgLength - 1, 1):
            if (i == 0):
                networkHiddenTrg[i] = self.createDecoderInitNetwork(initTrgHidden)
            else:
                contextVect = self.createAttentionNet(networkHiddenSrc, networkHiddenTrg[i - 1], srcLength)
                curWord = self.EmbTrg(inputTrg[i])
                curInput = C.splice(contextVect, curWord, axis=2)
                networkHiddenTrg[i] = self.Decoder.createNetwork(curInput, networkHiddenTrg[i - 1])
            preSoftmax = C.times(networkHiddenTrg[i], self.Wt) + self.Wtb
            ce = C.cross_entropy_with_softmax(preSoftmax, inputTrg[i + 1], 2)
            tce += C.times(C.reshape(ce, shape=(1, Config.BatchSize)),
                           C.reshape(self.maskMatrixTrg[i], shape=(Config.BatchSize, 1)))
        return tce

    def createNetwork(self, srcLength, trgLength):
        (networkHiddenSrc, initTrgHidden) = self.createEncoderNetwork(srcLength)
        decoderNet = self.createDecoderNetwork(networkHiddenSrc, initTrgHidden, srcLength, trgLength)
        return decoderNet

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