from __future__ import print_function
import cntk as C
import numpy as np
import numpy
import Config
import Corpus

class NMT_Decoder:

    def __init__(self, model, srcVocab, trgVocab):
        self.model = model
        self.srcVocab = srcVocab
        self.trgVocab = trgVocab
        self.srcHiddenStatesMem = np.zeros(shape=(Config.SrcMaxLength, Config.BatchSize, Config.SrcHiddenSize * 2), dtype=np.float32)
        self.srcSentEmbMem = np.zeros(shape=(1, Config.BatchSize, Config.SrcHiddenSize), dtype=np.float32)
        self.srcHiddenStates = C.input_variable(shape=(Config.SrcMaxLength, Config.BatchSize, Config.SrcHiddenSize * 2))
        self.srcSentEmb = C.input_variable(shape=(Config.BatchSize, Config.SrcHiddenSize))
        self.trgPreHidden = C.input_variable(shape=(Config.BatchSize, Config.TrgHiddenSize))
        self.trgPreWord = C.input_variable(shape=(Config.BatchSize, Config.TrgVocabSize), is_sparse=True)
        self.networkBucket = {}
        self.decoderInitNetwork = self.model.createDecodingInitNetwork(self.srcSentEmb)

    def getDecodingNetwork(self, srcLength):
        if(not self.networkBucket.has_key(srcLength)):
            decoderNetwork= self.model.createDecodingNetworks(self.srcHiddenStates, self.trgPreWord, self.trgPreHidden, srcLength)
            encoderNetwork = self.model.createEncoderNetwork(srcLength)
            self.networkBucket[srcLength]=[encoderNetwork, decoderNetwork]
        return self.networkBucket[srcLength]

    def runEncoderNetwork(self, sourceHiddenNet, batchSrc, batchSrcMask, maxSrcLength):
        sourceHiddens = sourceHiddenNet.eval({self.model.inputMatrixSrc: batchSrc, self.model.maskMatrixSrc:batchSrcMask})
        sourceHiddens = sourceHiddens.reshape(maxSrcLength, Config.BatchSize, Config.SrcHiddenSize * 2)
        self.srcSentEmbMem[:, :, :] = sourceHiddens[0:1, :, Config.SrcHiddenSize:Config.SrcHiddenSize * 2]
        self.srcHiddenStatesMem[0:maxSrcLength, :, :] = sourceHiddens


    def buildTrgForDebug(self, sentences, maxLength,endID):
        sent = []
        for i in range(0, maxLength, 1):
            for j in range(0, Config.BatchSize, 1):
                if (j < len(sentences) and i < len(sentences[j])):
                    sent.append(sentences[j][i])
                else:
                    sent.append(endID)
        return sent

    def greedyDecoding(self, srcWords):
        count = 0
        transCands = []
        sentEndID = self.trgVocab.getEndId()
        maxSrcLength = max(len(x) for x in srcWords)
        [encoderNetwork, decoderNetwork] =  self.getDecodingNetwork(maxSrcLength)

        (batchSrc, batchSrcMask) = Corpus.MonoCorpus.buildInputMono(srcWords, Config.SrcVocabSize, Config.SrcMaxLength,
                                                                    self.srcVocab.getEndId())
        self.runEncoderNetwork(encoderNetwork, batchSrc, batchSrcMask, maxSrcLength)

        initPredict = self.decoderInitNetwork[0].eval({self.srcSentEmb: self.srcSentEmbMem})
        decoderHidden = initPredict[self.decoderInitNetwork[1][0]]
        #preSoftmax = initPredict[self.decoderInitNetwork[1][1]]
        trans = initPredict[self.decoderInitNetwork[1][1]]
        #numpy.savetxt("encoder.txt0" , preSoftmax.reshape(preSoftmax.shape[0] * preSoftmax.shape[1],
        #                                                  preSoftmax.shape[2]*preSoftmax.shape[3]), fmt='%.5e')
        while (np.any(trans != sentEndID) and count < Config.TrgMaxLength):

            transList = [int(t) for t in trans.tolist()[0]]
            transCands.append(transList)
            count += 1

            preWords = C.Value.one_hot(transList, Config.TrgVocabSize)
            decoderPredict = decoderNetwork[0].eval(
                {self.srcHiddenStates: self.srcHiddenStatesMem, self.trgPreWord: preWords, self.trgPreHidden: decoderHidden,
                 self.model.maskMatrixSrc: batchSrcMask})

            decoderHidden = decoderPredict[decoderNetwork[1][0]]
            #preSoftmax = decoderPredict[decoderNetwork[1][1]]
            trans = decoderPredict[decoderNetwork[1][1]]
            #numpy.savetxt("encoder.txt" + str(i), preSoftmax.reshape(preSoftmax.shape[0] * preSoftmax.shape[1],
            #                                                         preSoftmax.shape[2] * preSoftmax.shape[3]), fmt='%.5e')

        transR = []
        for i in range(0, len(srcWords), 1):
            transR.append([t[i] for t in transCands])
        return transR