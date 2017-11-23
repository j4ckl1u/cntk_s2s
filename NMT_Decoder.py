from __future__ import print_function
import os
import cntk as C
import numpy as np
import math
import Config
import NMT_Model
import Corpus

class NMT_Decoder:

    def __init__(self, model, srcVocab, trgVocab):
        self.model = model
        self.srcVocab = srcVocab
        self.trgVocab = trgVocab
        self.networkBucket = {}
        self.srcHiddenStatesMem = np.zeros(shape=(Config.SrcMaxLength, Config.BatchSize, Config.SrcHiddenSize * 2), dtype=np.float32)
        self.srcSentEmbMem = np.zeros(shape=(1, Config.BatchSize, Config.SrcHiddenSize), dtype=np.float32)
        self.srcHiddenStates = C.input_variable(shape=(Config.SrcMaxLength, Config.BatchSize, Config.SrcHiddenSize * 2))
        self.srcSentEmb = C.input_variable(shape=(Config.BatchSize, Config.SrcHiddenSize))
        self.decoderPreHidden = C.input_variable(shape=(Config.BatchSize, Config.TrgHiddenSize))
        self.decoderPreWord = C.input_variable(shape=(Config.BatchSize, Config.TrgVocabSize), is_sparse=True)

    def getDecodingNetwork(self, srcLength):
        if(not self.networkBucket.has_key(srcLength)):
            self.networkBucket[srcLength] = self.model.createDecodingNetworks(self.srcSentEmb, self.srcHiddenStates,
                                                                              self.decoderPreWord, self.decoderPreHidden, srcLength)
        return self.networkBucket[srcLength]

    def greedyDecoding(self, src):
        maxSrcLength = max(len(x) for x in src)
        (sourceHiddenNet, decoderInitNet, decoderNet, predictNet) = self.getDecodingNetwork(maxSrcLength)
        (batchSrc, batchSrcMask) = Corpus.MonoCorpus.buildInputMono(src, Config.SrcVocabSize, Config.SrcMaxLength, self.srcVocab.getEndId())
        sourceHiddens = sourceHiddenNet.eval({self.model.inputMatrixSrc: batchSrc})
        sourceHiddens = sourceHiddens.reshape(maxSrcLength, Config.BatchSize, Config.SrcHiddenSize * 2)
        self.srcSentEmbMem[:,:,:] = sourceHiddens[0:1, :, Config.SrcHiddenSize:Config.SrcHiddenSize * 2]
        self.srcHiddenStatesMem[0:maxSrcLength,:,:] = sourceHiddens
        decoderHidden = decoderInitNet.eval({self.srcSentEmb: self.srcSentEmbMem})
        trans = predictNet.eval({self.decoderPreHidden: decoderHidden})
        count = 0
        transCands = []
        sentEndID = self.trgVocab.getEndId()
        while (np.all(trans != sentEndID) and count < Config.TrgMaxLength):
            transList = [int(t) for t in trans.tolist()[0]]
            transCands.append(transList)
            preWords = C.Value.one_hot(transList, Config.TrgVocabSize)
            decoderHidden = decoderNet.eval(
                {self.srcHiddenStates: self.srcHiddenStatesMem, self.decoderPreWord: preWords, self.decoderPreHidden: decoderHidden,
                 self.model.maskMatrixSrc: batchSrcMask})
            trans = predictNet.eval({self.decoderPreHidden: decoderHidden})
            count += 1

        trans = []
        for i in range(0, len(src), 1):
            trans.append([t[i] for t in transCands])
        return trans