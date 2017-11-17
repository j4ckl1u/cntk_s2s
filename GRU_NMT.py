from __future__ import print_function
import numpy as np
import os
import operator
import cntk as C
import numpy as np
import math
from scipy.sparse import csr_matrix
import GRU
import Config
import NMT_Model
import BiCorpus

class GRU_NMT:
    networkBucket = {}

    def __init__(self):
        self.model = NMT_Model.GRU_NMT_Model()
        self.trainData = BiCorpus.BiCorpus(Config.srcVocabF, Config.trgVocabF, Config.trainSrcF, Config.trainTrgF)
        self.valData = BiCorpus.BiCorpus(Config.srcVocabF, Config.trgVocabF, Config.valSrcF, Config.valTrgF)
        self.exampleNet = self.getNetwork(5, 5)
        if os.path.isfile(Config.initModelF): self.model.loadModel(Config.initModelF)

    def getNetwork(self, lengthSrc, lengthTrg):
        networkSrcid = (lengthSrc/5 if lengthSrc%5 ==0 else lengthSrc/5 + 1)*5
        networkTrgid = (lengthTrg / 5 if lengthTrg % 5 == 0 else lengthTrg / 5 + 1) * 5
        networkSrcid = networkSrcid if networkSrcid <= Config.SrcMaxLength else Config.SrcMaxLength
        networkTrgid = networkTrgid if networkTrgid <= Config.TrgMaxLength else Config.TrgMaxLength

        if(not self.networkBucket.has_key((networkSrcid, networkTrgid))):
            self.networkBucket[(networkSrcid, networkTrgid)] = self.model.createNetwork(networkSrcid, networkTrgid)
        return self.networkBucket[(networkSrcid, networkTrgid)]

    def train(self):
        for i in range(0, 100000, 1):
            if (i % 100 == 0):
                self.validate()
                self.model.saveModel(Config.modelF + "." + str(i))
            print("traing with batch " + str(i), end="\r")
            trainBatch = self.trainData.getTrainBatch()
            maxSrcLength = len(max(trainBatch, key=lambda x: x[0])[0])
            maxTrgLength = len(max(trainBatch, key=lambda x: x[1])[1])
            network = self.getNetwork(maxSrcLength, maxTrgLength)

            (batchSrc, batchTrg, batchSrcMask, batchTrgMask) = self.buildInput(trainBatch)
            batchGrad = network.grad({self.model.inputMatrixSrc: batchSrc,
                                      self.model.inputMatrixTrg:batchTrg,
                                      self.model.maskMatrixSrc:batchSrcMask,
                                      self.model.maskMatrixTrg: batchTrgMask},
                                     self.model.Parameters)
            self.mySGDUpdate(batchGrad)

    def mySGDUpdate(self, batchGrad):
        for parameter in batchGrad:
            parameter.value = parameter.value - Config.LearningRate * batchGrad[parameter]

    def validate(self):
        self.curValSent = 0
        valBatch = self.valData.getValBatch()
        countAll = 0
        ceAll = 0
        while(valBatch):
            count = sum(len(s[1]) for s in valBatch)
            countAll += count
            maxSrcLength = max(len(x[0]) for x in valBatch)
            maxTrgLength = max(len(x[1]) for x in valBatch)
            network = self.getNetwork(maxSrcLength, maxTrgLength)

            (batchSrc, batchTrg, batchSrcMask, batchTrgMask) = self.buildInput(valBatch)
            ce = network.eval({self.model.inputMatrixSrc: batchSrc,
                                      self.model.inputMatrixTrg:batchTrg,
                                      self.model.maskMatrixSrc:batchSrcMask,
                                      self.model.maskMatrixTrg: batchTrgMask})
            ceAll += ce
            valBatch = self.valData.getValBatch()
        print("Validation cross entropy :" + str(ceAll/countAll))

    def buildInputMono(self, sentences, srctrg = 0):
        vocabSize = Config.SrcVocabSize if srctrg==0 else Config.TrgVocabSize
        maxLength = Config.SrcMaxLength if srctrg == 0 else Config.TrgMaxLength

        sent = []
        for i in range(0, maxLength, 1):
            for j in range(0, Config.BatchSize, 1):
                if (j < len(sentences) and i < len(sentences[j][srctrg])):
                    sent.append(sentences[j][srctrg][i])
                else:
                    sent.append(self.trainData.getEndId(i==0))
        batch = C.Value.one_hot(sent, vocabSize)

        batchMask = np.zeros((maxLength, Config.BatchSize), dtype=np.float32)
        for i in range(0, len(sentences), 1):
            sentence = sentences[i][srctrg]
            lastIndex = len(sentence) if len(sentence) < maxLength else maxLength
            batchMask[0:lastIndex, i] = 1

        return (batch, batchMask)

    def buildInput(self, sentences):
        (batchSrc, batchSrcMask) = self.buildInputMono(sentences, 0)
        (batchTrg, batchTrgMask) = self.buildInputMono(sentences, 1)
        return (batchSrc, batchTrg, batchSrcMask, batchTrgMask)

if __name__ == '__main__':
    C.device.try_set_default_device(C.device.gpu(2))
    gruNMT = GRU_NMT()
    gruNMT.train()
