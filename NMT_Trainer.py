from __future__ import print_function
import numpy as np
import os
import operator
import cntk as C
import numpy as np
import math
from scipy.sparse import csr_matrix
import RNN
import Config
import NMT_Model
import BiCorpus

class GRU_NMT:

    def __init__(self):
        self.model = NMT_Model.GRU_NMT_Model()
        self.trainData = BiCorpus.BiCorpus(Config.srcVocabF, Config.trgVocabF, Config.trainSrcF, Config.trainTrgF)
        self.valData = BiCorpus.BiCorpus(Config.srcVocabF, Config.trgVocabF, Config.valSrcF, Config.valTrgF)
        self.networkBucket = {}
        self.exampleNetwork = self.getNetwork(Config.BucketGap, Config.BucketGap)
        if os.path.isfile(Config.initModelF): self.model.loadModel(Config.initModelF)

    def getNetwork(self, lengthSrc, lengthTrg):
        bucketGap = Config.BucketGap
        networkSrcid = int(math.ceil(lengthSrc/float(bucketGap))*bucketGap)
        networkTrgid = int(math.ceil(lengthTrg/float(bucketGap))*bucketGap)
        networkSrcid = networkSrcid if networkSrcid <= Config.SrcMaxLength else Config.SrcMaxLength
        networkTrgid = networkTrgid if networkTrgid <= Config.TrgMaxLength else Config.TrgMaxLength

        if(not self.networkBucket.has_key((networkSrcid, networkTrgid))):
            print("Creating network (" + str(networkSrcid) + "," + str(networkTrgid) + ")", end="\r")
            self.networkBucket[(networkSrcid, networkTrgid)] = self.model.createNetwork(networkSrcid, networkTrgid)
            print("Bucket contains network for ", end="")
            for key in self.networkBucket:
                print("(" + str(key[0]) + "," + str(key[1]) + ")", end=" ")
            print()
        return self.networkBucket[(networkSrcid, networkTrgid)]

    def train(self):
        cePerWordBest = 10000
        for i in range(0, 100000, 1):
            cePerWordBest = self.validateAndSaveModel(i, cePerWordBest)
            print("Traing with batch " + str(i), end="\r")

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

    def validateAndSaveModel(self, i, cePerWordBest):
        if (i % Config.ValiditionPerBatch == 0):
            cePerWord = self.validate()
            if (cePerWord < cePerWordBest):
                self.model.saveModel(Config.modelF + "." + str(i))
                cePerWordBest = cePerWord
        return cePerWordBest

    def validate(self):
        self.curValSent = 0
        valBatch = self.valData.getValBatch()
        countAll = 0
        ceAll = 0
        print("Validation ...", end="\r")
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
        cePerWord = ceAll/countAll
        print("Validation cross entropy :" + str(cePerWord))
        return cePerWord

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
    C.device.try_set_default_device(C.device.gpu(Config.GPUID))
    gruNMT = GRU_NMT()
    gruNMT.train()
