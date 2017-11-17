from __future__ import print_function
import os
import cntk as C
import numpy as np
import math
import Config
import RNNLM_Model
import Corpus

class RNNLM_Trainer:

    def __init__(self):
        self.model = RNNLM_Model.LM_Model()
        self.trainData = Corpus.MonoCorpus(Config.trgVocabF, Config.trainTrgF)
        self.valData = Corpus.MonoCorpus(Config.trgVocabF, Config.valTrgF)
        self.networkBucket = {}
        self.exampleNetwork = self.getNetwork(Config.BucketGap)
        if os.path.isfile(Config.initModelF): self.model.loadModel(Config.initModelF)

    def getNetwork(self, lengthTrg):
        bucketGap = Config.BucketGap
        networkTrgid = int(math.ceil(lengthTrg/float(bucketGap))*bucketGap)
        networkTrgid = networkTrgid if networkTrgid <= Config.TrgMaxLength else Config.TrgMaxLength
        if(not self.networkBucket.has_key(networkTrgid)):
            print("Creating network (" + str(networkTrgid) + ")", end="\r")
            self.networkBucket[networkTrgid] = self.model.createNetwork(networkTrgid)
            print("Bucket contains networks for ", end="")
            for key in self.networkBucket: print("(" + str(key) + ")", end=" ")
            print()
        return self.networkBucket[networkTrgid]

    def train(self):
        cePerWordBest = 10000
        for i in range(0, 100000, 1):
            cePerWordBest = self.validateAndSaveModel(i, cePerWordBest)
            print("Traing with batch " + str(i), end="\r")

            trainBatch = self.trainData.getTrainBatch()
            maxTrgLength = max(len(x) for x in trainBatch)
            network = self.getNetwork(maxTrgLength)

            (batch, batchMask) = self.buildInput(trainBatch)
            batchGrad = network.grad({self.model.inputMatrixTrg:batch,
                                      self.model.maskMatrixTrg: batchMask},
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
            count = sum(len(s) for s in valBatch)
            countAll += count
            maxTrgLength = max(len(x) for x in valBatch)
            network = self.getNetwork(maxTrgLength)

            (batch, batchMask) = self.buildInput(valBatch)
            ce = network.eval({self.model.inputMatrixTrg:batch, self.model.maskMatrixTrg: batchMask})
            ceAll += ce
            valBatch = self.valData.getValBatch()
        cePerWord = ceAll/countAll
        print("Validation cross entropy :" + str(cePerWord))
        return cePerWord

    def buildInput(self, sentences):
        vocabSize = Config.TrgVocabSize
        maxLength = Config.TrgMaxLength

        sent = []
        for i in range(0, maxLength, 1):
            for j in range(0, Config.BatchSize, 1):
                if (j < len(sentences) and i < len(sentences[j])):
                    sent.append(sentences[j][i])
                else:
                    sent.append(self.trainData.getEndId())
        batch = C.Value.one_hot(sent, vocabSize)

        batchMask = np.zeros((maxLength, Config.BatchSize), dtype=np.float32)
        for i in range(0, len(sentences), 1):
            sentence = sentences[i]
            lastIndex = len(sentence) if len(sentence) < maxLength else maxLength
            batchMask[0:lastIndex, i] = 1

        return (batch, batchMask)

if __name__ == '__main__':
    C.device.try_set_default_device(C.device.gpu(Config.GPUID))
    rnnLMTrainer = RNNLM_Trainer()
    rnnLMTrainer.train()
