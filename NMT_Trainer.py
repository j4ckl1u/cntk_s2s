from __future__ import print_function
import os
import cntk as C
import math
import Config
import NMT_Model
import NMT_Decoder
import Corpus
import nltk

class NMT_Trainer:

    def __init__(self):
        self.model = NMT_Model.NMT_Model()
        self.srcVocab = Corpus.Vocabulary()
        self.trgVocab = Corpus.Vocabulary()
        self.srcVocab.loadDict(Config.srcVocabF)
        self.trgVocab.loadDict(Config.trgVocabF)
        self.trainData = Corpus.BiCorpus(self.srcVocab, self.trgVocab, Config.trainSrcF, Config.trainTrgF)
        self.valData = Corpus.BiCorpus(self.srcVocab, self.trgVocab, Config.valSrcF, Config.valTrgF)
        self.valBleuData = Corpus.ValCorpus(self.srcVocab, self.trgVocab, Config.valFile, Config.refCount)
        self.decoder = NMT_Decoder.NMT_Decoder(self.model, self.srcVocab, self.trgVocab)
        self.networkBucket = {}
        self.exampleNetwork = self.getNetwork(1, 1)
        self.badValCount = 0
        self.maxBadVal = 5
        self.learningRate = Config.LearningRate
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
            print("Bucket contains networks for ", end="")
            for key in self.networkBucket: print("(" + str(key[0]) + "," + str(key[1]) + ")", end=" ")
            print()
        return self.networkBucket[(networkSrcid, networkTrgid)]

    def train(self):
        bestValScore = 10000
        for i in range(0, 100000000, 1):
            bestValScore = self.validateAndSaveModel(i, bestValScore)
            print("Traing with batch " + str(i), end="\r")

            trainBatch = self.trainData.getTrainBatch()
            maxSrcLength = max(len(x[0]) for x in trainBatch)
            maxTrgLength = max(len(x[1]) for x in trainBatch)
            network = self.getNetwork(maxSrcLength, maxTrgLength)

            (batchSrc, batchTrg, batchSrcMask, batchTrgMask) = self.trainData.buildInput(trainBatch)
            batchGrad = network.grad({self.model.inputMatrixSrc: batchSrc,
                                      self.model.inputMatrixTrg:batchTrg,
                                      self.model.maskMatrixSrc:batchSrcMask,
                                      self.model.maskMatrixTrg: batchTrgMask},
                                     self.model.Parameters)
            self.mySGDUpdate(batchGrad)

    def mySGDUpdate(self, batchGrad):
        for parameter in batchGrad:
            parameter.value = parameter.value - self.learningRate * batchGrad[parameter]

    def validateAndSaveModel(self, i, bestValScore):
        if (i % Config.ValiditionPerBatch == 0):
            valScore = self.validateBLEU()
            if (valScore < bestValScore):
                self.model.saveModel(Config.modelF + "." + str(i))
                bestValScore = valScore
                self.badValCount = 0
            else:
                self.badValCount += 1
                if(self.badValCount >= self.maxBadVal):
                    self.learningRate /=2
                    self.badValCount = 0
        return bestValScore

    def validate(self):
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

            (batchSrc, batchTrg, batchSrcMask, batchTrgMask) = self.valData.buildInput(valBatch)
            ce = network.eval({self.model.inputMatrixSrc: batchSrc,
                                      self.model.inputMatrixTrg:batchTrg,
                                      self.model.maskMatrixSrc:batchSrcMask,
                                      self.model.maskMatrixTrg: batchTrgMask})
            ceAll += ce
            valBatch = self.valData.getValBatch()
        cePerWord = ceAll/countAll
        print("Validation Cross Entropy :" + str(cePerWord))
        return cePerWord


    def validateBLEU(self):
        valBatch = self.valBleuData.getValBatch()
        valSrc = []
        valTrgGolden=[]
        valTrgTrans=[]
        print("Validation ...", end="\r")
        while (valBatch):
            srcIds = [pair[0] for pair in valBatch]
            transIDs = self.decoder.greedyDecoding(srcIds)
            src = [self.trainData.iD2Sent(srcId, True) for srcId in srcIds]
            golden = [[self.trainData.iD2Sent(ref, False) for ref in pair[1]] for pair in valBatch]
            trans = [self.trainData.iD2Sent(tran, False) for tran in transIDs]
            valSrc.extend(src)
            valTrgGolden.extend(golden)
            valTrgTrans.extend(trans)
            valBatch = self.valBleuData.getValBatch()
        bleu = -self.computeBleu(valTrgTrans, valTrgGolden)
        print("Validation BLEU Score :" + str(bleu))
        return bleu

    def computeBleu(self, trans, golden):
        return nltk.translate.bleu_score.corpus_bleu(golden, trans)


if __name__ == '__main__':
    C.device.try_set_default_device(C.device.gpu(Config.GPUID))
    nmtTrainer = NMT_Trainer()
    nmtTrainer.train()
