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
import Corpus

class NMT_Trainer:

    def __init__(self):
        self.model = NMT_Model.NMT_Model()
        self.trainData = Corpus.BiCorpus(Config.srcVocabF, Config.trgVocabF, Config.trainSrcF, Config.trainTrgF)
        self.valData = Corpus.BiCorpus(Config.srcVocabF, Config.trgVocabF, Config.valSrcF, Config.valTrgF)
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

            (batchSrc, batchTrg, batchSrcMask, batchTrgMask) = self.buildInput(trainBatch)
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

            (batchSrc, batchTrg, batchSrcMask, batchTrgMask) = self.buildInput(valBatch)
            ce = network.eval({self.model.inputMatrixSrc: batchSrc,
                                      self.model.inputMatrixTrg:batchTrg,
                                      self.model.maskMatrixSrc:batchSrcMask,
                                      self.model.maskMatrixTrg: batchTrgMask})
            ceAll += ce
            valBatch = self.valData.getValBatch()
        cePerWord = ceAll/countAll
        print("Validation Cross Entropy :" + str(cePerWord))
        return cePerWord

    def greedyDecoding(self, src):
        maxSrcLength = max(len(x) for x in src)

        srcHiddenStates = C.input_variable(shape=(maxSrcLength * Config.BatchSize, Config.SrcHiddenSize * 2))
        srcSentEmb = C.input_variable(shape=(Config.BatchSize, Config.SrcHiddenSize))
        decoderPreHidden = C.input_variable(shape=(Config.BatchSize, Config.TrgHiddenSize))
        decoderPreWord = C.input_variable(shape=(Config.BatchSize, Config.TrgVocabSize), is_sparse=True)
        sourceHiddenNet = self.model.createEncoderNetwork(maxSrcLength)[0]
        decoderInitNet = self.model.createTestingDecoderInitNetwork(srcSentEmb)
        decoderNet = self.model.createTestingDecoderNetwork(srcHiddenStates, decoderPreWord, decoderPreHidden, maxSrcLength)
        predictNet = self.model.createTestingPredictNetwork(decoderPreHidden)

        (batchSrc, batchSrcMask) = self.buildInputMono(src, 0)
        sourceHiddens = sourceHiddenNet.eval({self.model.inputMatrixSrc: batchSrc, self.model.maskMatrixSrc:batchSrcMask})
        sourceHiddens = sourceHiddens.reshape(maxSrcLength*Config.BatchSize, Config.SrcHiddenSize*2)
        decoderHidden = decoderInitNet.eval({srcSentEmb:sourceHiddens[0:Config.BatchSize, Config.SrcHiddenSize:Config.SrcHiddenSize*2]})
        trans = predictNet.eval({decoderPreHidden:decoderHidden})
        count = 0
        transCands=[]
        sentEndID = self.valData.getEndId(False)
        while(np.all(trans != sentEndID) and count < Config.TrgMaxLength):
            transList = [int(t) for t in trans.tolist()[0]]
            transCands.append(transList)
            preWords = C.Value.one_hot(transList, Config.TrgVocabSize)
            decoderHidden = decoderNet.eval({srcHiddenStates: sourceHiddens, decoderPreWord:preWords, decoderPreHidden:decoderHidden, self.model.maskMatrixSrc:batchSrcMask})
            trans = predictNet.eval({decoderPreHidden:decoderHidden})
            count+=1

        trans = []
        for i in range(0, len(src), 1):
            trans.append([t[i] for t in transCands])
        return trans

    def validateBLEU(self):
        valBatch = self.valData.getValBatch()
        valSrc = []
        valTrgGolden=[]
        valTrgTrans=[]
        print("Validation ...", end="\r")
        while (valBatch):
            srcIds = [pair[0] for pair in valBatch]
            transIDs = self.greedyDecoding(srcIds)
            src = [self.trainData.iD2Sent(srcId, True) for srcId in srcIds]
            golden = [self.trainData.iD2Sent(pair[1], False) for pair in valBatch]
            trans = [self.trainData.iD2Sent(tran, False) for tran in transIDs]
            valSrc.extend(src)
            valTrgGolden.extend(golden)
            valTrgTrans.extend(trans)
            valBatch = self.valData.getValBatch()
        bleu = -self.computeBleu(valTrgTrans, valTrgGolden)
        print("Validation BLEU Score :" + str(bleu))
        return bleu

    def computeBleu(self, trans, golden):
        return 1.0

    def buildInputMono(self, sentences, srctrg = 0):
        vocabSize = Config.SrcVocabSize if srctrg==0 else Config.TrgVocabSize
        maxLength = Config.SrcMaxLength if srctrg == 0 else Config.TrgMaxLength

        sent = []
        for i in range(0, maxLength, 1):
            for j in range(0, Config.BatchSize, 1):
                if (j < len(sentences) and i < len(sentences[j])):
                    sent.append(sentences[j][i])
                else:
                    sent.append(self.trainData.getEndId(srctrg==0))
        batch = C.Value.one_hot(sent, vocabSize)

        batchMask = np.zeros((maxLength, Config.BatchSize), dtype=np.float32)
        for i in range(0, len(sentences), 1):
            sentence = sentences[i]
            lastIndex = len(sentence) if len(sentence) < maxLength else maxLength
            batchMask[0:lastIndex, i] = 1

        return (batch, batchMask)

    def buildInput(self, sentences):
        (batchSrc, batchSrcMask) = self.buildInputMono([pair[0] for pair in sentences], 0)
        (batchTrg, batchTrgMask) = self.buildInputMono([pair[1] for pair in sentences], 1)
        return (batchSrc, batchTrg, batchSrcMask, batchTrgMask)

if __name__ == '__main__':
    C.device.try_set_default_device(C.device.gpu(Config.GPUID))
    nmtTrainer = NMT_Trainer()
    nmtTrainer.train()
