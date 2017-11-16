from __future__ import print_function
import numpy as np
import os
import operator
from random import shuffle
import re
import cntk as C
import numpy as np
import math
from scipy.sparse import csr_matrix

class GRU:

    def __init__(self, embSize, hiddenSize, inputVocabSize, outputVocabSize):
        self.hiddenSize = hiddenSize
        self.Emb = C.parameter(shape=(inputVocabSize, embSize), init=C.glorot_uniform())
        self.Wt = C.parameter(shape=(hiddenSize, outputVocabSize), init=C.glorot_uniform())
        self.Wtb = C.parameter(shape=(outputVocabSize), init=C.glorot_uniform())

        self.Wr = C.parameter(shape=(embSize, hiddenSize), init=C.glorot_uniform())
        self.Wrb = C.parameter(shape=(hiddenSize), init=C.glorot_uniform())
        self.Wz = C.parameter(shape=(embSize, hiddenSize), init=C.glorot_uniform())
        self.Wzb = C.parameter(shape=(hiddenSize), init=C.glorot_uniform())
        self.W = C.parameter(shape=(embSize, hiddenSize), init=C.glorot_uniform())
        self.Wb = C.parameter(shape=(hiddenSize), init=C.glorot_uniform())

        self.Ur = C.parameter(shape=(hiddenSize, hiddenSize), init=C.glorot_uniform())
        self.Uz = C.parameter(shape=(hiddenSize, hiddenSize), init=C.glorot_uniform())
        self.U = C.parameter(shape=(hiddenSize, hiddenSize), init=C.glorot_uniform())
        self.Ub = C.parameter(shape=(hiddenSize), init=C.glorot_uniform())

        self.Parameters = [self.Emb, self.Wt, self.Wtb, self.Wr, self.Wrb, self.Wz, self.Wzb, self.W, self.Wb, self.Ur, self.Uz, self.U, self.Ub]

    def createNetwork(self, inputWord, preHidden):
        inputEmb = C.times(inputWord, self.Emb)
        WrX = C.times(inputEmb, self.Wr) + self.Wrb
        UrH = C.times(preHidden, self.Ur)
        R = C.sigmoid(WrX + UrH)

        WzX = C.times(inputEmb, self.Wz) + self.Wzb
        UzH = C.times(preHidden, self.Uz)
        Z = C.sigmoid(WzX + UzH)

        UH = C.times(preHidden, self.U) + self.Ub
        UHR = C.element_times(UH, R)

        WX = C.times(inputEmb, self.W) + self.Wb
        HTilde = C.tanh(WX + UHR)

        CurH = C.element_times(HTilde, 1 - Z) + C.element_times(preHidden, Z)
        Output = C.times(CurH, self.Wt) + self.Wtb
        return (CurH, Output)

class GRULM:
    networkBucket = {}
    word2ID = {}
    id2Word = []
    trainData = []
    valData=[]
    curTrainSent = 0
    curValSent = 0
    batchSize = 100
    def __init__(self, maxLength, vocabSize, embSize, hiddenSize, learningRate = 1.0):
        self.vocabSize = vocabSize
        self.maxLength = maxLength
        self.embSize = embSize
        self.hiddenSize = hiddenSize
        self.model = GRU(embSize, hiddenSize, vocabSize, vocabSize)
        self.firstHidden =  C.constant(0, shape=(self.batchSize, self.hiddenSize))
        self.inputMatrix = C.input_variable(shape=(self.maxLength, self.batchSize, self.vocabSize))
        self.maskMatrix = C.input_variable(shape=(self.maxLength, self.batchSize))
        self.batchInput = np.zeros((self.maxLength, self.batchSize, self.vocabSize), dtype=np.float32)
        self.batchMask = np.zeros((self.maxLength, self.batchSize), dtype=np.float32)

    def init(self, dictf, trainf, valf):
        self.loadDict(dictf)
        self.trainData = self.loadData(trainf)
        self.valData = self.loadData(valf)
        self.createNetwork()

    def createNetworkI(self, length, input, mask):
        networkHidden = {}
        tce = 0
        for i in range(0, length-1, 1):
            curTimeNet = self.model.createNetwork(input[i], self.firstHidden if i ==0 else networkHidden[i-1])
            networkHidden[i] = curTimeNet[0]
            preSoftmax = curTimeNet[1]
            ce = C.cross_entropy_with_softmax(preSoftmax, input[i+1], 2)
            #mce= C.element_times(C.reshape(ce, shape=(1, self.batchSize)), C.reshape(mask[i], shape=(1, self.batchSize)))
            #tce = C.splice(tce, mce, axis=0)
            tce += C.times(C.reshape(ce, shape=(1, self.batchSize)), C.reshape(mask[i], shape=(self.batchSize, 1)))
        return tce

    def createNetwork(self):
        print("creating network ", end="")
        for i in range(5, self.maxLength + 1, 5):
            print(str(i) + " ", end="")
            network = self.createNetworkI(i, self.inputMatrix, self.maskMatrix)
            self.networkBucket[i] = network
        print("")

    def getNetwork(self, length):
        networkid = (length/5 if length%5 ==0 else length/5 + 1)*5
        if(self.networkBucket.has_key(networkid)):
            return self.networkBucket[networkid]
        else:
            return None

    def train(self):
        for i in range(0, 100000, 1):
            if (i % 100 == 0):
                self.validate()
            print("traing with batch " + str(i), end="\r")
            sentences = self.getTrainBatch()
            (length, count) = self.buildInput(sentences)
            network = self.getNetwork(length)
            batchGrad = network.grad({self.inputMatrix: self.batchInput,
                                      self.maskMatrix: self.batchMask},
                                     self.model.Parameters)
            self.mySGDUpdate(batchGrad)

    def mySGDUpdate(self, batchGrad, lr = 0.0001):
        for parameter in batchGrad:
            parameter.value = parameter.value - lr * batchGrad[parameter]

    def validate(self):
        self.curValSent = 0
        valBatch = self.getValBatch()
        countAll = 0
        ceAll = 0
        while(valBatch):
            (length,count) = self.buildInput(valBatch)
            network = self.getNetwork(length)
            countAll += count
            ce = network.eval({self.inputMatrix:self.batchInput, self.maskMatrix:self.batchMask})
            #for sentence in valBatch:
            #    print(str(len(sentence)), end="\t")
            #print(str(ce) + "/" + str(count) + "=" + str(ce/count))
            ceAll += ce
            valBatch = self.getValBatch()
        print(" Validation cross entropy :" + str(ceAll/countAll))

    def getTrainBatch(self):
        #print("load training batch from " + str(self.curTrainSent))
        if (self.curTrainSent >= len(self.trainData) - 1):
            self.curTrainSent = 0
            self.shuffleData()
        sentences = []
        for i in range(0, self.batchSize, 1):
            if(self.curTrainSent + i >= len(self.trainData)):
                break
            sentence = self.trainData[self.curTrainSent + i]
            sentences.append(sentence)
        self.curTrainSent += len(sentences)
        return sentences

    def getValBatch(self):
        #print("load val batch from " + str(self.curValSent))
        if (self.curValSent >= 1000 or self.curValSent >= len(self.valData) - 1):
            return None
        sentences = []
        for i in range(0, self.batchSize, 1):
            if(self.curValSent + i >= len(self.valData)):
                break
            sentence = self.valData[self.curValSent + i]
            sentences.append(sentence)
        self.curValSent += len(sentences)
        return sentences

    def buildInput(self, sentences):
        self.batchMask = 0*self.batchMask
        self.batchInput = 0*self.batchInput
        maxLength = 0
        count = 0
        for i in range(0, len(sentences), 1):
            sentence = sentences[i]
            maxLength = maxLength if maxLength > len(sentence) else len(sentence)
            lastIndex = len(sentence) if len(sentence) < self.maxLength else self.maxLength
            for j in range(0, lastIndex, 1):
                self.batchInput[j, i, sentence[j]] = 1
                self.batchMask[j, i] = 1
            self.batchMask[lastIndex - 1, i] = 0
            count += lastIndex-1
        maxLength = maxLength if maxLength < self.maxLength else self.maxLength
        return (maxLength, count)

    def loadDict(self, dictf):
        print("loading vocabulary " + dictf)
        f = open(dictf)
        line = f.readline()
        wid = 0
        while line:
            key = line.split("\t")[0]
            self.word2ID[key] = wid
            wid = wid + 1
            self.id2Word.append(key)
            line = f.readline()
        f.close()

    def buildDict(self, file, dictf):
        print("building vocabulary for " + file)
        f = open(file)
        line = f.readline()
        wordCounter = {}
        while line:
            line = line.strip()+ " </s>"
            words = re.split("\s+", line)
            for word in words:
                if(wordCounter.has_key(word)):
                    wordCounter[word] = wordCounter[word] + 1
                else:
                    wordCounter[word] = 1
            line = f.readline()
        f.close()
        wordCounter["</s>"] = 100000000
        wordCounter["<unk>"] = 90000000
        f = open(dictf, "w")
        for word in sorted(wordCounter, key = wordCounter.get, reverse=True):
            f.write(word + "\t" + str(wordCounter[word]) + "\n")
        f.close()

    def getID(self, word):
        if self.word2ID.has_key(word):
            return self.word2ID[word]
        else:
            return self.word2ID["<unk>"]

    def loadData(self, file):
        print("loading data " + file)
        sentences = []
        f=open(file)
        line = f.readline()
        while line:
            line = line.strip() + " </s>"
            words = re.split("\s+", line)
            sentence = []
            for word in words:
                sentence.append(self.getID(word))
            sentences.append(sentence)
            line = f.readline()
        f.close()
        return sentences

    def shuffleData(self):
        shuffle(self.trainData)

if __name__ == '__main__':
    gruLM = GRULM(55, 10000, 512, 1024)
    #gruLM.buildDict("D:\user\Shujie\Data\PennLM\data\ptb.train.txt", "D:\user\Shujie\Data\PennLM\data\dict.txt")
    gruLM.init("D:\user\Shujie\Data\PennLM\data\dict.txt", "D:\user\Shujie\Data\PennLM\data\ptb.train.txt", "D:\user\Shujie\Data\PennLM\data\ptb.valid.txt")
    gruLM.train()
