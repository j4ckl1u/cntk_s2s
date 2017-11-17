import Config
import re
from random import shuffle

class Vocabulary:

    def __init__(self):
        self.word2ID = {}
        self.id2Word = []

    def loadDict(self, dictf):
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

class BiCorpus:

    def __init__(self, srcVocabF, trgVocabF, srcF, trgF, shuffle = False):
        self.srcVocab = Vocabulary()
        self.trgVocab = Vocabulary()
        self.sentencePairs = []
        self.batchPool = []
        self.batchId = 0
        self.curSent = 0
        self.srcVocab.loadDict(srcVocabF)
        self.trgVocab.loadDict(trgVocabF)
        self.needShuffle = shuffle
        self.loadData(srcF, trgF)

    def loadData(self, fileSrc, fileTrg):
        print("loading data " + fileSrc + "-->" + fileTrg)
        sentences = []
        fsrc=open(fileSrc)
        ftrg = open(fileTrg)
        line = fsrc.readline()
        while line:
            line = line.strip()
            words = re.split("\s+", line)
            srcS = []
            for word in words:
                srcS.append(self.srcVocab.getID(word))
            line = ftrg.readline()
            line = line.strip() + " </s>"
            words = re.split("\s+", line)
            trgS = []
            for word in words:
                trgS.append(self.trgVocab.getID(word))
            self.sentencePairs.append((srcS, trgS))
            line = fsrc.readline()
        fsrc.close()
        ftrg.close()

    def buildBatchPool(self):
        batchPool = []
        sentences = self.getSentences(Config.BatchSize * 100)
        sentences = sorted(sentences, key=lambda sent : len(sent[1]))
        self.batchPool = [sentences[x:x + Config.BatchSize] for x in xrange(0, len(sentences), Config.BatchSize)]
        shuffle(self.batchPool)
        self.batchId = 0

    def reset(self):
        self.curSent = 0
        if (self.needShuffle): shuffle(self.sentencePairs)

    def getSentences(self, num):
        sentences = []
        for i in range(0, num, 1):
            if(self.curSent + i >= len(self.sentencePairs)-1): self.reset()
            sentence = self.sentencePairs[self.curSent + i]
            sentences.append(sentence)
        self.curSent += len(sentences)
        return sentences

    def getTrainBatch(self):
        if(self.batchId >= len(self.batchPool) -1): self.buildBatchPool()
        rBatch = self.batchPool[self.batchId]
        self.batchId += 1
        return rBatch

    def getValBatch(self, num=Config.BatchSize):
        if (self.curSent >= len(self.sentencePairs) - 1):
            self.curSent = 0
            return None
        sentences = []
        for i in range(0, num, 1):
            if(self.curSent + i >= len(self.sentencePairs)): break
            sentence = self.sentencePairs[self.curSent + i]
            sentences.append(sentence)
        self.curSent += len(sentences)
        return sentences

    def getEndId(self, source=True):
        if(source):
            return self.srcVocab.getID("</s>")
        else:
            return self.trgVocab.getID("</s>")

