import cntk as C

GPUID = 6
UseLSTM = False
EmbeddingSize = 256
SrcHiddenSize = 256
TrgHiddenSize = 512
SrcVocabSize = 10000
TrgVocabSize = 10000
SrcMaxLength = 50
TrgMaxLength = 50
BatchSize = 8
LearningRate = 0.00001
ValiditionPerBatch = 2000
BucketGap = 10

srcVocabF = "D:/user/Shujie/Data/IWSLT/train/c.dict.txt"
trgVocabF = "D:/user/Shujie/Data/IWSLT/train/e.dict.txt"
trainSrcF = "D:/user/Shujie/Data/IWSLT/train/c.txt"
trainTrgF =  "D:/user/Shujie/Data/IWSLT/train/e.txt"
valSrcF = "D:/user/Shujie/Data/IWSLT/valid/c.txt"
valTrgF = "D:/user/Shujie/Data/IWSLT/valid/e.txt"
valFile = "D:/user/Shujie/Data/IWSLT/valid/dev.txt"
refCount = 7
modelF = "D:/user/Shujie/Data/IWSLT/model/cntk.model"
initModelF = "D:/user/Shujie/Data/IWSLT/model/cntk.model.xx"

#trgVocabF = "D:/user/Shujie/Data/PennLM/data/dict.txt"
#trainTrgF =  "D:/user/Shujie/Data/PennLM/data/ptb.train.txt"
#valTrgF = "D:/user/Shujie/Data/PennLM/data/ptb.valid.txt"
#modelF = "D:/user/Shujie/Data/PennLM/model/cntk.model"
#initModelF = "D:/user/Shujie/Data/PennLM/model/cntk.model.99000"

defaultInit = C.initializer.glorot_normal
