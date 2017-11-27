import cntk as C

GPUID = 2
UseLSTM = False
EmbeddingSize = 256
SrcHiddenSize = 256
TrgHiddenSize = 512
SrcVocabSize = 10000
TrgVocabSize = 10000
SrcMaxLength = 50
TrgMaxLength = 50
BatchSize = 8
LearningRate = 0.001
ValiditionPerBatch = 10000
BucketGap = 10

srcVocabF = "D:/users/Shujie/Code/cntk_s2s/IWSLT/train/c.dict.txt"
trgVocabF = "D:/users/Shujie/Code/cntk_s2s/IWSLT/train/e.dict.txt"
trainSrcF = "D:/users/Shujie/Code/cntk_s2s/IWSLT/train/c.txt"
trainTrgF =  "D:/users/Shujie/Code/cntk_s2s/IWSLT/train/e.txt"
valSrcF = "D:/users/Shujie/Code/cntk_s2s/IWSLT/valid/c.txt"
valTrgF = "D:/users/Shujie/Code/cntk_s2s/IWSLT/valid/e.txt"
valFile = "D:/users/Shujie/Code/cntk_s2s/IWSLT/valid/dev.txt"
refCount = 7
modelF = "D:/users/Shujie/Code/cntk_s2s/IWSLT/model1/cntk.model"
initModelF = "D:/users/Shujie/Code/cntk_s2s/IWSLT/model/cntk.model.21000"

#trgVocabF = "D:/user/Shujie/Data/PennLM/data/dict.txt"
#trainTrgF =  "D:/user/Shujie/Data/PennLM/data/ptb.train.txt"
#valTrgF = "D:/user/Shujie/Data/PennLM/data/ptb.valid.txt"
#modelF = "D:/user/Shujie/Data/PennLM/model/cntk.model"
#initModelF = "D:/user/Shujie/Data/PennLM/model/cntk.model.99000"

defaultInit = C.initializer.glorot_normal
