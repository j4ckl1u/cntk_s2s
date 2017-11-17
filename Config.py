import cntk as C

EmbeddingSize = 256
SrcHiddenSize = 256
TrgHiddenSize = 512
SrcVocabSize = 10000
TrgVocabSize = 10000
SrcMaxLength = 60
TrgMaxLength = 60
BatchSize = 32
LearningRate = 0.0001
ValiditionPerBatch = 1000
BucketGap = 10

srcVocabF = "D:/user/Shujie/Data/IWSLT/train/c.dict.txt"
trgVocabF = "D:/user/Shujie/Data/IWSLT/train/e.dict.txt"
trainSrcF = "D:/user/Shujie/Data/IWSLT/train/c.txt"
trainTrgF =  "D:/user/Shujie/Data/IWSLT/train/e.txt"
valSrcF = "D:/user/Shujie/Data/IWSLT/valid/c.txt"
valTrgF = "D:/user/Shujie/Data/IWSLT/valid/e.txt"
modelF = "D:/user/Shujie/Data/IWSLT/model/cntk.model"
initModelF = "D:/user/Shujie/Data/IWSLT/model/cntk.model.0"

defaultInit = C.initializer.glorot_normal
