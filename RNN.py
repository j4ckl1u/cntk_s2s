import cntk as C
import Config

class GRU:

    def __init__(self, inputSize, hiddenSize):
        self.Wr = C.parameter(shape=(inputSize, hiddenSize), init=Config.defaultInit())
        self.Wrb = C.parameter(shape=(hiddenSize), init=Config.defaultInit())
        self.Wz = C.parameter(shape=(inputSize, hiddenSize), init=Config.defaultInit())
        self.Wzb = C.parameter(shape=(hiddenSize), init=Config.defaultInit())
        self.W = C.parameter(shape=(inputSize, hiddenSize), init=Config.defaultInit())
        self.Wb = C.parameter(shape=(hiddenSize), init=Config.defaultInit())

        self.Ur = C.parameter(shape=(hiddenSize, hiddenSize), init=Config.defaultInit())
        self.Uz = C.parameter(shape=(hiddenSize, hiddenSize), init=Config.defaultInit())
        self.U = C.parameter(shape=(hiddenSize, hiddenSize), init=Config.defaultInit())
        self.Ub = C.parameter(shape=(hiddenSize), init=Config.defaultInit())

        self.Parameters = [self.Wr, self.Wrb, self.Wz, self.Wzb, self.W, self.Wb, self.Ur, self.Uz, self.U, self.Ub]

    def createNetwork(self, inputEmb, preHidden, preMem=None):
        WrX = C.times(inputEmb, self.Wr) + self.Wrb
        UrH = C.times(preHidden, self.Ur)
        R = C.sigmoid(WrX+UrH)

        WzX = C.times(inputEmb, self.Wz) + self.Wzb
        UzH = C.times(preHidden, self.Uz)
        Z = C.sigmoid(WzX + UzH)

        UH=C.times(preHidden, self.U) + self.Ub
        UHR = C.element_times(UH, R)

        WX = C.times(inputEmb, self.W) + self.Wb
        HTilde = C.tanh(WX + UHR)

        CurH = C.element_times(HTilde, 1-Z) + C.element_times(preHidden, Z)
        return (CurH, None)

class GRUN:

    def __init__(self, inputSize, hiddenSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.W = C.parameter(shape=(inputSize, hiddenSize*3), init=Config.defaultInit())
        self.Wb = C.parameter(shape=(hiddenSize*3), init=Config.defaultInit())
        self.U = C.parameter(shape=(hiddenSize, hiddenSize*3), init=Config.defaultInit())
        self.Ub = C.parameter(shape=(hiddenSize*3), init=Config.defaultInit())
        self.Parameters = [self.W, self.Wb, self.U, self.Ub]

    def createNetwork(self, inputEmb, preHidden, preMem=None):
        WX = C.times(inputEmb, self.W) + self.Wb
        UH = C.times(preHidden, self.U) + self.Ub

        R = C.sigmoid(C.slice(WX, -1, 0, self.hiddenSize) + C.slice(UH, -1, 0, self.hiddenSize))
        Z = C.sigmoid(C.slice(WX, -1, self.hiddenSize, self.hiddenSize*2) + C.slice(UH, -1, self.hiddenSize, self.hiddenSize*2))

        UHR = C.element_times(C.slice(UH, -1, self.hiddenSize*2, self.hiddenSize*3), R)
        HTilde = C.tanh(C.slice(WX, -1, self.hiddenSize*2, self.hiddenSize*3) + UHR)

        CurH = C.element_times(HTilde, 1-Z) + C.element_times(preHidden, Z)
        return (CurH, None)

class LSTM:

    def __init__(self, inputSize, hiddenSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.W = C.parameter(shape=(inputSize, hiddenSize*4), init=Config.defaultInit())
        self.Wb = C.parameter(shape=(hiddenSize*4), init=Config.defaultInit())
        self.U = C.parameter(shape=(hiddenSize, hiddenSize*4), init=Config.defaultInit())
        self.Ub = C.parameter(shape=(hiddenSize*4), init=Config.defaultInit())
        self.Parameters = [self.W, self.Wb, self.U, self.Ub]

    def createNetwork(self, inputEmb, preHidden, preMem):
        WX = C.times(inputEmb, self.W) + self.Wb
        UH = C.times(preHidden, self.U) + self.Ub

        I = C.sigmoid(C.slice(WX, -1, 0, self.hiddenSize) +
                      C.slice(UH, -1, 0, self.hiddenSize))
        O = C.sigmoid(C.slice(WX, -1, self.hiddenSize, self.hiddenSize*2) +
                      C.slice(UH, -1, self.hiddenSize, self.hiddenSize*2))
        F = C.sigmoid(C.slice(WX, -1, self.hiddenSize*2, self.hiddenSize*3) +
                      C.slice(UH, -1, self.hiddenSize*2, self.hiddenSize*3))
        N = C.tanh(C.slice(WX, -1, self.hiddenSize*3, self.hiddenSize*4) +
                      C.slice(UH, -1, self.hiddenSize*3, self.hiddenSize*4))

        NI = C.element_times(N, I)
        FM = C.element_times(F, preMem)
        CurMem = NI + FM
        CurH = C.element_times(C.tanh(CurMem), O)
        return (CurH, CurMem)