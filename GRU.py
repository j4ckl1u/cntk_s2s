import cntk as C

class GRU:

    def __init__(self, inputSize, hiddenSize):
        self.Wr = C.parameter(shape=(inputSize, hiddenSize), init=C.glorot_uniform())
        self.Wrb = C.parameter(shape=(hiddenSize), init=C.glorot_uniform())
        self.Wz = C.parameter(shape=(inputSize, hiddenSize), init=C.glorot_uniform())
        self.Wzb = C.parameter(shape=(hiddenSize), init=C.glorot_uniform())
        self.W = C.parameter(shape=(inputSize, hiddenSize), init=C.glorot_uniform())
        self.Wb = C.parameter(shape=(hiddenSize), init=C.glorot_uniform())

        self.Ur = C.parameter(shape=(hiddenSize, hiddenSize), init=C.glorot_uniform())
        self.Uz = C.parameter(shape=(hiddenSize, hiddenSize), init=C.glorot_uniform())
        self.U = C.parameter(shape=(hiddenSize, hiddenSize), init=C.glorot_uniform())
        self.Ub = C.parameter(shape=(hiddenSize), init=C.glorot_uniform())

        self.Parameters = [self.Wr, self.Wrb, self.Wz, self.Wzb, self.W, self.Wb, self.Ur, self.Uz, self.U, self.Ub]

    def createNetwork(self, inputEmb, preHidden):
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
        return CurH
