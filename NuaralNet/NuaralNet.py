import random
import math
import copy
import tensorflow as tf

def listMul(list):
    ValueCount = 1;                                             # Total number of enteties in Matrix
    for i in list:                                             
        ValueCount = ValueCount * i
    return(ValueCount)

def flattern(mat):
    retMat = mat[0]
    for i in range(len(mat[1]) - 1):
        newMat = []
        for a1 in retMat:
            for a2 in a1:
                newMat.append(a2)
        retMat = newMat
    ValueCount = listMul(mat[1])                                         # Total number of enteties in Matrix
    return([retMat, [ValueCount]])  
def shape(mat, Dimentions):
    if(listMul(Dimentions) == mat[1][0]):
        RetMat = mat[0]
        iMul = 1;
        for i in Dimentions:                                        # Creating the dimentions
            TempMat = []
            iMul = iMul * i
            for i2 in range(int(mat[1][0]/iMul)):
                SubMat = []
                for i3 in range(i):
                    SubMat.append(RetMat[i2*i+i3])
                TempMat.append(SubMat)
            RetMat = TempMat
        RetMat.append(Dimentions)
        return(RetMat)
    return
def generateValues(count, type):
    list = []
    for i in range(count):
        if(type[0].lower() == "random"):
            list.append(random.uniform(type[1], type[2]))
        elif(type[0].lower() == "mono"):
            list.append(type[1])
    return(list)
def matrix(Dimentions, config):
    valueCount = listMul(Dimentions)
    values = generateValues(valueCount, config)
    return(shape([values, [valueCount]], Dimentions))
def visualizeMatrix(mat):
    if(len(mat[1]) == 1):                                               # Print 1D
        for value in mat[0]:
            print(value, end="\t")

    elif(len(mat[1]) == 2):                                             # Print 2D
        for row in mat[0]:
            for value in row:
                print(value, end="\t")
            print()
    elif(len(mat[1]) == 3):                                             # Print 3D
        layercounter = 0;
        for layer in mat[0]:
            print("[Layer: ", layercounter,"]\n")
            for row in layer:
                for value in row:
                    print(value, end="\t")
                print()
            layercounter = layercounter+1;
            print()
    return()
def matrixMul(mat1, mat2):
    mat1_ = copy.deepcopy(mat1)
    mat2_ = copy.deepcopy(mat2)
    if(len(mat1_[1]) >= len(mat2_[1])):                                   # Finding highest dimention matrix and ajdusting the other one to fit it
        matDim = len(mat1_[1])
        for i in range(matDim-len(mat2_[1])):
            mat2_[0] = [mat2_[0]]
            mat2_[1].append(1)
    else:
        matDim = len(mat2_[1])
        for i in range(matDim-len(mat1_[1])):
            mat1_[0] = [mat1_[0]]
            mat1_[1].append(1)
    if(mat1_[1][0] != mat2_[1][1]):
        return

    if(matDim == 2):                                                        # Making sure the matrix is 2D
        RetMat = []
        for row1 in range(mat1_[1][1]):                                     # Matrix multiplication 
            row = []                                                    
            for column2 in range(mat2_[1][0]):
                sum = 0
                for column1 in range(mat1_[1][0]):
                    sum = sum + mat1_[0][row1][column1] * mat2_[0][column1][column2]
                row.append(sum)
            RetMat.append(row)
        if(mat1_[1][1] == 1):
            return([RetMat[0], [mat2_[1][0]]])
        else:
            return([RetMat, (mat2_[1][0], mat1_[1][1])])
    return
def matrixAdd(mat1,mat2):
    if(mat1[1] == mat2[1]):
        retMat = []
        flatMat1 = flattern(mat1)
        flatMat2 = flattern(mat2)
        for i in range(flatMat1[1][0]):
            retMat.append(flatMat1[0][i]+flatMat2[0][i])
        return(shape([retMat, flatMat1[1]], mat1[1]))
    return
def network(layerConfig, valueConfig, functionConfig):
    weights = []
    biases = []
    for i in range(len(layerConfig)):
        if(i != len(layerConfig)-1):
            iList = isinstance(layerConfig[i], (list,))
            i1List = isinstance(layerConfig[i+1], (list,))
            
            if(iList):
                inputs = layerConfig[i][0]
            else:
                inputs = layerConfig[i]
            if(i1List):
                nodes = layerConfig[i+1][0]
            else:
                nodes = layerConfig[i+1]

            weights.append(matrix([nodes, inputs], valueConfig))
            biases.append(matrix([nodes], valueConfig))
            if(i1List):
                for layer in range(layerConfig[i+1][1]-1):
                    weights.append(matrix([layerConfig[i+1][0],layerConfig[i+1][0]], valueConfig))
                    biases.append(matrix([layerConfig[i+1][0]], valueConfig))

    network = {"biases": biases, "weights": weights, "functionConfig": functionConfig}
    return(network)

def actFunc(functionConfig):
    functions = {"sigmoid": (lambda x : 1 / (1 + math.e**(-x)))}
    return(functions[functionConfig.lower()])
def actDer(functionConfig):
    functions = {"sigmoid": (lambda x : (math.e**x) / (1 + math.e**x)**2)}
    return(functions[functionConfig.lower()])
def costFunc(functionConfig):
    functions = {"abs": (lambda x, y: abs(x-y)), "square": (lambda x, y: (x-y)**2)}
    return(functions[functionConfig])
def costDer(functionConfig):
    functions = {"abs": (lambda x, y: (x-y)/abs(x-y)), "square": (lambda x, y: 2*(x-y))}
    return(functions[functionConfig])

def matAct(mat, functionConfig):
    flatMat = flattern(mat)
    func = actFunc(functionConfig)
    retMat = list(map(func, flatMat[0]))
    return(shape([retMat, [len(retMat)]], mat[1]))
def runNetwork(inputs, net):
    weights = net["weights"]
    biases = net["biases"]
    functionConfig = net["functionConfig"]
    if(not isinstance(inputs[0], (list,))):
        inputs = [inputs, [len(inputs)]]
    runValues = [inputs]
    for i in range(len(weights)):
        nodes = matrixMul(runValues[-1], weights[i])
        nodes = matrixAdd(nodes, biases[i])
        nodes = matAct(nodes, functionConfig)
        runValues.append(nodes)
        
    return(runValues) 

def matCost(mat1, mat2, functionConfig):
    if(not isinstance(mat1[0], (list,))):
        mat1 = [mat1, [len(mat1)]]
    if(not isinstance(mat2[0], (list,))):
        mat2 = [mat2, [len(mat2)]]
    if(mat1[1] == mat2[1]):
        flatMat1 = flattern(mat1)
        flatMat2 = flattern(mat2)
        retMat = []
        for i in range(flatMat1[1][0]):
            retMat.append(costFunc(functionConfig)(flatMat1[0][i], flatMat2[0][i]))
        return([shape([retMat, flatMat1[1]], mat1[1]), functionConfig])
    return
def diag(mat1):
    if(len(mat1[1]) == 1):
        retMat = []
        for i in range(mat1[1][0]):
            row = []
            for i2 in range(mat1[1][0]):
                if(i == i2):
                    row.append(mat1[0][i])
                else:
                    row.append(0)
            retMat.append(row)
        return([retMat, [mat1[1][0], mat1[1][0]]])
    return
def getDer(net, results, y, costFunc):
    biasDer = []
    weightDer = []
    for i in range(len(results)-1):
        index = len(results) - i - 1
        if(i == 0):
            NodeOutputDer = list(map(costDer(costFunc),results[index][0], y))                                           # d(cost) / d(last node output)
        else:
            NodeOutputDer = flattern(matrixMul(net["weights"][index], shape(biasDer[0], [1, biasDer[0][1][0]])))[0]
        actDerList = [list(map(actDer(net["functionConfig"]), results[index][0])), results[index][1]]                   # d(ActivationInput) / d(ActivationOutput)
        biasDer.insert(0, matrixMul(actDerList, diag([NodeOutputDer, [len(NodeOutputDer)]])))                           # d(cost) / d(bias)

        tempWeightDer = []                                                                                                                                                                    
        for prevNode in results[index-1][0]:
            tempRowDer = []
            for nextNodeDer in biasDer[0][0]:                       
                tempRowDer.append(prevNode*nextNodeDer)                                                                 # [d(nextNode) / d(weight))] * [d(cost) / d(nextNode)]
            tempWeightDer.append(tempRowDer)
        weightDer.insert(0, [tempWeightDer, net["weights"][index-1][1]])                                                # d(cost) / d(weight)    
    return({"biasDer":biasDer, "weightDer":weightDer})
def adjustNet(net, der):
    for i in range(len(der["weightDer"])):
        weightDer = flattern(der["weightDer"][i])
        change = shape([list(map((lambda x : -0.001 if x > 0.2 else 0.01 if x < -0.2 else 0), weightDer[0])), weightDer[1]], der["weightDer"][i][1])
        net["weights"][i] = matrixAdd(change, net["weights"][i])
    for i in range(len(der["biasDer"])):
        change = [list(map((lambda x : 0 if x == 0 else 0.0001 if x < 0 else -0.0001), der["biasDer"][i][0])), der["biasDer"][i][1]]
        net["biases"][i] = matrixAdd(change, net["biases"][i])
def averageList(list):
    sum = 0
    for i in list:
        sum = sum + i
    return(sum/len(list))
def int2LayerOutput(value, length):
    retList = [0]*length
    retList[value] = 1
    return(retList)
def biggiestIntInList(list):
    x = 0
    y = 0
    for i in range(len(list)):
        if(list[i] > x):
            x = list[i]
            y = i
    return(y)
imgs = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = imgs.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

net = network([784,[15,6],10], ["random", -1,1], "sigmoid")
for i in range(len(x_train)):
    inputValue = []
    for i2 in x_train[i]:
        row = []
        for i3 in i2:
            row.append(i3)
        inputValue.append(row)
    results = runNetwork(flattern([inputValue, [28,28]]),net)
    
    y = int2LayerOutput(y_train[i], 10)
    cost = matCost(results[-1], y, "square")
    resultPrint = []
    for i4 in results[-1][0]:
        resultPrint.append("%.5f" % i4)

    print(averageList(cost[0][0]), "\t", resultPrint, "\t", y_train[i], "\t", biggiestIntInList(results[-1][0])) 

    
    c = getDer(net, results, y, "square")
    adjustNet(net, c)
  





jens = 2
