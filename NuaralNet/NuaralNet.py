import random
import math

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
    if(len(mat1[1]) >= len(mat2[1])):                                   # Finding highest dimention matrix and ajdusting the other one to fit it
        matDim = len(mat1[1])
        for i in range(matDim-len(mat2[1])):
            mat2[0] = [mat2[0]]
            mat2[1].append(1)
    else:
        matDim = len(mat2[1])
        for i in range(matDim-len(mat1[1])):
            mat1[0] = [mat1[0]]
            mat1[1].append(1)
    if(mat1[1][0] != mat2[1][1]):
        return

    if(matDim == 2):                                                    # Making sure the matrix is 2D
        RetMat = []
        for row1 in range(mat1[1][1]):                                  # Matrix multiplication 
            row = []
            for column2 in range(mat2[1][0]):
                sum = 0
                for column1 in range(mat1[1][0]):
                    sum = sum + mat1[0][row1][column1] * mat2[0][column1][column2]
                row.append(sum)
            RetMat.append(row)
        if(mat1[1][1] == 1):
            return([RetMat[0], [mat2[1][0]]])
        else:
            return([RetMat, (mat2[1][0], mat1[1][1])])
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
def costFunc(functionConfig):
    functions = {"abs": (lambda x, y: abs(x-y)), "square": (lambda x, y: (x-y)**2)}
    return(functions[functionConfig])
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
        return(shape([retMat, flatMat1[1]], mat1[1]))
    return


net = network([3,[5,3],2], ["random", -1,1], "sigmoid")
results = runNetwork([2,3,4],net)
cost = matCost(results[-1], [0.5,0.5], "square")

a = matrix([2,2], ["mono", 0.5])
b = matAct(a, "sigmoid")
jens = 2
