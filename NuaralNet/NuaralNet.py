import random

def matrix(Dimentions, config):
    type = config[0]
    RetMat = []                                                 # Return Matrix
    ValueCount = 1;                                             # Total number of enteties in Matrix
    for i in Dimentions:                                             
        ValueCount = ValueCount * i
    for i in range(ValueCount):                                 # Creating list of random numbers with len of [ValueCount]
        if(type.lower() == "random"):
            RetMat.append(random.uniform(config[1], config[2]))
        elif(type.lower() == "mono"):
            RetMat.append(config[1])
    iMul = 1;
    for i in Dimentions:                                        # Creating the dimentions
        TempMat = []
        iMul = iMul * i
        for i2 in range(int(ValueCount/iMul)):
            SubMat = []
            for i3 in range(i):
                SubMat.append(RetMat[i2*i+i3])
            TempMat.append(SubMat)
        RetMat = TempMat
        RetMat.append(Dimentions)
    return(RetMat)                                           # Returning the Matrix

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

input = matrix([16], ("mono", 2))
input_h1 = matrix([32,16], ("random", -1, 1))

visualizeMatrix(input)
visualizeMatrix(input_h1)

c = matrixMul(input,input_h1)
if(c != 0):
    visualizeMatrix(c)