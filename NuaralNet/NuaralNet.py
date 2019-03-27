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
    return(RetMat)                                              # Returning the Matrix

a = matrix((3,4,2), ("mono", 2))
b=a

