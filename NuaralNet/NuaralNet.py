import random

def RandomMatrix(Dimentions, randmin, randmax):
    RetMat = []                                                 # Return Matrix
    ValueCount = 1;                                             # Total number of enteties in Matrix
    for i in Dimentions:                                             
        ValueCount = ValueCount * i
    for i in range(ValueCount):                                 # Creating list of random numbers with len of [ValueCount]
        RetMat.append(random.uniform(randmin, randmax))
    iMul = 1;
    for i in Dimentions:                                                 # Creating the dimentions
        TempMat = []
        iMul = iMul * i
        for i2 in range(int(ValueCount/iMul)):
            SubMat = []
            for i3 in range(i):
                SubMat.append(RetMat[i2*i+i3])
            TempMat.append(SubMat)
        RetMat = TempMat
    return(RetMat)                                              # Returning the Matrix



a = RandomMatrix((3,4,2), 0, 1)
b=a

