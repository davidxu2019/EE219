import numpy as np

def scalar_to_one_hot(data):
    max_number=max(data)
    result=[]
    for number in data:
        temp=[0 for i in range(max_number)]
        temp[number-1]=1
        result.append(temp)
    return np.array(result)




