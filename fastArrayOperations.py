import numpy as np

a = np.array([0,1,2])
print(a.dtype)

a = np.array([1,2,3], dtype='float32')
print(a.astype('float32'))

a = np.array([[0,1,2],[3,4,5]])
print(a)
print(a.shape)

a = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])


print(a.shape)
print(a.reshape(4,4))

print(np.zeros((3,3)))
print(np.empty((3,3)))
print(np.ones((3,3))) 

print(np.random.rand(3,3))


print(np.zeros_like(a))

A = np.array([0,1,2,3,4,5,6,7,8])
print(A[1])

print([a for a in A])

print(np.array(A[0:6]).reshape(2,3))

""" 
BroadCasting 
 """

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

print(A * B)


A = np.random.rand(5, 10, 2)
B = np.random.rand(5,2)

print(A * B[:, np.newaxis, :])



print(np.sqrt(np.array([4,9,16])))




a = np.random.rand(10000)
b = np.random.rand(10000)
c = np.random.rand(10000)
d = ne.evaluate('a + b * c')




""" PANDAS """


import pandas as pd

patients = [1,2,3,4]
effective = [True, True, False, False]

effective_series = pd.Series(effective, index=patients)

print(effective_series)



patients = ["a", "b", "c", "d"]

columns = {
    "sys_initial":[120, 126, 130, 115],
    "dia_initial":[75, 85, 90, 87],
    "sys_final":[115,123,130,118],
    "dia_final":[70,82,92,87]
}

df = pd.DataFrame(columns, index=patients)
print(df)

""" Indexing Series and DataFrame objects """
# print(effective_series.iloc[0])



""" Database-style operations with Pandas """

a  = pd.Series([1, 2, 3], index=["a", "b", "c"])
b =  pd.Series([4, 5, 6], index=["a", "b", "c"])
print(a + b)

b = pd.Series([4,None,5,None,6], index=["a", "b", "c" ,"d","e"])
print(b)

a = pd.Series([1,2,3], index=["a","b","c"])

def superstar(x):
    return '*' + str(x) + "->"
print(a.map(superstar))



""" Grouping, aggregations, and transforms """

patients = ["a","b","c","d","e","f"]


columns = {
    "sys_initial": [120, 126, 130, 115, 150, 117],
    "dia_initial": [75, 85, 90, 87, 90, 74],
    "sys_final": [115, 123, 130, 118, 130, 121],
    "dia_final": [70, 82, 92, 87, 85, 74],
    "drug_admst": [True, True, True, True, False, False],
}

df = pd.DataFrame(columns, index=patients)

# print(df)

# print(df.loc['a'])

hospitals = pd.DataFrame(
    {   "name" : ["City 1", "City 2", "City 3"],
        "address" : ["Address 1", "Address 2", "Address 3"],
        "city": ["City 1", "City 2", "City 3"] 
    },index=["H1", "H2", "H3"])
hospital_id = ["H1", "H2", "H2", "H3", "H3", "H3"]
df['hospital_id'] = hospital_id

print(df)

