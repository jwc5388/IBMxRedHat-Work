import numpy as np

x1 = np.array([1,2,3])
print("x1=", x1.shape)     # x1 = (3,) 벡터 스칼라가 세개 짜리인 벡터라는 뜻


x2 = np.array([[1,2,3]]) #(1,3) 행렬
print("x2:", x2.shape)

x3 = np.array([[1,2], [3,4]])   #(2,2) 행렬
print("x3:", x3.shape)

x4 = np.array([[1,2],[3,4],[5,6]])  #(3,2)
print("x4:", x4.shape)


x5 = np.array([[[1,2],[3,4],[5,6]]]) #(1,3,2)
print("x5:", x5.shape)


x6 = np.array([[[1,2],[3,4]], [[5,6],[7,8]]]) #(2,2,2)


x7 = np.array([[[[1,2,3,4,5],[6,7,8,9,0]]]]) #(1,1,2,5)


x10 = np.array([[[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]]]) #(1,1,2,2,3)
print("x10:", x10.shape)