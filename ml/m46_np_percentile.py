import numpy as np
data  = [1,2,3,4,5]

print(np.percentile(data, 25))# 2.0


data  = [1,2,3,4]

print(np.percentile(data, 25))# 1.75



#(전체갯수-1) * (q / 100) = (4-1) * (25/100) 3 * 0.75 =  



"""
rank = (n-1) * (q/100)
(4-1) * (25/100)
 = 3* 0.25 = 0.75
보간법
작은값 = data의 0번쨰 = 10
큰값 = dat의 1번째 =20
백분위값 = 작은값 + (큰값 - 작은값)* rank
        = 10 + 10*0.75
        = 10 + 7.5
"""