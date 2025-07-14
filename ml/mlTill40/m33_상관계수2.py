import pandas as pd

df = pd.DataFrame({
    'x1' : [1,2,3,4,5],
    'x2' : [5,4,3,2,1],
    'x3' : [-1000, 50, 7, 100, -1100],
    'y' : [2,3,4,5,6],
})

print(df)


correlations = df.corr()
print(correlations)