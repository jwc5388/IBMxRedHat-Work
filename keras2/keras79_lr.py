x = 10
y = 10
w =0.001
lr = 0.01
epochs = 1000



for i in range(epochs):
    hypothesis = x * w 
    loss = (hypothesis - y)**2 #bias, sigma 고려하지 않았다.
    print('Loss:', round(loss, 4), '\tPredict:', round(hypothesis, 4))
    up_predict  = x*(w +lr)
    up_loss = (y - up_predict)**2

    down_predict = x*(w-lr)
    down_loss = (y-down_predict)**2

    if(up_loss > down_loss):
        w = w -lr
    else:
        w = w +lr    

