import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt

wot = '/path0/' # path to create your graph outputs
directory = '/path1/' # path where your financial data is stored

# for this example we used Unemployment data from FRED St. Louis, downloadable
# using PANDAS or manually. 

start_time = time.time()

data = open(directory,'r')
names = data.readline().split(',')[1:-1]

X = np.genfromtxt(directory,delimiter=',',skip_header=1)
X = X[:,1:-1] # remove date and Unemployment Rate
y = X[:,-1] # Unemployment Level
X = X[:,:-1] # remove Unemployment Level from predictors
print(X.shape)
print(len(y))
print(time.time() - start_time)



def plot(loss_list,y_r,y_p,i,e,p1,p2,p3,p4):
    plt.subplot(3, 1, 1)
    plt.cla()
    plt.ylim([0, .1])
    plt.plot(loss_list,color='red',label='error x200')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(y_r[:i],color='black',label='Unemp. Rate')
    plt.plot(y_p[:i],color='blue',label='Unsup NN Pred')
    plt.legend()


    plt.subplot(3, 1, 3)
    plt.plot(p1[:i],color='blue',label='w0')
    plt.plot(p2[:i],color='yellow',label='b0')
    plt.plot(p3[:i],color='green',label='w1')
    plt.plot(p4[:i],color='red',label='b1')
    plt.legend()

    plt.xlabel('Unsupervised Neural Network Prediction for Unemployment, Epoch:'+str(e))
    plt.draw()
    plt.pause(0.0001)
    plt.savefig(wot+'graf'+str(i))
    plt.clf()


#~~~~~~~~~~~~~~~~~ P R E   N N ~~~~~~~~~~~~~~~
#predictors = 2
#x2 = np.array(X[:,:predictors-1]) # Include an inverse mult, or a time-convolutional layer later :)
y2 = np.array(y) # or even combinations of different types of networks and cost fn's

#print(x2.shape)
'''
NEXT STEP IS TO MAKE IT PREDICTIVE AND MAKE COST RELATIVE TO TIME ADVANCED Y
LINEAR REGRESSION ON PARAMETERS
RBF
PCA
OTHER DATA AND ITS PARAMETER CHANGES
'''
y3 = [1]
for i in y:
    y3.append(y3[-1]+i*y3[-1])


x2 = np.zeros([5380])
epochs = 100
#factors = len(x2[0]) # 100
length = len(x2) # 5380
factors = 1
batch = 1
n1 = 1
n2 = 1
n3 = 1

x_placeholder = tf.placeholder(tf.float32,[batch])#, factors],name='hi1')
y_placeholder = tf.placeholder(tf.float32,[batch])#,name='hi2')
x_input = tf.unstack(x_placeholder)
y_input = tf.unstack(y_placeholder)

W01 = tf.Variable(np.random.rand(factors), dtype=tf.float32) 
b01 = tf.Variable(n1, dtype=tf.float32)
#W02 = tf.Variable(np.random.rand(n1,n2), dtype=tf.float32) 
#b02 = tf.Variable(n2, dtype=tf.float32)
#W03 = tf.Variable(np.random.rand(n2,n3), dtype=tf.float32) 
#b03 = tf.Variable(n3, dtype=tf.float32)
W04 = tf.Variable(np.random.rand(n3,1), dtype=tf.float32) 
b04 = tf.Variable(1, dtype=tf.float32)

layer1 = tf.tanh( tf.multiply( x_input,W01 ) + b01)
#layer2 = tf.tanh( tf.matmul( layer1, W02 ) + b02)
#layer3 = tf.tanh( tf.matmul( layer2, W03 ) + b03)
y_pred =  tf.multiply( layer1,W04) + b04

loss = tf.reduce_mean(tf.square(y_pred - y_input))
train_step = tf.train.AdagradOptimizer(1).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = [1]
    diff = [1]
    yp = [1]
    p1 = [0]
    p2 = [0]
    p3 = [0]
    p4 = [0]
    for e in range(epochs):
        for i in range(length-batch-1):
            mini_batchX = x2[i:i+batch] # 3x100
            mini_batchY = y2[i:i+batch] # 3x1
            _train_step, _y_pred ,_loss,w0,b0,w1,b1 = sess.run([ train_step,
                            y_pred, loss,W01,b01,W04,b04],
                feed_dict={ x_placeholder:mini_batchX,
                            y_placeholder:mini_batchY})
            loss_list.append(_loss*1000)
            pred = np.mean(_y_pred)
            yp.append((1+pred)*yp[-1])
            p1.append(w0[0])
            p2.append(b0)
            p3.append(w1[0])
            p4.append(b1)
            if i%100==0:
                plot(loss_list,y3,yp,i,e,p1,p2,p3,p4)

        


