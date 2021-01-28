import numpy as np
import matplotlib.pyplot as plt
###############define###############

def generate_linear(n=100):
  pts=np.random.uniform(0,1,(n,2))
  inputs=[]
  labels=[]
  for pt in pts:
    inputs.append([pt[0],pt[1]])
    #distance=(pt[0]-pt[1])/1.414
    if pt[0]>pt[1]:
      labels.append(0)
    else:
      labels.append(1)
  return np.array(inputs),np.array(labels).reshape(n,1)
  
def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i,1-0.1*i])
        labels.append(1)
    return np.array(inputs)*100,np.array(labels).reshape(21,1)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def layer(input_data,output_num,weight_matrix):
    #transform type of weight from array to list
    #Matrix : dim*output_num
    weight=[]
    for i in range(output_num):
        for j in range(len(input_data[0])):
            weight.append(weight_matrix[j][i])
    """
    input_data : dataset
    output_num : How many Z node we want to produe
    weight :  W1 or W2
    """    
    num_input_data,dim_input_data=len(input_data),len(input_data[0])
    output=[]
    for i in range(num_input_data):
        z=[]
        wight_index=0
        for j in range(output_num):
            dim_num=0
            z_sum=0 
            for k in range(dim_input_data):
                z_sum+=input_data[i][dim_num]*weight[wight_index]
                dim_num+=1
                wight_index+=1
            #get z[j]
            z.append(z_sum)    
        output.append(z)        
    return np.array(output)


def forward_passing(input_data,W1,W2,W3):
    a1=layer(input_data,4,W1)
    z1=sigmoid(a1)

    a2=layer(z1,4,W2)
    z2=sigmoid(a2)
    
    a3=layer(z2,1,W3)
    z3=sigmoid(a3)
    
    return a1,z1,a2,z2,a3,z3

def back_propagation(input_data,a1,z1,a2,z2,a3,y_pred,GT,W1,W2,W3):
    d3=(y_pred-GT)*derivative_sigmoid(a3)
    D3=np.matmul(z2.T,d3)
    
    d2=np.matmul(d3,W3.T)*derivative_sigmoid(a2)
    D2=np.matmul(z1.T,d2)
    
    d1=np.matmul(d2,W2.T)*derivative_sigmoid(a1)
    D1=np.matmul(input_data.T,d1)
    return D1,D2,D3

def calculate_loss(Ground_truth,y_pred): 
    loss_=0
    for i in range(len(Ground_truth)):
        loss_+=(y_pred[i][0]-Ground_truth[i][0])**2/2
    return float(loss_)

def show_result(x,y,pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground Truth',fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
            
    plt.subplot(1,2,2)
    plt.title('Predict result',fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i] < 0.5:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()
    #print('ground truth : ','\n',y)
    #print('predict y :','\n',pred_y)

def calculate_accuracy(y_pred,GT):
    miss=0
    for i in range(len(y_pred)):
        if y_pred[i][0]!=GT[i][0]:
            miss+=1
    acc=float((len(y_pred)-miss)/len(y_pred))
    return acc

def pred_label(y_pred):
    label_pred=[]
    for i in range(len(y_pred)):
        if y_pred[i][0]>0.5:
            label_pred.append(1)
        else:
            label_pred.append(0)
    return np.array(label_pred).reshape(len(y_pred),1)

###############data_generator###############  
"""
    Generate Training data
"""
data_linear,label_linear=generate_linear(100) #100*2
data_XOR,label_XOR=generate_XOR_easy()







##################################################problem2##################################################

###############Weight Setting###############
"""
   Matrix form: 
            W1:2*4
            W2:4*4
            W3:4*1
"""
p2_W1=np.random.randn(2,4)
p2_W2=np.random.randn(4,4)
p2_W3=np.random.randn(4,1)

###############forward passing ###############
import time
iteration=3000
learning_rate=0.1
Loss_change2=[]
print('##########################Loss of problem2 :########################## ')
ts = time.time()
for i in range(iteration) :
    """
        Forward passing:
              a1   : 100*4
              z1   : 100*4
              a2   : 100*4
              z2   : 100*4
              a3   : 100*1
       (z3) y_pred : 100*1
    """
    a1,z1,a2,z2,a3,y_pred=forward_passing(data_XOR,p2_W1,p2_W2,p2_W3)
    
    """
        Back probagation:
    """
    D1,D2,D3=back_propagation(data_XOR,a1,z1,a2,z2,a3,y_pred,label_XOR,p2_W1,p2_W2,p2_W3)
    """
        Update parameters W1,W2,W3
    """
    p2_W1 = p2_W1-learning_rate*D1
    p2_W2 = p2_W2-learning_rate*D2
    p2_W3 = p2_W3-learning_rate*D3
    """
        Calculate Loss:
    """
    loss=calculate_loss(label_XOR,y_pred)
    Loss_change2.append(loss)
    if i % 1000==0:
        print('epoch %d loss = %f'%(i,loss))
te = time.time()
print(te-ts,'sec')
#print(y_pred)
"""
    Show the result : 
"""
###############Comparison graph###############

show_result(data_XOR,label_XOR,y_pred)
pred_label_problem2=pred_label(y_pred)
accuracy2=calculate_accuracy(pred_label_problem2,label_XOR)

###############Accuracy###############

print('Accuracy  of problem2 = ',100*accuracy2,'%')

###############learning curve###############
plt.subplot(1,2,1)
plt.title('problem 1 loss')
plt.plot(100)
plt.subplot(1,2,2)
plt.title('problem 2 loss')
plt.plot(Loss_change2)
plt.show()