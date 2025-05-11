import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('data.csv')

#initialise values 
b0=0
b1=0
alpha=0.01 #learning rate
err_limit=150
err=10000000 #initialise with a large value
iter=1 #to count the number of iterations

#repeat until error is less than error limit
while abs(err)>=err_limit:
    for x,y in zip(dataset['X'],dataset['Y']):
        err=b0+b1*x - y
        b0=b0-alpha*err
        b1=b1-alpha*err*x
        print(iter,err)
        iter+=1
        if(abs(err)<err_limit):
            break

dependant=[]
independant=[]
#store the data in a file to analyze the model
with open("pred_grad_descent.csv","a") as file:
    for x,y in zip(dataset['X'],dataset['Y']):
        dependant.append(x)
        independant.append(b0+b1*x)
        file.write(str(x)+", "+str(y)+", "+str(b0+b1*x) +"\n")

# plot the values for graphical analysis
for x,y in zip(dataset['X'],dataset['Y']) :      
    plt.scatter(x,y,color='red')
plt.plot(dependant,independant,color='blue')
plt.show()

print(b0,b1)
