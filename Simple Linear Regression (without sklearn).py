import pandas as pd
#here instead of 'Salary_Data.csv' use your data file name
dataset=pd.read_csv('Salary_Data.csv')

mean_x = dataset['YearsExperience'].mean() # here YearsExperience is the x column name use what is in your case
mean_y = dataset['Salary'].mean() # similarly here Salary is y column name 
sum_mx_my=0

sum_mx_my=sum(((exp - mean_x) * (salary - mean_y)) for exp,salary in zip(dataset['YearsExperience'],dataset['Salary']))


mean_x_sq=0

mean_x_sq=sum(((exp-mean_x)**2) for exp in dataset['YearsExperience'])


b1=sum_mx_my/mean_x_sq
b0=mean_y-b1*mean_x
print(b0,b1)
with open("data_slr.txt","a") as file:
    for exp,salary in zip(dataset['YearsExperience'],dataset['Salary']):
        y_pred=b0+b1*exp
        text=f"{exp}, {salary}, {y_pred}"
        file.write(text+"\n")
