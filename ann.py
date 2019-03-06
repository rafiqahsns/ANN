import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split as split

#Membbaca data
data = pd.read_csv('./iris.csv')

#Normalisasi Z score
sepall=[]
sepalw=[]
petall=[]
petalw=[]
for i in range(150):
    temp = ((data['SepalLengthCm'][i]-data['SepalLengthCm'].mean())/data['SepalLengthCm'].std())
    sepall.append(temp)
    temp = ((data['SepalWidthCm'][i]-data['SepalWidthCm'].mean())/data['SepalWidthCm'].std())
    sepalw.append(temp)
    temp = ((data['PetalLengthCm'][i]-data['PetalLengthCm'].mean())/data['PetalLengthCm'].std())
    petall.append(temp)
    temp = ((data['PetalWidthCm'][i]-data['PetalWidthCm'].mean())/data['PetalWidthCm'].std())
    petalw.append(temp)

#Normalisasi ke range [0,1] dengan min-max
for i in range(150):
    sepall[i]=((sepall[i]-min(sepall))/(max(sepall)-min(sepall)))
    sepalw[i]=((sepalw[i]-min(sepalw))/(max(sepalw)-min(sepalw)))
    petall[i]=((petall[i]-min(petall))/(max(petall)-min(petall)))
    petalw[i]=((petalw[i]-min(petalw))/(max(petalw)-min(petalw)))

#Membuat dataframe baru yang sudah ternormalisasi
newdata = pd.DataFrame(
    {'sepal_length': sepall,
     'sepal_width': sepalw,
     'petal_length': petall,
     'petal_width': petalw,
     'species': data['Species']
    })

#Memisahkan kelas dari data
X = newdata[list(newdata.columns)[:-1]]
Y = newdata['species']

#Perubahan kelas kategorik menjadi biner
tempo = pd.get_dummies(Y)
Y=pd.concat(objs=[Y,tempo],axis=1)

#Pembagian data training dan testing
X_train, X_test, Y_train, Y_test = split(X, Y, stratify=Y, test_size=0.5)
train=pd.concat(objs=[X_train,Y_train],axis=1)
test=pd.concat(objs=[X_test,Y_test],axis=1)

n=5
alpha=0.9
threshold = 0.05

w = []
v = []
v0 = []
w0 = []

#Menentukan bobot
for i in range(4):
    tempo = []
    for j in range(n):
        tempo.append(random.uniform(-0.5,0.5))
    v.append(tempo)

for i in range(n):
    tempo = []
    for j in range(3):
         tempo.append(random.uniform(-0.5,0.5))
    w.append(tempo)

for i in range(n):
    v0.append(random.uniform(-0.5,0.5))
for i in range(3):
    w0.append(random.uniform(-0.5,0.5))

#Fungsi aktivasi
def sigmoid(x):
    hasil = 1/(1+np.exp(-1*x))
    return hasil

def turunan(x):
    hasil = sigmoid(x)*(1-sigmoid(x))
    return hasil

var = (train.drop(columns=['species','Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))
target = (train.drop(columns=['sepal_length','sepal_width','petal_length','petal_width','species']))

for epoch in range(20):
    err = []
    for idx in range(75):
#Feedforward
        zin = []
        for j in range(n):
            zin.append(v0[j])
            for i in range(4):
                zin[j]=zin[j]+var.iloc[idx,i]*v[i][j]

        z = []

        for i in range(n):
            temp=sigmoid(zin[i])
            z.append(temp)

        yin = []

        for j in range(3):
            yin.append(w0[j])
            for i in range(n):
                yin[j]=yin[j]+z[i]*w[i][j]

        y = []

        for i in range(3):
            temp=sigmoid(yin[i])
            y.append(temp)

        y_error = []

        for i in range(3):
            y_error.append(target.iloc[idx,i]-y[i])


#Backpropagation

        dl = []

        for i in range(3):
            dl.append(y_error[i]*turunan(yin[i]))

        deltaw = []

        for i in range(n):
            tempo = []
            for j in range(3):
                tempo.append(alpha*dl[j]*z[i])
            deltaw.append(tempo)

        deltawo = []

        for i in range(3):
            deltawo.append(alpha*dl[i])

        dmin = []

        for i in range(n):
            dmin.append(0)
            for j in range(3):
                dmin[i]=dmin[i]+dl[j]*w[i][j]


        dm = []

        for i in range(n):
            dm.append(dmin[i]*turunan(zin[i]))

        deltav = []

        for i in range(4):
            tempo = []
            for j in range(n):
                tempo.append(alpha*dm[j]*var.iloc[idx,i])
            deltav.append(tempo)
        deltavo = []

        for i in range(n):
            deltavo.append(alpha*dm[i])

        for i in range(4):
            for j in range(n):
                v[i][j]=v[i][j]+deltav[i][j]


        for i in range(n):
            for j in range(3):
                w[i][j]=w[i][j]+deltaw[i][j]

        for i in range(n):
            v0[i]=v0[i]+deltavo[i]

        for i in range(3):
            w0[i]=w0[i]+deltawo[i]
#Menghitung MSE
        sum=0
        for i in range(3):
            sum+=y_error[i]**2

        err.append(sum/3)
#Menghitung MMSE
    mmse=np.mean(err)
    if(mmse<threshold):
        break


vart = (test.drop(columns=['species','Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))
targett = (test.drop(columns=['sepal_length','sepal_width','petal_length','petal_width','species']))

#Uji Akurasi
hasil = []
bener=0
for idxx in range(75):
    zin=[]
    for j in range(n):
        zin.append(v0[j])
        for i in range(4):
            zin[j]+=vart.iloc[idxx,i]*v[i][j]

    z = []
    for i in range(n):
        temp=sigmoid(zin[i])
        z.append(temp)
    yin = []

    for j in range(3):
        yin.append(w0[j])
        for i in range(n):
            yin[j]=yin[j]+z[i]*w[i][j]
    y= []
    for i in range(3):
        temp=sigmoid(yin[i])
        y.append(temp)

    if max(y) == y[0]:
        if('Iris-setosa'==test['species'].iloc[idxx]):
            bener+=1
    elif max(y) == y[1]:
        if 'Iris-versicolor'==test['species'].iloc[idxx]:
            bener+=1
    elif max(y) == y[2]:
        if 'Iris-virginica'==test['species'].iloc[idxx]:
            bener+=1

print('Akurasi:',round(bener/75*100),"%")

