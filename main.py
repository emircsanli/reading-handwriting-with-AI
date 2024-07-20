import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml  #dataseti yükleme
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

mist=fetch_openml("mnist_784")
#print(mist)

def showImage(df,index):
    some_digit=df.to_numpy()[index]
    some_digit_image=some_digit.reshape(28,28)

    plt.imshow(some_digit_image,cmap='binary')
    plt.axis("off")
    plt.show()
#showImage(mist.data,1)

train_img,test_img,train_lbl,test_lbl=train_test_split(mist.data,mist.target,test_size=1/7,random_state=0)
test_img_copy=test_img.copy()

scaler=StandardScaler()
scaler.fit(train_img)
train_img=scaler.transform(train_img)
test_img=scaler.transform(test_img)

pca=PCA(.95)
pca.fit(train_img)
print(pca.n_components_)
train_img=pca.transform(train_img)
test_img=pca.transform(test_img)

logisticRegression=LogisticRegression(max_iter=10000)
logisticRegression.fit(train_img,train_lbl)
logisticRegression.predict(test_img[0].reshape(1,-1))
showImage(test_img_copy,0)
logisticRegression.predict(test_img[2].reshape(1,-1))
showImage(test_img_copy,2)
score=logisticRegression.score(test_img,test_lbl)
#print(score)