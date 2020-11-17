import os
import math
import argparse
import pydicom
from pydicom import dcmread
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.naive_bayes import BernoulliNB
from openpyxl import Workbook
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision import transforms, utils
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import itertools

# Please tune the hyperparameters
parser = argparse.ArgumentParser()
#parser.add_argument("--train", default=1, type=int)
parser.add_argument("--train_dir", default="./data/TrainingData/")
parser.add_argument("--test_dir", default="./data/TestingData/")
parser.add_argument("--training_img", default="./data/TrainingImg/")
parser.add_argument("--testing_img", default="./data/TestingImg/")
args = parser.parse_args()
category=['epidural','healthy','intraparenchymal','intraventricular','subarachnoid','subdural']
fileList = []



class Preprocess(object):
    def __init__(self,train=1,category=category,dirPath=args.train_dir):
        self.dirPath=dirPath
        self.category=category
        self.train=train
        

    def start(self):
        if self.train==1:
            for i in range(len(category)):
                allFiles=os.listdir(self.dirPath+category[i])
                allFiles.sort()
                for fileName in allFiles:
                    filePath=os.path.join(self.dirPath+category[i],fileName)
                    
                    img_dicom = dcmread(filePath)
                    img_id=img_dicom.SOPInstanceUID

                    #fileList[i].append(img_id)
                    img=img_dicom.pixel_array
                    metadata = self.get_metadata_from_dicom(img_dicom)
                    img = self.window_image(img_dicom.pixel_array, **metadata)
                    img = self.normalize_minmax(img) * 255
                    img = self.resize(img,256,256)
                    #plt.imshow(img,cmap='bone')
                    #plt.show()
                    self.save_img(img,args.training_img+category[i]+'/',img_id)
        elif self.train==0:
                allFiles=os.listdir(self.dirPath)
                allFiles.sort()
                for fileName in allFiles:
                    filePath=os.path.join(self.dirPath,fileName)
                    
                    img_dicom = dcmread(filePath)
                    img_id=img_dicom.SOPInstanceUID
                    fileName_split=fileName.split(".")[0]
                    fileList.append(fileName_split)
                    img=img_dicom.pixel_array
                    metadata = self.get_metadata_from_dicom(img_dicom)
                    img = self.fix(img, **metadata)
                    img = self.window_image(img, **metadata)
                    img = self.normalize_minmax(img) * 255
                    img = self.resize(img,256,256)
                    #plt.imshow(img,cmap='bone')
                    #plt.show()
                    self.save_img(img,args.testing_img,fileName_split)           
                
    def get_first_of_dicom_field_as_int(self,x):
        if type(x) == pydicom.multival.MultiValue:
            return int(x[0])
        return int(x)

    def get_metadata_from_dicom(self,img_dicom):
        metadata = {
            "window_center": img_dicom.WindowCenter,
            "window_width": img_dicom.WindowWidth,
            "intercept": img_dicom.RescaleIntercept,
            "slope": img_dicom.RescaleSlope,
            "bits": img_dicom.BitsStored,
            "pixel" : img_dicom.PixelRepresentation,
        }
        return {k: self.get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}

    def fix(self, img, window_center, window_width, intercept, slope, bits, pixel):
    # In some cases intercept value is wrong and can be fixed
    # Ref. https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
        if bits == 12 and pixel == 0 and intercept > -100:
            image = image.copy() + 1000
            px_mode = 4096
            image[image>=px_mode] = image[image>=px_mode] - px_mode
            intercept = -1000
        return image.astype(np.float32)


    def window_image(self, img, window_center, window_width, intercept, slope, bits, pixel):
        img = img * slope + intercept
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img[img < img_min] = img_min
        img[img > img_max] = img_max
        return img 

    def resize(self,img, new_w, new_h):
        img = Image.fromarray(img.astype(np.int8), mode="L")
        #return img.resize((new_w, new_h), resample=Image.BICUBIC)
        return img.resize((new_w, new_h))

    def save_img(self,img_pil, subfolder, name):
        img_pil.save(subfolder+name+'.png')

    def normalize_minmax(self,img):
        mi, ma = img.min(), img.max()
        if mi==ma:
            return img * 0
        return (img - mi) / (ma - mi)

def sklearn_train():
    NB = GaussianNB()
    logreg1 = LogisticRegression(C=1)
    logreg10 = LogisticRegression(C=10,max_iter=100)
    logreg100 = LogisticRegression(C=100)
    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn8 = KNeighborsClassifier(n_neighbors=8)
    svm_model = OneVsRestClassifier(svm.LinearSVC(dual=False))


    j=0
    for i in range(len(category)):
        print(i)
        allFiles=os.listdir(args.training_img+category[i])
        allFiles.sort()
        for fileName in allFiles:
            filePath=os.path.join(args.training_img+category[i],fileName)
            img=Image.open(filePath)
            img=np.array(img).reshape(1,-1)
            if j==0:
                x_train=img
                y_train=np.array(i)
                j=j+1
            else: 
                x_train=np.vstack((x_train,img))
                y_train=np.hstack((y_train,np.array(i)))

    x_train,x_val,y_train,y_val= train_test_split(x_train, y_train ,test_size=0.1, random_state = 20, shuffle=True)
    '''    NB.fit(x_train,y_train)
    score=NB.score(x_val,y_val)
    y_pred=NB.predict(x_val)
    print("Validation set score: {:.3f}".format(score))
    plt.figure()
    cnf_matrix = confusion_matrix(y_val, y_pred)

    plot_confusion_matrix(cnf_matrix, classes=category ,normalize=False, title="Gaussian Naive Bayes confusion matrix")

    plt.show()
    '''
    svm_model.fit(x_train,y_train)
    score=svm_model.score(x_val,y_val)
    y_pred=svm_model.predict(x_val)
    print("Validation set score: {:.3f}".format(score))
    plt.figure()
    cnf_matrix = confusion_matrix(y_val, y_pred)

    plot_confusion_matrix(cnf_matrix, classes=category ,normalize=False, title="SVM confusion matrix")

    plt.show()

    logreg1.fit(x_train,y_train)
    score=logreg1.score(x_val,y_val)
    y_pred=logreg1.predict(x_val)
    print("Validation set score: {:.3f}".format(score))
    plt.figure()
    cnf_matrix = confusion_matrix(y_val, y_pred)

    plot_confusion_matrix(cnf_matrix, classes=category ,normalize=False, title="Logistic regression confusion matrix")

    plt.show()

'''
    allFiles=os.listdir(args.testing_img)
    allFiles.sort()
    j=0
    for fileName in allFiles:
        filePath=os.path.join(args.testing_img,fileName)
        img=Image.open(filePath)
        img=np.array(img).reshape(1,-1)
        if j==0:
            x_test=img
            j=j+1
        else: 
            x_test=np.vstack((x_test,img))
    y_pred=NB.predict(x_test)
    print(y_pred.shape)
    print(y_pred)

    wb = Workbook()
    sheet=wb.active
    i=0
    for row in sheet.iter_rows(min_row=1, min_col=1, max_row=len(fileList), max_col=2):
        j=0
        for cell in row:
            if j==0:
                cell.value=fileList[i]
            else:
                cell.value=category[y_pred[i]]
            j=j+1
        i=i+1
        print(i)
    wb.save("testing_submission_trail1_CT_G11.xlsx")
'''
def pytorch_learn():

    transform1 = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform2 = transforms.Compose([
        transforms.Normalize((0.5),(0.5))
    ])

    j=0
    for i in range(len(category)):
        print(i)
        allFiles=os.listdir(args.training_img+category[i])
        allFiles.sort()
        for fileName in allFiles:
            filePath=os.path.join(args.training_img+category[i],fileName)
            img=Image.open(filePath)
            img_t=transform1(img)
            img_t=torch.unsqueeze(img_t,dim=0)
            if j==0:
                x=img_t
                y=np.array(i)
                j=j+1
            else: 
                x=torch.cat((x,img_t),0)
                y=np.hstack((y,np.array(i)))
    y=torch.from_numpy(y)
    print(x.size(),y.size())
    x_train,x_val,y_train,y_val= train_test_split(x, y ,test_size=0.1, random_state = 20, shuffle=True)

    EPOCH=1000
    LR=0.001
    BATCH_SIZE=50
    INPUT_SIZE = 256

    torch_dataset = Data.TensorDataset(x_train,y_train)
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()
            self.rnn = nn.LSTM(      #nn.RNN   nn.LSTM    nn.GRU
                input_size=INPUT_SIZE,
                hidden_size =20,
                num_layers=2,
                batch_first=True
            )
            self.out = nn.Linear(20, 6)
            self.sigmoid = nn.Sigmoid()

        def forward(self,x):
            r_out,_=self.rnn(x,None)
            out=self.out(r_out[:,-1,:])
            out=self.sigmoid(out)
            return out

    rnn=RNN()

    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all parameters
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

    ac_train=[]
    ac_test=[]
    y_pred=np.zeros(600)

    for epoch in range(EPOCH):
        print(epoch)
        for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data
            b_x = b_x.view(-1, INPUT_SIZE, INPUT_SIZE)   # reshape x to (batch, time_step, input_size)

            output = rnn(b_x)               # rnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

        '''
        with torch.no_grad():
            #training accuracy
            hit=0
            for i in range(x_train.size()[0]):
                outputs=rnn(x_train[i].view(-1,256,256))  
                _, pred=torch.max(outputs.data,1)
                if pred==y_train[i]:
                    hit+=1  
            accuracy=hit/x_train.size()[0]    
            ac_train.append(accuracy)
            #testing
            hit=0
            for i in range(x_test.size()[0]):
                outputs=rnn(x_test[i].view(-1,256,256))  
                _, pred=torch.max(outputs.data,1)
                if pred==y_test[i]:
                    hit+=1  
            accuracy=hit/x_test.size()[0]    
            ac_test.append(accuracy)
        '''
        
        if epoch%50==49 and step==95:
            with torch.no_grad():
                #testing
                hit=0
                for i in range(x_val.size()[0]):
                    outputs=rnn(x_val[i].view(-1,INPUT_SIZE,INPUT_SIZE))  
                    _, pred=torch.max(outputs.data,1)
                    y_pred[i]=pred
                    if pred==y_val[i]:
                        hit+=1  
                accuracy=hit/x_val.size()[0]    
                print(accuracy)
            x_torch=x_val.numpy()
            y_torch=y_val.numpy()
            plt.figure()
            cnf_matrix = confusion_matrix(y_val, y_pred)

            plot_confusion_matrix(cnf_matrix, classes=category ,normalize=False, title="LSTM confusion matrix")

            plt.show()


            
def resnet():
    img_data = torchvision.datasets.ImageFolder('data/TrainingImg',
                                                transform=transforms.Compose([
                                                    transforms.Scale(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor()])
                                                )
    train,test = torch.utils.data.random_split(img_data,[5400,600])
    data_loader = torch.utils.data.DataLoader(train, batch_size=8,shuffle=True)
    testLoader = torch.utils.data.DataLoader(test, batch_size=8,shuffle=True)

    top1_score=[]

    net = models.resnet101(pretrained=True)
    
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    epochs = 5
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,20,25,30], gamma=0.5)

    def acc():
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testLoader:
                inputs, labels = data
                inputs, labels = inputs, labels
                outputs = net(inputs)
                _, predicted = torch.topk(outputs.data, 1)
                total += labels.size(0)
                c=[]
                for i in range(8):
                    c.append(int(labels[i] in predicted[i]))
                correct += sum(c)
                top1_score.append(correct/total)
            print('Accuracy top1 of the network on the 10000 test inputs: %d %%' % (100 * correct / total))

    print('\nTrain start')
    for epoch in range(epochs):
        running_loss = 0.0

        for times, data in enumerate(data_loader , 0):
            inputs, labels = data
            inputs, labels = inputs, labels
            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if times % 2000 == 99 or times+1 == len(data_loader):
                print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, times+1, len(data_loader), running_loss/2000))
        acc()
        #scheduler.step()
    print('Finished Training')

    torch.save(net.state_dict(), 'params.pkl')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def main():
    #pre_train=Preprocess(train=1,category=category,dirPath=args.train_dir)
    #pre_train.start()
    print("Training data preprocessing is over.")
    #pre_test=Preprocess(train=0,category=category,dirPath=args.test_dir)
    #pre_test.start()
    print("Testing data preprocessing is over.")


    #sklearn_train()
    pytorch_learn()
    #resnet()




if __name__ == "__main__" :
    main()

