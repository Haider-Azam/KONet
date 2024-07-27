import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights,densenet121,DenseNet121_Weights
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
import warnings
from tqdm import tqdm
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KONet(torch.nn.Module):

    def __init__(
            self,
            m1_ratio=0.6,
            m2_ratio=0.4,
            m1_dropout=0.1,
            m2_dropout=0.3,
            n_classes=2
    ):
        super().__init__()
        assert m1_ratio+m2_ratio==1
        self.n_classes=n_classes
        self.m1_ratio=m1_ratio
        self.m2_ratio=m2_ratio
        self.m1_dropout=m1_dropout
        self.m2_dropout=m2_dropout

        self.efficient=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.efficient.classifier[0]=torch.nn.Dropout(p=self.m1_dropout,inplace=True)
        self.efficient.classifier[-1]=torch.nn.Linear(in_features=1280,out_features=self.n_classes)

        self.dense=densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.dense.classifier=torch.nn.Sequential(torch.nn.Dropout(p=self.m2_dropout,inplace=True),
                                            torch.nn.Linear(in_features=1024,out_features=n_classes),
                                            )

    def forward(self, x):
        m1=self.efficient(x)
        m2=self.dense(x)
        out=self.m1_ratio*m1+self.m2_ratio*m2
        return out


def prep_dataset(path,image_shape=224,augmented_dataset_size=4000
                 ,train_split=0.8,valid_split=0.1,test_split=0.1):

    non_augment_transform=v2.Compose([v2.ToImageTensor(),
                        v2.ToDtype(torch.float32),
                        v2.Resize((image_shape,image_shape),antialias=True),
                        v2.Normalize(mean=[0],std=[1]),
                        ])
    transforms=v2.Compose([v2.ToImageTensor(),
                        v2.ToDtype(torch.float32),
                        v2.RandomAffine(degrees=30,shear=30),
                        v2.RandomZoomOut(side_range=(1,1.5)),
                        v2.Resize((image_shape,image_shape),antialias=True),
                        v2.Normalize(mean=[0],std=[1]),
                        ])
    non_augmented_dataset=torchvision.datasets.ImageFolder(path,transform=non_augment_transform)
    dataset=torchvision.datasets.ImageFolder(path,transform=transforms)
    factor=augmented_dataset_size//len(dataset)-1

    new_dataset=torch.utils.data.ConcatDataset([non_augmented_dataset]+[dataset for _ in range(factor)])
    del non_augmented_dataset,dataset
    
    #dataset=torchvision.datasets.ImageFolder(path,transform=transforms)
    generator1 = torch.Generator().manual_seed(42)
    return torch.utils.data.random_split(new_dataset, [train_split,valid_split,test_split],
                                                                generator=generator1)
        

def test(model,dataloader,loss_fn):
    model.eval()
    loss=0
    labels=[]
    probabilities=[]
    for data,label in tqdm(dataloader):
        with torch.no_grad():
            data , label=data.to(device) , label.to(device)

            output=model(data)
            #print(output.shape)
            loss+=loss_fn(output , label)
            prob=output.softmax(dim=1)
            labels.append(label.detach().cpu().numpy())
            probabilities.append(prob.detach().cpu().numpy())

    labels=np.concatenate(labels,axis=0)
    probabilities=np.concatenate(probabilities,axis=0)

    loss=loss/len(dataloader)
    return loss,labels,probabilities

if __name__=='__main__':
    
    warnings.filterwarnings("ignore")
    n_classes=2
    image_shape=224
    augmented_dataset_size=4000

    path2="D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray modified 3 class"

    new_n_classes=3


    train_set2,valid_set2,test_set2=prep_dataset(path2,image_shape,augmented_dataset_size)

    
    train_dataloader2 = DataLoader(train_set2, batch_size=8, num_workers=4, pin_memory=True,
                                   persistent_workers=True, shuffle=True)
    
    valid_dataloader1 = DataLoader(valid_set2, batch_size=8, num_workers=4, pin_memory=True,
                                   persistent_workers=True, shuffle=True)
    
    
    model_name='dense_3_class'
    large_model_name='denseOtherFinetuned'
    #Large model initiallization

    if large_model_name=='dense' or large_model_name=='denseOtherFinetuned':
        model=densenet121(weights=DenseNet121_Weights.DEFAULT)
        p=0.3
        model.classifier=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                            torch.nn.Linear(in_features=1024,out_features=n_classes),
                                            )
    #Loads the model with the old classifier head, saves the weights of old classifier and transfers it to new_classifier
    model.load_state_dict(torch.load(f'model/{large_model_name}best_param.pkl'))
    old_classifier=model.classifier.state_dict()

    model.classifier=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                            torch.nn.Linear(in_features=1024,out_features=new_n_classes),
                                            )
    
    #For generalization, use a sorted dict to hold class idxs and convert values into a list on indexes
    with torch.no_grad():
        for name,param in model.classifier.named_parameters():
            param[[0,2]]=old_classifier[name]

    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0000625)

    model.train()
    optimizer.zero_grad()
    loss_fn=torch.nn.CrossEntropyLoss()

    epochs=30

    print('Training on second dataset')
    best_test_acc=0
    for i in range(epochs):
        print('Training')
        model.train()
        total_loss=0
        for train_data,train_label in tqdm(train_dataloader2):
            
            train_data , train_label=train_data.to(device) , train_label.to(device)

            #Zero the gradient for every batch for mini-batch gradient descent
            optimizer.zero_grad()

            output=model(train_data)

            loss=loss_fn(output , train_label)

            total_loss+=loss

            loss.backward()
            optimizer.step()

        print('Testing on first dataset')
        model.eval()

        loss1,labels,probabilities=test(model,valid_dataloader1,loss_fn)
        pred_labels=np.argmax(probabilities,axis=1)
        accuracy1=np.mean(pred_labels==labels)


        total_loss=total_loss/(len(train_dataloader2))
        test_acc1=np.mean(accuracy1)

        print('Train Epoch:',i+1,'loss:',total_loss.item())
        print('test loss1:',loss1.item(),'test accuracy1:',test_acc1)

        if best_test_acc<test_acc1:
            best_test_acc=test_acc1
            print('Loss improved, saving weights')
            torch.save(model.state_dict(),f'model/{model_name}best_param.pkl') 
    #loss,accuracy=test(model,test_dataloader2,loss_fn)
    #print('loss:',loss,'\nAccuracy:',accuracy)
    #torch.save(model.state_dict(),f'model/{model_name}best_param.pkl')