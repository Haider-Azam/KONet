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
import random
import os
import matplotlib.pyplot as plt  
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Only give first model output
def small_accuracy(net, ds, y=None):
    # assume ds yields (X, y), e.g. torchvision.datasets.MNIST
    y_true = [y for _, y in ds]
    y_pred = net.predict_proba(ds)[0]
    #y_pred=y_pred.argmax(axis=1)
    print(y_pred)
    return accuracy_score(y_true, y_pred)

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

def set_random_seed(seed: int = 2222, deterministic: bool = False):
        """Set seeds"""
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = deterministic  # type: ignore

def prep_dataset(path,image_shape=224,augmented_dataset_size=4000
                 ,train_split=0.8,valid_split=0.1,test_split=0.1):

    non_augment_transform=v2.Compose([v2.ToImageTensor(),
                        v2.ToDtype(torch.float32),
                        v2.Resize((image_shape,image_shape),antialias=True),
                        v2.Normalize(mean=[0.5], std=[0.5]),
                        ])
    transforms=v2.Compose([v2.ToImageTensor(),
                        v2.ToDtype(torch.float32),
                        v2.RandomAffine(degrees=30,shear=30),
                        v2.RandomZoomOut(side_range=(1,1.5)),
                        v2.Resize((image_shape,image_shape),antialias=True),
                        v2.Normalize(mean=[0.5], std=[0.5]),
                        ])
    non_augmented_dataset=torchvision.datasets.ImageFolder(path,transform=non_augment_transform)
    dataset=torchvision.datasets.ImageFolder(path,transform=transforms)
    factor=augmented_dataset_size//len(dataset)

    new_dataset=torch.utils.data.ConcatDataset([non_augmented_dataset]+[dataset for _ in range(factor)])
    del non_augmented_dataset,dataset

    
    #dataset=torchvision.datasets.ImageFolder(path,transform=transforms)
    generator1 = torch.Generator().manual_seed(42)

    #return torch.utils.data.random_split(new_dataset, [train_split,valid_split,test_split],
    #                                                            generator=generator1)
    return torch.utils.data.random_split(new_dataset, [train_split+valid_split,test_split],
                                                                generator=generator1)

def ewc_init(ewc_model,dataloader,loss_fn):
    model.train()
    optimizer.zero_grad()

    optpar_dict={}
    fisher_dict={}
    for train_data,train_label in tqdm(dataloader):
        train_data , train_label=train_data.to(device) , train_label.to(device)
        output=ewc_model(train_data)

        loss=loss_fn(output , train_label)
        loss.backward()

    #After iterating through dataset, save the parameters and the gradients squared in a dict
    for name , param in ewc_model.named_parameters():
        optpar_dict[name] = deepcopy(param)
        optpar_dict[name].requires_grad=False

        fisher_dict[name] = deepcopy(param.grad).pow(2)
        fisher_dict[name].requires_grad=False

    return ewc_model,optpar_dict,fisher_dict

def ewc_loss(ewc_model,optpar_dict,fisher_dict,ewc_lambda=8):
    distill_loss=0
    for name , param in ewc_model.named_parameters():
        if 'classifier' in name:
            optpar = optpar_dict[name]
            fisher = fisher_dict[name]

            distill_loss+= (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
    return distill_loss
 
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
    return loss.item(),labels,probabilities

if __name__=='__main__':
    
    warnings.filterwarnings("ignore")
    n_classes=2
    image_shape=224
    augmented_dataset_size=4000
    batch_size=4
    seed=42
    path1="D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray Dataset"
    path2="D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray modified\Osteoporosis Knee X-ray"
    
    set_random_seed(seed)
    
    train_set1,valid_set1,test_set1=prep_dataset(path1,image_shape,augmented_dataset_size)
    train_set2,valid_set2,test_set2=prep_dataset(path2,image_shape,augmented_dataset_size)

    train_dataloader1 = DataLoader(train_set1, batch_size=batch_size, num_workers=4, pin_memory=True,
                                   shuffle=True)
    
    train_dataloader2 = DataLoader(train_set2, batch_size=batch_size, num_workers=4, pin_memory=True,
                                   shuffle=True)
    
    valid_dataloader1 = DataLoader(valid_set1, batch_size=batch_size, num_workers=4, pin_memory=True,
                                   shuffle=True)
    
    valid_dataloader2 = DataLoader(valid_set2, batch_size=batch_size, num_workers=4, pin_memory=True,
                                   shuffle=True)
    
    model_name='conv_next_incremental'
    large_model_name='conv_next'
    #Large model initiallization

    if large_model_name=='dense':
        model=densenet121(weights=DenseNet121_Weights.DEFAULT)
        p=0.3
        model.classifier=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                            torch.nn.Linear(in_features=1024,out_features=n_classes),
                                            )
    elif large_model_name=='KONet':
        m1_ratio=0.6
        m2_ratio=0.4
        m1_dropout=0.1
        m2_dropout=0.3
        model=KONet(m1_ratio=m1_ratio,m2_ratio=m2_ratio,m1_dropout=m1_dropout,m2_dropout=m2_dropout,n_classes=n_classes)

    elif large_model_name=='conv_next' or large_model_name=='conv_next_distilled':
        p=0.3
        model=torchvision.models.convnext_tiny(weights='DEFAULT')
        model.classifier[2]=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                            torch.nn.Linear(in_features=768,out_features=n_classes),
                                            )
        
    elif 'mobilenet' in large_model_name:
        model=torchvision.models.mobilenet_v3_small(weights='DEFAULT')
        model.classifier[3]=torch.nn.Linear(in_features=1024,out_features=n_classes)

    model.load_state_dict(torch.load(f'model/{large_model_name}best_param.pkl'))
    #model=increte_model(model,)
    
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0000625)

    #After training, pass the dataset through the model to get the gradients of dataset
    model.train()
    optimizer.zero_grad()
    loss_fn=torch.nn.CrossEntropyLoss()

    print('Saving first dataset gradients and parameters')
    model,optpar_dict,fisher_dict=ewc_init(model,train_dataloader1,loss_fn)
    epochs=20
    ewc_lambda=16
    loss_results = []
    acc_results = []

    print('Training on second dataset')
    best_test_acc=0
    for i in range(epochs):
        print('Training')
        model.train()
        total_loss=0
        total_acc=0
        for train_data,train_label in tqdm(train_dataloader2):
            
            train_data , train_label=train_data.to(device) , train_label.to(device)

            #Zero the gradient for every batch for mini-batch gradient descent
            optimizer.zero_grad()

            output=model(train_data)

            train_pred = output.softmax(dim=1).argmax(dim=1).detach().cpu().numpy()
            train_label_np = train_label.detach().cpu().numpy()

            total_acc+= np.mean(train_pred==train_label_np)

            loss=loss_fn(output , train_label)

            distill_loss=ewc_loss(model,optpar_dict,fisher_dict,ewc_lambda=ewc_lambda)
            loss+=distill_loss

            total_loss+=loss.item()
            #print('distill loss:',distill_loss)
            #print('total loss:',loss)
            loss.backward()
            optimizer.step()

        print('Testing on first dataset')
        model.eval()

        loss1,labels,probabilities=test(model,valid_dataloader1,loss_fn)
        pred_labels=np.argmax(probabilities,axis=1)
        accuracy1=np.mean(pred_labels==labels)

        print('Testing on second dataset')
        model.eval()

        loss2,labels,probabilities=test(model,valid_dataloader2,loss_fn)
        pred_labels=np.argmax(probabilities,axis=1)
        accuracy2=np.mean(pred_labels==labels)

        total_loss=total_loss/(len(train_dataloader2))
        total_acc=total_acc/(len(train_dataloader2))
        test_acc1=np.mean(accuracy1)
        test_acc2=np.mean(accuracy2)
        print('Train Epoch:',i+1,'loss:',total_loss)
        print('test loss1:',loss1,'test accuracy1:',test_acc1,
              'test loss2:',loss2,'test accuracy2:',test_acc2)

        harmonic_acc=2*test_acc1*test_acc2/(test_acc1+test_acc2)
        if best_test_acc<harmonic_acc:
            best_test_acc=harmonic_acc
            print('Loss improved, saving weights')
            torch.save(model.state_dict(),f'model/{model_name}best_param.pkl') 
        loss_results.append((total_loss,loss1,loss2))
        acc_results.append((total_acc,test_acc1,test_acc2))
    plt.plot(loss_results)
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy loss")
    plt.legend(['train loss','dataset 1 valid loss','dataset 2a valid loss'])
    plt.title(f'{model_name} loss')
    plt.show()

    plt.plot(acc_results)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['train accuracy','dataset 1 valid accuracy','dataset 2a valid accuracy'])
    plt.title(f'{model_name} accuracy')
    plt.show()