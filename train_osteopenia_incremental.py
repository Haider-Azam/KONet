import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2,ToTensor,Lambda
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights,densenet121,DenseNet121_Weights
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
import warnings
from tqdm import tqdm
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from typing import Tuple, List, Dict
import copy

map=None
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
    
class CustomImageFolder(torchvision.datasets.ImageFolder):
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Override this method to load from setting file instead of scanning directory
        """
        self.map=map
        if map is not None:
            classes = list(self.map.keys())
            classes_to_idx = self.map
        else:
            classes, classes_to_idx=super().find_classes(directory)
        return classes, classes_to_idx

def prep_dataset(path,image_shape=224,augmented_dataset_size=4000,new_map=None):
    global map
    #mapping=Lambda(lambda x: ToTensor(map[x.item()]))
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
    map=new_map
    non_augmented_dataset=CustomImageFolder(path,transform=non_augment_transform)
    dataset=CustomImageFolder(path,transform=transforms)
    factor=augmented_dataset_size//len(dataset)-1

    print(dataset.__getitem__(0)[1])
    print(dataset.class_to_idx)
    new_dataset=torch.utils.data.ConcatDataset([non_augmented_dataset]+[dataset for _ in range(factor)])
    del non_augmented_dataset,dataset

    
    #dataset=torchvision.datasets.ImageFolder(path,transform=transforms)
    generator1 = torch.Generator().manual_seed(42)
    train_split=0.8
    valid_split=0.1
    test_split=0.1
    return torch.utils.data.random_split(new_dataset, [train_split,valid_split,test_split],
                                                                generator=generator1)

def ewc_init(ewc_model,dataloader,loss_fn,optimizer):
    ewc_model.train()
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
    return loss,labels,probabilities

def distillation_loss(new_logits,old_logits,T=2):
	outputs = torch.log_softmax(new_logits/T, dim=1)   # compute the log of softmax values
	labels = torch.softmax(old_logits/T, dim=1)
	# print('outputs: ', outputs)
	# print('labels: ', labels.shape)
	outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
	outputs = -torch.mean(outputs, dim=0, keepdim=False)
	# print('OUT: ', outputs)
	return outputs

def replace_head(model,n_new_classes):
    old_head=copy.deepcopy(list(model.parameters())[-1])
    in_features=old_head.in_features

if __name__=='__main__':
    
    warnings.filterwarnings("ignore")
    n_classes=2
    image_shape=224
    augmented_dataset_size=4000

    path1='D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray modified\Osteoporosis Knee X-ray'
    path2="D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray only osteopenia"
    path3="D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray modified 3 class"

    new_n_classes=3
    map1={'normal':0,'osteoporosis':2}
    map2={'osteopenia':1}
    
    train_set1,valid_set1,test_set1=prep_dataset(path1,image_shape,augmented_dataset_size,map1)
    train_set2,valid_set2,test_set2=prep_dataset(path2,image_shape,augmented_dataset_size,map2)
    train_set3,valid_set3,test_set3=prep_dataset(path3,image_shape,augmented_dataset_size)

    train_dataloader1 = DataLoader(train_set1, batch_size=8, num_workers=4, pin_memory=True,
                                   persistent_workers=True, shuffle=True)
    
    train_dataloader2 = DataLoader(train_set2, batch_size=8, num_workers=4, pin_memory=True,
                                   persistent_workers=True, shuffle=True)
    
    valid_dataloader1 = DataLoader(valid_set1, batch_size=8, num_workers=4, pin_memory=True,
                                   persistent_workers=True, shuffle=True)
    
    valid_dataloader3 = DataLoader(valid_set3, batch_size=8, num_workers=4, pin_memory=True,
                                   persistent_workers=True, shuffle=True)
    
    
    model_name='conv_next_distilled_incremental_lwf_3_class'
    large_model_name='conv_next_distilled_other'
    #Large model initiallization

    if large_model_name=='dense' or large_model_name=='denseOtherFinetuned':
        model=densenet121(weights=DenseNet121_Weights.DEFAULT)
        p=0.3
        in_features=model.classifier[1].in_features
        model.classifier=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                            torch.nn.Linear(in_features=in_features,out_features=n_classes),
                                            )
        
    elif 'conv_next' in large_model_name :
        p=0.3
        model=torchvision.models.convnext_tiny(weights='DEFAULT')
        in_features=model.classifier[2].in_features
        model.classifier[2]=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                            torch.nn.Linear(in_features=in_features,out_features=n_classes),
                                            )
    #Loads the model with the old classifier head, saves the weights of old classifier and transfers it to new_classifier
    model.load_state_dict(torch.load(f'model/{large_model_name}best_param.pkl'))

    #Copy the model before adding the new classifier head and freeze all the layers
    old_model=copy.deepcopy(model)
    for name,param in old_model.named_parameters():
        param.requires_grad=False
    old_classifier=model.classifier[2].state_dict()

    model.classifier[2]=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                            torch.nn.Linear(in_features=in_features,out_features=new_n_classes),
                                            )
    
    #For later generalization, use a sorted dict to hold class idxs and convert values into a list on indexes
    with torch.no_grad():
        for name,param in model.classifier[2].named_parameters():
            param[[0,2]]=old_classifier[name]
            #if 'weight' in name:
            #    param[1]=0
            #elif 'bias' in name:
            #    param[1]=-3

    model.to(device)
    old_model.to(device)

    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(),lr=0.00001,momentum=0.9, weight_decay=5e-4)
    print('Extracting first dataset gradients')
    model,optpar_dict,fisher_dict=ewc_init(model,train_dataloader1,loss_fn,optimizer)

    model.train()
    optimizer.zero_grad()
    #for name,param in model.named_parameters():
    #    if 'classifier' not in name:
    #        param.requires_grad=False

    print('Testing on old dataset')
    loss1,labels,probabilities=test(model,valid_dataloader1,loss_fn)
    pred_labels=np.argmax(probabilities,axis=1)
    accuracy1=np.mean(pred_labels==labels)
    print('binary class accuracy',accuracy1)
    ewc_lambda=1000
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

            #Get probabilities from old model and corresponding probabilities from new model
            old_output=old_model(train_data)
            new_output=output[:,[0,2]]
            
            distill_loss=ewc_loss(model,optpar_dict,fisher_dict,ewc_lambda=ewc_lambda)

            distill_loss=+distillation_loss(new_output,old_output)*ewc_lambda
            
            loss+=distill_loss

            total_loss+=loss.item()

            loss.backward()
            optimizer.step()

        print('Testing on 3 class dataset valid split')
        model.eval()
        print('distill loss:',distill_loss)
        loss1,labels,probabilities=test(model,valid_dataloader3,loss_fn)
        pred_labels=np.argmax(probabilities,axis=1)
        accuracy_binary=np.mean(pred_labels[labels!=1]==labels[labels!=1])
        accuracy_osteopenia=np.mean(pred_labels[labels==1]==labels[labels==1])


        total_loss=total_loss/(len(train_dataloader2))
        test_acc1=2*accuracy_binary*accuracy_osteopenia/(accuracy_osteopenia+accuracy_binary)

        print('Train Epoch:',i+1,'loss:',total_loss)
        print('test loss1:',loss1.item(),'test accuracy harmonic mean',test_acc1)
        print('test binary accuracy:',accuracy_binary,'test osteopenia accuracy:',accuracy_osteopenia)

        if best_test_acc<test_acc1:
            best_test_acc=test_acc1
            print('Loss improved, saving weights')
            torch.save(model.state_dict(),f'model/{model_name}best_param.pkl') 
    #loss,accuracy=test(model,test_dataloader2,loss_fn)
    #print('loss:',loss,'\nAccuracy:',accuracy)
    torch.save(model.state_dict(),f'model/{model_name}best_param.pkl')