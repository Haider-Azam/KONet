import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights,densenet121,DenseNet121_Weights
from torch.utils.data import DataLoader
from typing import Optional
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt  
import warnings
import random
import os
from sklearn.model_selection import KFold
import csv
script_name = os.path.basename(__file__)
results_file = "fold_results.csv"
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
    
class distiller(torch.nn.Module):
    def __init__(
            self,
            large_model,
            small_model
    ):
        super().__init__()
        self.large_model=large_model
        self.small_model=small_model

    def forward(self, x):
        large_output=self.large_model(x)
        small_output=self.small_model(x)
        return (large_output,small_output)
    

#Now we need to create our own loss function which will perform cross entropy loss
class distill_loss(_WeightedLoss):

    def __init__(self, weight= None, size_average=None, ignore_index= -100,
                 reduce=None, reduction= 'mean', label_smoothing= 0.0, T= 2,
                 soft_target_loss_weight= 0.25, ce_loss_weight= 0.75,):
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.T=T
        self.soft_target_loss_weight=soft_target_loss_weight
        self.ce_loss_weight=ce_loss_weight

    def forward(self, input, old_input, target):
        soft_targets = F.softmax(old_input / self.T, dim=-1)
        soft_prob = F.softmax(input / self.T, dim=-1)
        soft_targets_loss = -torch.sum(soft_targets * (soft_prob.log())) / soft_prob.size()[0] * (self.T**2)
        label_loss = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)   
        loss = self.soft_target_loss_weight * soft_targets_loss + self.ce_loss_weight * label_loss
        return label_loss, loss

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
   
if __name__=='__main__':
    warnings.filterwarnings("ignore")
    n_classes=2
    image_shape=224
    augmented_dataset_size=4000
    batch_size=4
    seed=42
    path="D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray modified\Osteoporosis Knee X-ray Preprocessed"

    set_random_seed(seed)
    
    # train_split=0.2
    # valid_split=0.1
    # test_split=0.7
    # train_set,valid_set,test_set=prep_dataset(path,image_shape,augmented_dataset_size
    #                                           ,train_split,valid_split,test_split)
    #train_set,valid_set,test_set=prep_dataset(path,image_shape,augmented_dataset_size)
    dataset,test_set=prep_dataset(path,image_shape,augmented_dataset_size)

    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    splits = list(kf.split(np.arange(len(dataset))))
    best_test_acc=0
    for fold, (train_indices, val_indices) in enumerate(splits):
        print(f"Fold {fold+1}")
        train_set = torch.utils.data.Subset(dataset, train_indices)
        valid_set = torch.utils.data.Subset(dataset, val_indices)
        print(f"Train set size: {len(train_set)}, Validation set size: {len(valid_set)}")
        # You can now use train_set and valid_set for training/validation in this fold
        
        train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=4, pin_memory=True,
                                    persistent_workers=True, shuffle=True)
        
        valid_dataloader = DataLoader(valid_set, batch_size=batch_size, num_workers=4, pin_memory=True,
                                    persistent_workers=True, shuffle=True)
        
        large_model_name='dense'
        small_model_name='mobilenet'
        #Large model initiallization

        if 'dense' in large_model_name:
            large_model=densenet121(weights=DenseNet121_Weights.DEFAULT)
            p=0.3
            large_model.classifier=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                                torch.nn.Linear(in_features=1024,out_features=n_classes),
                                                )
        elif 'KONet' in large_model_name:
            m1_ratio=0.6
            m2_ratio=0.4
            m1_dropout=0.1
            m2_dropout=0.3
            large_model=KONet(m1_ratio=m1_ratio,m2_ratio=m2_ratio,m1_dropout=m1_dropout,m2_dropout=m2_dropout,n_classes=n_classes)
        
        if small_model_name=='conv_next':
            p=0.3
            small_model=torchvision.models.convnext_tiny(weights='DEFAULT')
            small_model.classifier[2]=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                                torch.nn.Linear(in_features=768,out_features=n_classes),
                                                )
        elif small_model_name=='mobilenet':
            small_model=torchvision.models.mobilenet_v3_small(weights='DEFAULT')
            small_model.classifier[3]=torch.nn.Linear(in_features=1024,out_features=n_classes)

        #Load the pre-trained large model used for distilling
        large_model.load_state_dict(torch.load(f'model/{large_model_name}OtherFinetunedbest_param.pkl'))
        for name,param in large_model.named_parameters():
            param.requires_grad=False
        
        large_model.to(device)
        large_model.eval()
        small_model.to(device)

        loss_fn=distill_loss()
        test_loss_fn=torch.nn.CrossEntropyLoss()
        optimizer=torch.optim.AdamW(small_model.parameters(),lr=0.0000625)

        optimizer.zero_grad()

        epochs=20
        loss_results = []
        acc_results = []

        
        for i in range(epochs):
            print('Training')
            small_model.train()
            total_loss=0
            total_label_loss=0
            total_acc=0
            for train_data,train_label in tqdm(train_dataloader):
                
                train_data , train_label=train_data.to(device) , train_label.to(device)

                #Zero the gradient for every batch for mini-batch gradient descent
                optimizer.zero_grad()

                old_output=large_model(train_data)
                output=small_model(train_data)

                label_loss, loss=loss_fn(output, old_output, train_label)

                train_pred = output.softmax(dim=1).argmax(dim=1).detach().cpu().numpy()
                train_label_np = train_label.detach().cpu().numpy()

                total_acc+= np.mean(train_pred==train_label_np)

                total_loss+=loss.item()
                total_label_loss+=label_loss.item()

                loss.backward()
                optimizer.step()

            print('Testing on first dataset')
            small_model.eval()

            test_loss,labels,probabilities=test(small_model,valid_dataloader,test_loss_fn)
            pred_labels=np.argmax(probabilities,axis=1)
            test_acc=np.mean(pred_labels==labels)

            total_loss=total_loss/(len(train_dataloader))
            total_label_loss=total_label_loss/(len(train_dataloader))
            total_acc=total_acc/(len(train_dataloader))

            print('Train Epoch:',i+1,'loss:',total_loss)
            print('test loss1:',test_loss,'test accuracy1:',test_acc)

            if best_test_acc<test_acc:
                best_test_acc=test_acc
                print('Loss improved, saving weights')
                torch.save(small_model.state_dict(),f'model/{small_model_name}_distilledOtherFinetunedbest_param.pkl')
            loss_results.append((total_label_loss,test_loss))
            acc_results.append((total_acc,test_acc))

        plt.plot(loss_results)
        plt.xlabel("Epochs")
        plt.ylabel("Cross Entropy loss")
        plt.legend(['train loss','valid loss'])
        plt.title(f'{small_model_name} loss')
        plt.show()

        plt.plot(acc_results)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(['train accuracy','valid accuracy'])
        plt.title(f'{small_model_name} accuracy')
        plt.show()

    # Save best accuracy for this fold
    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['script', 'model', 'fold', 'best_accuracy'])
        writer.writerow([script_name, small_model_name, fold+1, best_test_acc])