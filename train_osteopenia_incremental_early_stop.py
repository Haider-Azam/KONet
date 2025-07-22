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
import random
import os
from tqdm import tqdm
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from typing import Tuple, List, Dict
import copy
from sklearn.model_selection import KFold
import csv
script_name = os.path.basename(__file__)
results_file = "fold_results.csv"
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

def set_random_seed(seed: int = 2222, deterministic: bool = False):
        """Set seeds"""
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = deterministic  # type: ignore

    
def prep_dataset(path,image_shape=224,augmented_dataset_size=4000,new_map=None
                 ,train_split=0.8,valid_split=0.1,test_split=0.1):
    global map
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
    return torch.utils.data.random_split(new_dataset, [train_split+valid_split,test_split],
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
    path1='D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray modified\Osteoporosis Knee X-ray Preprocessed'
    path2="D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray only osteopenia Preprocessed"

    new_n_classes=3
    map1={'normal':0,'osteoporosis':2}
    map2={'osteopenia':1}
    
    set_random_seed(seed)

    dataset1,test_set1=prep_dataset(path1,image_shape,augmented_dataset_size,map1)
    dataset2,test_set2=prep_dataset(path2,image_shape,augmented_dataset_size,map2)

    model_name='mobilenet'
    dataset1_bestfold=4

    kf1 = KFold(n_splits=10, shuffle=True, random_state=seed)
    splits1 = list(kf1.split(np.arange(len(dataset1))))

    train_indices, val_indices = splits1[dataset1_bestfold-1]
    train_set1 = torch.utils.data.Subset(dataset1, train_indices)
    valid_set1 = torch.utils.data.Subset(dataset1, val_indices)

    kf2 = KFold(n_splits=10, shuffle=True, random_state=seed)
    splits2 = list(kf2.split(np.arange(len(dataset2))))

    all_loss_results = []
    all_acc_results = []
    best_test_acc = 0
    for fold, (train_indices, val_indices) in enumerate(splits2):
        print(f"Fold {fold+1}")
        train_set2 = torch.utils.data.Subset(dataset2, train_indices)
        valid_set2 = torch.utils.data.Subset(dataset2, val_indices)
        print(f"Train set size: {len(train_set2)}, Validation set size: {len(valid_set2)}")
        # You can now use train_set and valid_set for training/validation in this fold

        train_dataloader1 = DataLoader(train_set1, batch_size=batch_size, num_workers=4, pin_memory=True,
                                    shuffle=True)
        
        train_dataloader2 = DataLoader(train_set2, batch_size=batch_size, num_workers=4, pin_memory=True,
                                    shuffle=True)
        
        valid_dataloader1 = DataLoader(valid_set1, batch_size=batch_size, num_workers=4, pin_memory=True,
                                    shuffle=True)
        
        valid_dataloader2 = DataLoader(valid_set2, batch_size=batch_size, num_workers=4, pin_memory=True,
                                    shuffle=True)
        
        test_dataloader1 = DataLoader(test_set1, batch_size=batch_size, num_workers=4, pin_memory=True,
                                    shuffle=True)
        
        test_dataloader2 = DataLoader(test_set2, batch_size=batch_size, num_workers=4, pin_memory=True,
                                    shuffle=True)
        
        #Large model initiallization

        if 'dense' in model_name:
            model=densenet121(weights=DenseNet121_Weights.DEFAULT)
            p=0.3
            in_features=model.classifier.in_features
            model.classifier=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                                torch.nn.Linear(in_features=in_features,out_features=n_classes),
                                                )
            
        elif 'conv_next' in model_name :
            p=0.3
            model=torchvision.models.convnext_tiny(weights='DEFAULT')
            in_features=model.classifier[2].in_features
            model.classifier[2]=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                                torch.nn.Linear(in_features=in_features,out_features=n_classes),
                                                )
            
        elif 'mobilenet' in model_name:
            model=torchvision.models.mobilenet_v3_small(weights='DEFAULT')
            in_features=model.classifier[3].in_features
            model.classifier[3]=torch.nn.Linear(in_features=in_features,out_features=n_classes)

        #Loads the model with the old classifier head, saves the weights of old classifier and transfers it to new_classifier
        model.load_state_dict(torch.load(f'model/{model_name}_otherbest_param.pkl'))

        #Copy the model before adding the new classifier head and freeze all the layers
        old_model=copy.deepcopy(model)
        for name,param in old_model.named_parameters():
            param.requires_grad=False

        if 'dense' in model_name :
            old_classifier=model.classifier.state_dict()
            model.classifier=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                                    torch.nn.Linear(in_features=in_features,out_features=new_n_classes),
                                                    )
            with torch.no_grad():
                for name,param in model.classifier.named_parameters():
                    param[[0,2]]=old_classifier[name]

        elif 'conv_next' in model_name :
            old_classifier=model.classifier[2].state_dict()
            model.classifier[2]=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                                    torch.nn.Linear(in_features=in_features,out_features=new_n_classes),
                                                    )
            with torch.no_grad():
                for name,param in model.classifier[2].named_parameters():
                    param[[0,2]]=old_classifier[name]

        elif 'mobilenet' in model_name:
            old_classifier=model.classifier[3].state_dict()
            model.classifier[3]=torch.nn.Linear(in_features=in_features,out_features=new_n_classes)
            with torch.no_grad():
                for name,param in model.classifier[3].named_parameters():
                    param[[0,2]]=old_classifier[name]

        model.to(device)
        old_model.to(device)

        loss_fn=torch.nn.CrossEntropyLoss()
        optimizer=torch.optim.SGD(model.parameters(),lr=0.000008,momentum=0.9, weight_decay=5e-4)#torch.optim.AdamW(model.parameters(),lr=0.0000625)

        model.train()
        optimizer.zero_grad()
        epochs=20
        loss_results = []
        acc_results = []
        print('Training on second dataset')
        
        patience = 2  # Number of epochs to wait for improvement
        best_val_acc = 0
        epochs_no_improve = 0
        best_model_state = None
        best_epoch = 0

        for i in range(epochs):
            print('Training')
            model.train()
            total_loss=0
            total_ce_loss=0

            for train_data,train_label in tqdm(train_dataloader2):
                train_data , train_label=train_data.to(device) , train_label.to(device)
                optimizer.zero_grad()
                output=model(train_data)
                ce_loss=loss_fn(output , train_label)
                total_loss+=ce_loss.item()
                ce_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_ce_loss+=ce_loss.item()

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

            total_ce_loss=total_ce_loss/(len(train_dataloader2))
            val_acc1=np.mean(accuracy1)
            val_acc2=np.mean(accuracy2)
            print('Train Epoch:',i+1,'ce loss:',total_ce_loss)
            print('val loss1:',loss1,'val accuracy1:',val_acc1,
                'val loss2:',loss2,'val accuracy2:',val_acc2)

            harmonic_acc=3/(2/val_acc1+1/val_acc2)
            loss_results.append((total_ce_loss,loss1,loss2))
            acc_results.append((harmonic_acc))

            # Early stopping logic
            if harmonic_acc > best_val_acc:
                best_val_acc = harmonic_acc
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = i + 1
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {i+1} for fold {fold+1}")
                    break

        # Restore best model state before testing
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Best model restored from epoch {best_epoch} for fold {fold+1}")
        model.eval()

        loss1,labels,probabilities=test(model,test_dataloader1,loss_fn)
        pred_labels=np.argmax(probabilities,axis=1)
        accuracy1=np.mean(pred_labels==labels)

        loss2,labels,probabilities=test(model,test_dataloader2,loss_fn)
        pred_labels=np.argmax(probabilities,axis=1)
        accuracy2=np.mean(pred_labels==labels)

        test_acc1=np.mean(accuracy1)
        test_acc2=np.mean(accuracy2)
        print('Test loss1:',loss1,'test accuracy1:',test_acc1,
            'test loss2:',loss2,'test accuracy2:',test_acc2)

        harmonic_test_acc=3/(2/test_acc1+1/test_acc2)

        if best_test_acc<harmonic_test_acc:
            best_test_acc=harmonic_test_acc
            print('Loss improved, saving weights')
            torch.save(model.state_dict(),f'model/{model_name}_incremental_early_stop_3_classbest_param.pkl')  
        # Save loss and accuracy results for this fold
        all_loss_results.append(loss_results)
        all_acc_results.append(acc_results)

        final_val_acc = acc_results[-1]

        # Save best accuracy for this fold
        with open(results_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['script', 'model', 'fold', 'valid_accuracy', 'test_accuracy'])
            writer.writerow([script_name, model_name, fold+1, final_val_acc, harmonic_test_acc])