import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights,densenet121,DenseNet121_Weights
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.nn import functional as F
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import os
import warnings
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
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
    batch_size=4
    path1="D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray Dataset Preprocessed"
    path2="D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray modified\Osteoporosis Knee X-ray Preprocessed"
    seed=42
    
    set_random_seed(seed)
    
    dataset1,test_set1=prep_dataset(path1,image_shape,augmented_dataset_size)
    dataset2,test_set2=prep_dataset(path2,image_shape,augmented_dataset_size)

    large_model_name='mobilenet_incremental_early_stop'
    #Large model initiallization

    if 'dense' in large_model_name:
        model=densenet121(weights=DenseNet121_Weights.DEFAULT)
        p=0.3
        model.classifier=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                            torch.nn.Linear(in_features=1024,out_features=n_classes),
                                            )
    elif 'KONet' in large_model_name:
        m1_ratio=0.6
        m2_ratio=0.4
        m1_dropout=0.1
        m2_dropout=0.3
        model=KONet(m1_ratio=m1_ratio,m2_ratio=m2_ratio,m1_dropout=m1_dropout,m2_dropout=m2_dropout,n_classes=n_classes)

    elif 'conv_next' in large_model_name:
        p=0.3
        model=torchvision.models.convnext_tiny(weights='DEFAULT')
        model.classifier[2]=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                            torch.nn.Linear(in_features=768,out_features=n_classes),
                                            )
    
    elif 'mobilenet' in large_model_name:
        model=torchvision.models.mobilenet_v3_small(weights='DEFAULT')
        model.classifier[3]=torch.nn.Linear(in_features=1024,out_features=n_classes)

    model.load_state_dict(torch.load(f'model/{large_model_name}best_param.pkl'))
    model.to(device)

    loss_fn=torch.nn.CrossEntropyLoss()
    
    test_dataloader1 = DataLoader(test_set1, batch_size=batch_size, num_workers=4, pin_memory=True,
                                   persistent_workers=True, shuffle=True)
    
    test_dataloader2 = DataLoader(test_set2, batch_size=batch_size, num_workers=4, pin_memory=True,
                                   persistent_workers=True, shuffle=True)

    print('Testing on dataset 1')
    iterations=5
    accuracy=[]
    f1=[]
    auc=[]
    labels=[]
    pred_labels=[]
    for i in range(iterations):
        model.eval()
    
        _,iteration_labels,probabilities=test(model,test_dataloader1,loss_fn)
        pred_iteration_labels=np.argmax(probabilities,axis=1)
        iteration_auc=roc_auc_score(iteration_labels,probabilities[:,1])
        iteration_accuracy=np.mean(pred_iteration_labels==iteration_labels)
        iteration_f1=f1_score(iteration_labels,pred_iteration_labels)

        labels.append(iteration_labels)
        pred_labels.append(pred_iteration_labels)
        accuracy.append(iteration_accuracy)
        f1.append(iteration_f1)
        auc.append(iteration_auc)

    print(f"Accuracy mean: {np.mean(accuracy)} standard deviation: {np.std(accuracy)}")
    print(f"F1-Score mean: {np.mean(f1)} standard deviation: {np.std(f1)}")
    print(f"ROC_AUC  mean: {np.mean(auc)} standard deviation: {np.std(auc)}")

    labels=np.concatenate(labels,axis=0)
    pred_labels=np.concatenate(pred_labels,axis=0)

    ConfusionMatrixDisplay.from_predictions(labels,pred_labels,display_labels=['normal','osteoporosis'],normalize='all')
    plt.show()
    print('Testing on dataset 2')

    accuracy=[]
    f1=[]
    auc=[]
    labels=[]
    pred_labels=[]
    for i in range(iterations):
        model.eval()
    
        _,iteration_labels,probabilities=test(model,test_dataloader2,loss_fn)
        pred_iteration_labels=np.argmax(probabilities,axis=1)
        
        iteration_auc=roc_auc_score(iteration_labels,probabilities[:,1])
        iteration_accuracy=np.mean(pred_iteration_labels==iteration_labels)
        iteration_f1=f1_score(iteration_labels,pred_iteration_labels)

        labels.append(iteration_labels)
        pred_labels.append(pred_iteration_labels)
        accuracy.append(iteration_accuracy)
        f1.append(iteration_f1)
        auc.append(iteration_auc)

    print(f"Accuracy mean: {np.mean(accuracy)} standard deviation: {np.std(accuracy)}")
    print(f"F1-Score mean: {np.mean(f1)} standard deviation: {np.std(f1)}")
    print(f"ROC_AUC  mean: {np.mean(auc)} standard deviation: {np.std(auc)}")

    labels=np.concatenate(labels,axis=0)
    pred_labels=np.concatenate(pred_labels,axis=0)

    ConfusionMatrixDisplay.from_predictions(labels,pred_labels,display_labels=['normal','osteoporosis'],normalize='all')
    plt.show()

    