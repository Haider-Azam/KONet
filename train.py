import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights,densenet121,DenseNet121_Weights
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
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
    path="D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray Dataset Preprocessed"

    set_random_seed(seed)

    #train_set,valid_set,test_set=prep_dataset(path,image_shape,augmented_dataset_size)
    dataset,test_set=prep_dataset(path,image_shape,augmented_dataset_size)

    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    splits = list(kf.split(np.arange(len(dataset))))
    all_loss_results = []
    all_acc_results = []
    best_test_acc = 0
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
        
        test_dataloader = DataLoader(test_set, batch_size=batch_size, num_workers=4, pin_memory=True,
                                     persistent_workers=True, shuffle=False)
        
        model_name='mobilenet'
        print('Model: ',model_name)
        #EfficientNetB0 has 16 MBConv layers, freeze till 8th MBConv layer then. Freeze all till before 5th sequential
        #DenseNet121 has 58 dense layers, freeze till 29th dense layer then. #Till before dense block 3
        if model_name=='efficient':
            model=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            p=0.1
            model.classifier[0]=torch.nn.Dropout(p=p,inplace=True)
            model.classifier[-1]=torch.nn.Linear(in_features=1280,out_features=n_classes)

        elif model_name=='dense':
            model=densenet121(weights=DenseNet121_Weights.DEFAULT)
            p=0.3
            model.classifier=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                                torch.nn.Linear(in_features=1024,out_features=n_classes),
                                                )

        elif model_name=='conv_next':
            p=0.3
            model=torchvision.models.convnext_tiny(weights='DEFAULT')
            model.classifier[2]=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                                torch.nn.Linear(in_features=768,out_features=n_classes),
                                                )

        elif model_name=='KONet':
            m1_ratio=0.6
            m2_ratio=0.4
            m1_dropout=0.1
            m2_dropout=0.3
            model=KONet(m1_ratio=m1_ratio,m2_ratio=m2_ratio,m1_dropout=m1_dropout,m2_dropout=m2_dropout,n_classes=n_classes)

        elif model_name=='mobilenet':
            model=torchvision.models.mobilenet_v3_small(weights='DEFAULT')
            model.classifier[3]=torch.nn.Linear(in_features=1024,out_features=n_classes)
            

        model.to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.0000625)

        model.train()
        optimizer.zero_grad()
        loss_fn=torch.nn.CrossEntropyLoss()

        epochs=20
        loss_results = []
        acc_results = []

        for i in range(epochs):
            print('Training')
            model.train()
            total_loss=0
            total_acc=0
            for train_data,train_label in tqdm(train_dataloader):
                
                train_data , train_label=train_data.to(device) , train_label.to(device)

                #Zero the gradient for every batch for mini-batch gradient descent
                optimizer.zero_grad()

                output=model(train_data)

                loss=loss_fn(output , train_label)

                train_pred = output.softmax(dim=1).argmax(dim=1).detach().cpu().numpy()
                train_label_np = train_label.detach().cpu().numpy()

                total_acc+= np.mean(train_pred==train_label_np)

                total_loss+=loss.item()

                loss.backward()
                optimizer.step()

            print('Testing on first dataset')
            model.eval()

            val_loss,labels,probabilities=test(model,valid_dataloader,loss_fn)
            pred_labels=np.argmax(probabilities,axis=1)
            val_acc=np.mean(pred_labels==labels)

            total_loss=total_loss/(len(train_dataloader))
            total_acc=total_acc/(len(train_dataloader))

            print('Train Epoch:',i+1,'loss:',total_loss)
            print('val loss:',val_loss,'val accuracy:',val_acc)

            loss_results.append((total_loss,val_loss))
            acc_results.append((total_acc,val_acc))

        model.eval()
        test_loss, test_labels, test_probabilities = test(model, test_dataloader, loss_fn)
        test_pred_labels = np.argmax(test_probabilities, axis=1)
        test_acc = np.mean(test_pred_labels == test_labels)
        print(f"Test set accuracy for fold {fold+1}: {test_acc}")

        if best_test_acc<test_acc:
            best_test_acc=test_acc
            print('Loss improved, saving weights')
            torch.save(model.state_dict(),f'model/{model_name}best_param.pkl') 

        
        # Save loss and accuracy results for this fold
        all_loss_results.append(loss_results)
        all_acc_results.append(acc_results)

        final_val_acc = acc_results[-1][1]

        # Save best accuracy for this fold
        with open(results_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['script', 'model', 'fold', 'valid_accuracy', 'test_accuracy'])
            writer.writerow([script_name, model_name, fold+1, final_val_acc, test_acc])


    # Plot all folds overlapped for loss (epochs 1 to 20)
    plt.figure()
    epochs_range = range(1, 21)
    for fold, loss_results in enumerate(all_loss_results):
        train_losses = [x[0] for x in loss_results]
        val_losses = [x[1] for x in loss_results]
        plt.plot(epochs_range, train_losses, label=f"Fold {fold+1} train", alpha=0.5)
        plt.plot(epochs_range, val_losses, label=f"Fold {fold+1} valid", linestyle='--', alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy loss")
    plt.title(f"{model_name} loss (all folds)")
    plt.xlim(1, 20)
    plt.xticks(epochs_range)
    plt.legend()
    plt.savefig(f"{script_name}_{model_name}_allfolds_loss.png")
    plt.show()

    # Plot all folds overlapped for accuracy (epochs 1 to 20)
    plt.figure()
    for fold, acc_results in enumerate(all_acc_results):
        train_accs = [x[0] for x in acc_results]
        val_accs = [x[1] for x in acc_results]
        plt.plot(epochs_range, train_accs, label=f"Fold {fold+1} train", alpha=0.5)
        plt.plot(epochs_range, val_accs, label=f"Fold {fold+1} valid", linestyle='--', alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} accuracy (all folds)")
    plt.xlim(1, 20)
    plt.xticks(epochs_range)
    plt.legend()
    plt.savefig(f"{script_name}_{model_name}_allfolds_acc.png")
    plt.show()

    