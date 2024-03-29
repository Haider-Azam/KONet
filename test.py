import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights,densenet121,DenseNet121_Weights
from torch.utils.data import DataLoader
import skorch
from skorch.helper import predefined_split
from skorch.callbacks import Checkpoint,Freezer
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
import warnings

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
if __name__=='__main__':
    warnings.filterwarnings("ignore")
    n_classes=2
    image_shape=224
    augmented_dataset_size=4000
    
    path="D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray Dataset"
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
    factor=augmented_dataset_size//len(dataset)
    new_dataset=torch.utils.data.ConcatDataset([non_augmented_dataset]+[dataset for _ in range(factor)])
    del non_augmented_dataset,dataset

    generator1 = torch.Generator().manual_seed(42)
    train_split=0.8
    valid_split=0.1
    test_split=0.1
    train_set,valid_set,test_set=torch.utils.data.random_split(new_dataset, [train_split,valid_split,test_split],
                                                                generator=generator1)


    model_name='conv_next_distilled'
    print('Model: ',model_name)
    #EfficientNetB0 has 16 MBConv layers, freeze till 8th MBConv layer then. Freeze all till before 5th sequential
    #DenseNet121 has 58 dense layers, freeze till 29th dense layer then. #Till before dense block 3
    if model_name=='efficient':
        model=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        p=0.1
        model.classifier[0]=torch.nn.Dropout(p=p,inplace=True)
        model.classifier[-1]=torch.nn.Linear(in_features=1280,out_features=n_classes)
        frozen_layers=4
        freeze=['features.{}*.weight'.format(i) for i in range(frozen_layers)] + ['features.{}*.bias'.format(i) for i in range(frozen_layers)]

    elif model_name=='dense':
        model=densenet121(weights=DenseNet121_Weights.DEFAULT)
        p=0.3
        model.classifier=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                            torch.nn.Linear(in_features=1024,out_features=n_classes),
                                            )
        freeze=['features.conv0.weight','features.conv0.bias','features.norm0.weight','features.norm0.bias',
                'features.denseblock1.*.weight','features.denseblock1.*.bias','features.denseblock2.*.weight','features.denseblock2.*.bias',
                ]
        freeze+=['features.denseblock3.denselayer{}.*.weight'.format(i) for i in range(1,12)]
        freeze+=['features.denseblock3.denselayer{}.*.bias'.format(i) for i in range(1,12)]

    elif model_name=='conv_next' or 'conv_next_distilled':
        p=0.3
        model=torchvision.models.convnext_tiny(weights='DEFAULT')
        model.classifier[2]=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                            torch.nn.Linear(in_features=768,out_features=n_classes),
                                            )
        frozen_layers=5
        freeze=['features.{}*.weight'.format(i) for i in range(frozen_layers)]
        freeze+=['features.{}*.bias'.format(i) for i in range(frozen_layers)]

        freeze=['features.5.{}*.weight'.format(i) for i in range(2)]
        freeze+=['features.5.{}*.bias'.format(i) for i in range(2)]

    elif model_name=='KONet':
        m1_ratio=0.6
        m2_ratio=0.4
        m1_dropout=0.1
        m2_dropout=0.3
        model=KONet(m1_ratio=m1_ratio,m2_ratio=m2_ratio,m1_dropout=m1_dropout,m2_dropout=m2_dropout,n_classes=n_classes)
        #Defines the blocks to be frozen
        m1_frozen_layers=4
        freeze=['efficient.features.{}*.weight'.format(i) for i in range(m1_frozen_layers)]
        freeze+=['efficient.features.{}*.bias'.format(i) for i in range(m1_frozen_layers)]

        freeze+=['dense.features.conv0.weight','dense.features.conv0.bias','dense.features.norm0.weight','dense.features.norm0.bias',
                'dense.features.denseblock1.*.weight','dense.features.denseblock1.*.bias','dense.features.denseblock2.*.weight',
                'dense.features.denseblock2.*.bias',
                ]
        freeze+=['dense.features.denseblock3.denselayer{}.*.weight'.format(i) for i in range(1,12)]
        freeze+=['dense.features.denseblock3.denselayer{}.*.bias'.format(i) for i in range(1,12)]


        monitor = lambda net: any(net.history[-1, ('valid_accuracy_best','valid_loss_best')])
    cp=Checkpoint(monitor='valid_loss_best',dirname='model',f_params=f'{model_name}best_param.pkl')
    cb = skorch.callbacks.Freezer(freeze)
    classifier = skorch.NeuralNetClassifier(
            model,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            train_split=predefined_split(valid_set),
            iterator_train=DataLoader,
            iterator_valid=DataLoader,
            iterator_train__shuffle=True,
            iterator_train__pin_memory=True,
            iterator_valid__pin_memory=True,
            iterator_train__num_workers=4 ,
            iterator_valid__num_workers=4,
            iterator_train__persistent_workers=True,
            iterator_valid__persistent_workers=True,
            batch_size=16,
            device='cuda',
            callbacks=[cp,cb,skorch.callbacks.ProgressBar()],#Try to implement accuracy and f1 score callables here
            warm_start=True,
            )

    classifier.initialize()
    classifier.load_params(f_params=f'model/{model_name}best_param.pkl')
    print("Paramters Loaded")

    iterations=5
    accuracy=[]
    f1=[]
    auc=[]
    test_loader=DataLoader(test_set,batch_size=8,shuffle=False)
    for _ in range(iterations):
        probs=[]
        actual_labels=[]
        for test_features, actual_lb in iter(test_loader):
            prob=classifier.predict_proba(test_features)
            actual_lb=np.array(actual_lb)
            probs.append(prob)
            actual_labels.append(actual_lb)

        probs=np.concatenate(probs)
        pred_labels=np.argmax(probs,axis=1)
        actual_labels=np.concatenate(actual_labels)

        iteration_auc=roc_auc_score(actual_labels,probs[:,1])
        iteration_accuracy=np.mean(pred_labels==actual_labels)
        iteration_f1=f1_score(actual_labels,pred_labels)

        accuracy.append(iteration_accuracy)
        f1.append(iteration_f1)
        auc.append(iteration_auc)

    print(model_name)

    print(f"Accuracy mean: {np.mean(accuracy)} standard deviation: {np.std(accuracy)}")
    print(f"F1-Score mean: {np.mean(f1)} standard deviation: {np.std(f1)}")
    print(f"ROC_AUC  mean: {np.mean(auc)} standard deviation: {np.std(auc)}")