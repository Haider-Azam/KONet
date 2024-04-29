import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights,densenet121,DenseNet121_Weights
from torch.utils.data import DataLoader
import skorch
from skorch.helper import predefined_split
from skorch.callbacks import Checkpoint,Freezer,EpochScoring
from typing import Optional
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from skorch.callbacks import Callback
import warnings

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

#To implement incremental learning, we need to store the old model weights. Old and new model must be same.
class incrementer(torch.nn.Module):
    def __init__(
            self,
            old_model,
            new_model
    ):
        super().__init__()
        self.old_model=old_model
        self.new_model=new_model

    def forward(self, x):
        old_output=self.old_model(x)
        new_output=self.new_model(x)
        return (old_output,new_output)
    
class increte(Callback):
    def __init__(
            self,
        increment_period='batch'
    ):
        self.increment_period=increment_period

    def on_epoch_end(self, net, batch=None, training=None, **kwargs):
        #Loads the new weights onto the old model
        net.module_.old_model.load_state_dict(net.module_.new_model.state_dict())

def prep_dataset(path,image_shape=224,augmented_dataset_size=4000):

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
    train_split=0.8
    valid_split=0.1
    test_split=0.1
    return torch.utils.data.random_split(new_dataset, [train_split,valid_split,test_split],
                                                                generator=generator1)


#Now we need to create our own loss function which will perform cross entropy loss
class distill_loss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0, T:int = 2,
                 soft_target_loss_weight: float = 0.25, ce_loss_weight: float = 0.75,) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.T=T
        self.soft_target_loss_weight=soft_target_loss_weight
        self.ce_loss_weight=ce_loss_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        soft_targets = F.softmax(input[0] / self.T, dim=-1)
        soft_prob = F.softmax(input[1] / self.T, dim=-1)

        soft_targets_loss = -torch.sum(soft_targets * (soft_prob.log())) / soft_prob.size()[0] * (self.T**2)
        #print(input[1])
        label_loss = F.cross_entropy(input[1], target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)   
        loss = self.soft_target_loss_weight * soft_targets_loss + self.ce_loss_weight * label_loss
        return loss
    
    
if __name__=='__main__':
    warnings.filterwarnings("ignore")
    n_classes=2
    image_shape=224
    augmented_dataset_size=4000
    path1="D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray Dataset"
    path2="D:\Osteoporosis detection\datasets\Osteoporosis Knee X-ray modified\Osteoporosis Knee X-ray"
    
    train_set1,valid_set1,test_set1=prep_dataset(path1,image_shape,augmented_dataset_size)
    train_set2,valid_set2,test_set2=prep_dataset(path2,image_shape,augmented_dataset_size)

    model_name='incremental'
    large_model_name='dense'
    small_model_name='dense'
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
    
    if small_model_name=='conv_next':
        p=0.3
        small_model=torchvision.models.convnext_tiny(weights='DEFAULT')
        small_model.classifier[2]=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                            torch.nn.Linear(in_features=768,out_features=n_classes),
                                            )
    if small_model_name=='dense':
        small_model=densenet121(weights=DenseNet121_Weights.DEFAULT)
        p=0.3
        small_model.classifier=torch.nn.Sequential(torch.nn.Dropout(p=p,inplace=True),
                                            torch.nn.Linear(in_features=1024,out_features=n_classes),
                                            )
    #model=incrementer(old_model=model,new_model=small_model)
    #Freeze entirety of old_model so only new model changes
    freeze=['old_model.*.weight']
    #New weights loaded onto old model after each batch
    inc=increte()

    monitor = lambda net: any(net.history[-1, ('valid_accuracy_best','valid_loss_best')])
    cp=Checkpoint(monitor='valid_loss_best',dirname='model',f_params=f'{model_name}best_param.pkl',
                  f_optimizer=None,f_history=None)
    cb = skorch.callbacks.Freezer(freeze)

    classifier = skorch.NeuralNetClassifier(
            model,
            criterion=torch.nn.CrossEntropyLoss,#distill_loss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0000625,
            train_split=predefined_split(valid_set1),
            iterator_train=DataLoader,
            iterator_valid=DataLoader,
            iterator_train__shuffle=True,
            iterator_train__pin_memory=True,
            iterator_valid__pin_memory=True,
            iterator_train__num_workers=4 ,
            iterator_valid__num_workers=4,
            iterator_train__persistent_workers=True,
            iterator_valid__persistent_workers=True,
            batch_size=8,
            device='cuda',
            callbacks=[cp,cb,skorch.callbacks.ProgressBar(),
                       #,inc
                       ],#Try to implement accuracy and f1 score callables here
            warm_start=True,
            )
    
    epochs=30
    classifier.initialize()
    #Train on first dataset
    print('Training on first dataset')
    #classifier.fit(train_set1,y=None,epochs=epochs)

    #old_weights=classifier.module_.state_dict()
    #torch.save(old_weights, f'model/{model_name}_firstbest_param.pkl')
    #After training on one dataset, change the model and loss to include old model
    model=incrementer(old_model=model,new_model=small_model)
    classifier = skorch.NeuralNetClassifier(
            model,
            criterion=distill_loss(soft_target_loss_weight=0.5,ce_loss_weight=0.5),
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.0000625,
            train_split=predefined_split(valid_set1),
            iterator_train=DataLoader,
            iterator_valid=DataLoader,
            iterator_train__shuffle=True,
            iterator_train__pin_memory=True,
            iterator_valid__pin_memory=True,
            iterator_train__num_workers=4 ,
            iterator_valid__num_workers=4,
            iterator_train__persistent_workers=True,
            iterator_valid__persistent_workers=True,
            batch_size=8,
            device='cuda',
            callbacks=[cp,cb,skorch.callbacks.ProgressBar(),
                       #('valid_inc_acc',EpochScoring(small_accuracy,lower_is_better=False,name='valid_inc_acc')),
                       #,inc
                       ],#Try to implement accuracy and f1 score callables here
            warm_start=True,
            )
    classifier.initialize()
    #Loads the new weights onto the old model
    #classifier.module_.new_model.load_state_dict(torch.load(f'model/{model_name}best_param.pkl'))
    old_weights=torch.load(f'model/{model_name}_firstbest_param.pkl')
    classifier.module_.new_model.load_state_dict(old_weights)
    classifier.module_.old_model.load_state_dict(old_weights)

    #Then train on second dataset
    print('Training on second dataset')
    classifier.train_split=predefined_split(valid_set2)
    classifier.fit(train_set2,y=None,epochs=epochs)

    