import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import models

class CNN:
    """CNN Trainer class.
    
    This class is responsible for training a CNN model.

    Parameters:
        train_data (torchvision.datasets.ImageFolder): Training data.
        validation_data (torchvision.datasets.ImageFolder): Validation data.
        test_data (torchvision.datasets.ImageFolder): Test data.
        batch_size (int): Batch size.
    """
    
    def __init__(self, train_data, validation_data, test_data, batch_size):
        """The CNN Trainer class constructor."""
        
        # Train data loader
        self.train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # Validation data loader
        self.validation_loader = data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)
        
        # Test data loader
        self.test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        # Trainer device type
        self.device = torch.device("cpu")


    def create_and_train_cnn(self, model_name, num_epochs, learning_rate, weight_decay, replications):
        """Create and train a CNN model.
        
        Parameters:
            model_name (str): Model name to be trained.
            num_epochs (int): Number of epochs to be trained.
            learning_rate (float): Learning rate to be used at train.
            weight_decay (float): Weight decay to be used at train.
            replications (int): Number of replications used at each trained model.
        
        Returns:
            (dict): A dict mapping keys to the:
                * 'result_name': (str) Result name.
                * 'acc_avg': (float) Average accuracy.
                * 'iter_acc_max': (int) Iteration of maximum accuracy.
                * 'duration': (float) Duration of training.
        """
        begin = time.time()
        sum = 0
        acc_max = 0
        for i in range(0, replications):
            model = self.create_model(model_name)
            optimizerSGD = self.create_optimizer(model, learning_rate, weight_decay)
            criterionCEL = self.create_criterion()
            self.train_model(model, self.train_loader, optimizerSGD, criterionCEL, model_name, num_epochs, learning_rate, weight_decay, i) 
            acc = self.evaluate_model(model, self.validation_loader)
            sum = sum + acc
            if acc > acc_max:
                acc_max = acc
                iter_acc_max = i
        end = time.time()
        acc_avg = sum / replications
        duration = end - begin
        result_name = f"{model_name}-{num_epochs}-{learning_rate}-{weight_decay}"
        return result_name, acc_avg, iter_acc_max, duration
        

    def create_model(self, model_name):
        """Create a function to a CNN model to be trained.

        Note:
            At moment, the models available are: [VGG11, Alexnet, MobilenetV3Large].

        Parameters:
            model_name (str): CNN model name.
        
        Returns:
            (function): Function to CNN model selected.
        """
        if (model_name=='VGG11'):
            model = models.vgg11(weights='DEFAULT')  
            for param in model.parameters():
                param.requires_grad = False
            model.classifier[6] = nn.Linear(model.classifier[6].in_features,2)
            return model
        elif (model_name=='Alexnet'):
            model = models.alexnet(weights='DEFAULT')  
            for param in model.parameters():
                param.requires_grad = False
            model.classifier[6] = nn.Linear(model.classifier[6].in_features,2)
            return model
        else: # 'if (model_name=='MobilenetV3Large' ou qualquer outra coisa para não dar erro)
            model = models.mobilenet_v3_large(weights='DEFAULT')  
            for param in model.parameters():
                param.requires_grad = False
            model.classifier[3] = nn.Linear(model.classifier[3].in_features,2)
        return model
    

    def create_optimizer(self, model, learning_rate, weight_decay):
        """Create an optimizer.
        
        Parameters:
            model (function): CNN function.
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay
        
        Returns:
            (object): Optimizer object.
        """
        update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                update.append(param)
        optimizerSGD = optim.SGD(update, lr=learning_rate, weight_decay=weight_decay)
        return optimizerSGD

    
    def create_criterion(self):
        """Create a loss criterion.
        
        Parameters:
            None
        
        Returns:
            (object): Cross entropy loss object.
        """
        criterionCEL = nn.CrossEntropyLoss()
        return criterionCEL


    def train_model(self, model, train_loader, optimizer, criterion, model_name, num_epochs, learning_rate, weight_decay, replication):
        """Train a CNN model.

        Train a CNN model and save it (PTH file) at 'models' directory.
        
        Parameters:
            model (function): Model function.
            train_loader (DataLoader): Training data loader
            optimizer (object): Optimizer object.
            criterion (object): CEL object.
            model_name (str): Model name.
            num_epochs (int): Number of epochs.
            learning_rate (float): Learning rate.
            weight_decay (float): Weight decay.
            replication (int): Replication.
        
        Returns:
            None
        """
        model.to(self.device)
        min_loss = 100
        e_measures = []
        for i in (range(1,num_epochs+1)):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            if (train_loss < min_loss):
                min_loss = train_loss
                nome_arquivo = f"./models/{model_name}_{num_epochs}_{learning_rate}_{weight_decay}_{replication}.pth"
                torch.save(model.state_dict(), nome_arquivo)

    
    def train_epoch(self, model, trainLoader, optimizer, criterion):
        """Train an epoch.
        
        Parameters:
            model (function): Model function.
            trainLoader (DataLoader): Training data loader.
            optimizer (object): Optimizer object.
            criterion (object): CEL object.
        
        Returns:
            (float): Mean of losses.
        """
        model.train()
        losses = []
        for X, y in trainLoader:
            X = X.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        return np.mean(losses)

    ## @fn evaluate_model
    #  @brief Evaluate a model
    #  @param model Model
    #  @param loader Data loader
    #  @return Accuracy
    def evaluate_model(self, model, loader):
        """Evaluate a model.
        
        Parameters:
            model (function): Model function.
            loader (DataLoader): Data loader
        
        Returns:
            (float): Model (trained) accuracy.
        """
        total = 0
        correct = 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            output = model(X)
            _, y_pred = torch.max(output, 1)
            total += len(y)
            correct += (y_pred == y).sum().cpu().data.numpy()
        acc = correct/total
        return acc