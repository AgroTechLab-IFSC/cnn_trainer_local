## @file cnn.py
#  @author Wilson Castello Branco Neto (<mailto:wilson.castello@ifsc.edu.br>) / Robson Costa (<mailto:robson.costa@ifsc.edu.br>)
#  @brief CNN class.
#  @version 0.1.0
#  @since 06/12/2024
#  @date 09/12/2024
#  @copyright Copyright &copy; since 2024 <a href="https://agrotechlab.lages.ifsc.edu.br" target="_blank">AgroTechLab</a>.\n
#  ![LICENSE license](../figs/license.png)<br>
#  Licensed under the CC BY-NC-SA (<i>Creative Commons Attribution-NonCommercial-ShareAlike</i>) 4.0 International Unported License (the <em>"License"</em>). You may not
#  use this file except in compliance with the License. You may obtain a copy of the License <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode" target="_blank">here</a>.
#  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an <em>"as is" basis, 
#  without warranties or  conditions of any kind</em>, either express or implied. See the License for the specific language governing permissions 
#  and limitations under the License.
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import models

## CNN class.
#  @brief Train CCN models.
class CNN:
    ## @fn __init__
    #  @brief The CNN class initializer
    #  @param train_data Training data
    #  @param validation_data Validation data
    #  @param test_data Test data
    #  @param batch_size Batch size
    def __init__(self, train_data, validation_data, test_data, batch_size):
        
        ## Train data loader
        self.train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        ## Validation data loader
        self.validation_loader = data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)
        
        ## Test data loader
        self.test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        ## Trainer device type
        self.device = torch.device("cpu")

    ## @fn create_and_train_cnn
    #  @brief Create and train a CNN model
    #  @param model_name Model name
    #  @param num_epochs Number of epochs
    #  @param learning_rate Learning rate
    #  @param weight_decay Weight decay
    #  @param replications Number of replications
    #  @return Result name, average accuracy, maximum accuracy, iteration of maximum accuracy, duration
    def create_and_train_cnn(self, model_name, num_epochs, learning_rate, weight_decay, replications):
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
        
    ## @fn create_model
    #  @brief Create a model
    #  @param model_name Model name
    #  @return Model
    def create_model(self, model_name):
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
    
    ## @fn create_optimizer
    #  @brief Create an optimizer
    #  @param model Model
    #  @param learning_rate Learning rate
    #  @param weight_decay Weight decay
    #  @return Optimizer
    def create_optimizer(self, model, learning_rate, weight_decay):
        update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                update.append(param)
        optimizerSGD = optim.SGD(update, lr=learning_rate, weight_decay=weight_decay)
        return optimizerSGD

    ## @fn create_criterion
    #  @brief Create a loss criterion
    #  @return criterionCEL
    def create_criterion(self):
        criterionCEL = nn.CrossEntropyLoss()
        return criterionCEL

    ## @fn train_model
    #  @brief Train a model
    #  @param model Model
    #  @param train_loader Training data loader
    #  @param optimizer Optimizer
    #  @param criterion Loss criterion
    #  @param model_name Model name
    #  @param num_epochs Number of epochs
    #  @param learning_rate Learning rate
    #  @param weight_decay Weight decay
    #  @param replication Replication
    def train_model(self, model, train_loader, optimizer, criterion, model_name, num_epochs, learning_rate, weight_decay, replication): 
        model.to(self.device)
        min_loss = 100
        e_measures = []
        for i in (range(1,num_epochs+1)):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            if (train_loss < min_loss):
                min_loss = train_loss
                nome_arquivo = f"./models/{model_name}_{num_epochs}_{learning_rate}_{weight_decay}_{replication}.pth"
                torch.save(model.state_dict(), nome_arquivo)

    ## @fn train_epoch
    #  @brief Train an epoch
    #  @param model Model
    #  @param trainLoader Training data loader
    #  @param optimizer Optimizer
    #  @param criterion Loss criterion
    #  @return Mean of losses
    def train_epoch(self, model, trainLoader, optimizer, criterion):
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