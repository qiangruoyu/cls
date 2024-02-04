import torch
import os
import sys
import copy
import time
import matplotlib.pyplot as plt 
import numpy as np
from src.tool.test import *

def train_model(model_name, model, data_loaders, dataset_sizes, optimizer, criterion, device, num_epochs=5, scheduler=None):
    if not os.path.exists('weight'):
        os.mkdir('weight')
    if not os.path.exists('metrics'):
        os.mkdir('metrics')

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    time_start_train = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best = 0
    train_losses,train_acces,eval_losses,eval_acces = [],[],[],[]
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(data_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs,1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                iter_loss = loss.item() * inputs.size(0)
                running_loss += iter_loss
                running_corrects += torch.sum(preds == labels.data)

                print('Iteration: {}/{}, Loss: {}'.format(i+1,
                      len(data_loaders[phase]), iter_loss), end='')
                print("/n")
                # print((i + 1) * 100. / len(data_loaders[phase]), "% Complete")
                sys.stdout.flush()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc

            # print('[{}] Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())

        train_losses.append(train_loss)
        train_acces.append(train_acc)
        eval_losses.append(val_loss)
        eval_acces.append(val_acc)


        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        print()
        
        torch.save(model.state_dict(), 'weight/' +
                   str(model_name) + '_model_{}_{}_epoch.pt'.format(epoch+1,val_loss))

    time_elapsed = time.time() - time_start_train
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuracy: {}, Epoch: {}'.format(best_acc, best))
    draw_loss_acc(train_losses,train_acces,eval_losses,eval_acces,path="metrics",model_name=model_name)
    model.load_state_dict(best_model_wts)
    print()
    test_model(model_name, model, data_loaders, dataset_sizes, criterion, device, optimizer, phases=['test'])



    
def draw_loss_acc(train_losses,train_acces,eval_losses,eval_acces,path="metrics",model_name="cnn"):

    #绘图代码
    plt.plot(np.arange(len(train_losses)), train_losses,label="train loss")
    plt.plot(np.arange(len(train_acces)), train_acces, label="train acc")
    plt.plot(np.arange(len(eval_losses)), eval_losses, label="valid loss")
    plt.plot(np.arange(len(eval_acces)), eval_acces, label="valid acc")
    plt.legend() #显示图例
    plt.xlabel('epoches')
    plt.title('Model accuracy&loss')
    # plt.show()
    plt.savefig(os.path.join(path,model_name+".png"))


