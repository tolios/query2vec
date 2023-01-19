import time
import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch.optim import SGD
from graph import *
from torch.utils.tensorboard import SummaryWriter
from utils import save, load

def training(model: torch.nn.Module,
    train: Dataset, val: Dataset, writer: SummaryWriter,
    epochs = 50, batch_size = 1024, val_batch_size = 1024,
    lr = 0.1, weight_decay = 0.0005, patience = -1):
    '''
    Iplementation of training. Receives embedding model, dataset of training and val data!.
    Returns trained model, training losses, Uncorrupted and Corrupted energies.
    '''
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=val_batch_size, shuffle=True)
    #optimizers
    optimizer = SGD(model.parameters(), lr = lr, weight_decay = weight_decay)
    #training begins...
    t_start = time.time()
    #for early stopping!
    #start with huge number!
    highest_val_score = -1e5
    stop_counter = 1
    epoch_stop = 0 #keeps track of last epoch of checkpoint...
    #getting initial weights...
    model_dict = model.state_dict()
    for model_part in model_dict:
        writer.add_histogram(model_part, model_dict[model_part], 0)
    #training ...
    print('Training begins ...')
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_score = 0.0
        running_corr_score = 0.0
        # #perform normalizations before entering the mini-batch.
        # model.normalize()
        for i, qa_batch in enumerate(train_loader):
            batch, answers = qa_batch
            #get corrupted triples
            corrupted = corrupted_answer(model.num_entities, answers.size(), start = 1)
            #calculate loss...
            loss, score, corr_score = model(batch, answers, corrupted)
            #zero out gradients...
            optimizer.zero_grad()
            #loss backward
            loss.sum().backward()
            #update parameters!
            optimizer.step()
            #getting losses...
            running_loss += loss.mean().data.item()
            running_score += score.mean().data.item()
            running_corr_score += corr_score.mean().data.item()
        #calculating val energy....
        with torch.no_grad():
            running_val_score = 0.0
            for j, qa_batch in enumerate(val_loader):
                #questions and answers!
                batch, answers = qa_batch
                #calculate validation scores!!!
                running_val_score += model.predict(batch, answers).mean().data.item()
        #print results...
        print('Epoch: ', epoch, ', loss: ', "{:.4f}".format(running_loss/i),
            ', score: ', "{:.4f}".format(running_score/i),
            ', corrupted score: ', "{:.4f}".format(running_corr_score/i),
            ', val_score: ', "{:.4f}".format(running_val_score/j),
            ', time: ', "{:.4f}".format((time.time()-t_start)/60), 'min(s)')
        #collecting loss and scores!
        writer.add_scalar('Loss', running_loss/i, epoch)
        writer.add_scalar('Golden score', running_score/i, epoch)
        writer.add_scalar('Corrupted score', running_corr_score/i, epoch)
        writer.add_scalar('Val score', running_val_score/j, epoch)
        #collecting model weights!
        model_dict = model.state_dict()
        for model_part in model_dict:
            writer.add_histogram(model_part, model_dict[model_part], epoch)

        #implementation of early stop using val_energy (fastest route (could use mean_rank for example))
        if patience != -1:
            if highest_val_score <= running_val_score/j:
                #setting new score!
                highest_val_score = running_val_score/j
                #save model checkpoint...
                model.save('./checkpoint.pth.tar')
                epoch_stop = epoch
                #"zero out" counter
                stop_counter = 1
            else:
                if stop_counter >= patience:
                    #no more patience...early stopping!
                    print('Early stopping at epoch:', epoch)
                    print('Loading from epoch:', epoch_stop)
                    #load model from previous checkpoint!
                    model, _ = load('./checkpoint.pth.tar', model.__class__)
                    break
                else:
                    #be patient...
                    stop_counter += 1
                    #if in the end of training load from checkpoint!
                    if epoch == epochs:
                        print('Finished during early stopping...')
                        print('Loading from epoch:', epoch_stop)
                        #load model from previous checkpoint!
                        model, _ = load('./checkpoint.pth.tar', model.__class__)
        else:
            epoch_stop = epochs
    ## If checkpoint exists, delete it ##
    if os.path.isfile('./checkpoint.pth.tar'):
        os.remove('./checkpoint.pth.tar')

    print('Training ends ...')
    #returning model as well as writer and actual last epoch (early stopping)...
    return model, writer, epoch_stop