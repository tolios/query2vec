import time
import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch.optim import Adagrad
from graph import *
from utils import save_checkpoint, load_checkpoint
from mlflow import log_metrics

def training(model: torch.nn.Module, optimizer_dict:dict,
    train: Dataset, val: Dataset,
    epochs = 50, batch_size = 1024, val_batch_size = 1024,
    lr = 0.1, weight_decay = 0.0005, patience = -1, pretrained=False, device=torch.device('cpu')):
    '''
    Iplementation of training. Receives embedding model, dataset of training and val data!.
    Returns trained model, training losses, Uncorrupted and Corrupted energies.
    '''
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=val_batch_size, shuffle=True)
    #put in device...
    model.to(device)
    #optimizers
    optimizer = Adagrad(model.parameters(), lr = lr, weight_decay = weight_decay)
    if pretrained:
        #load also optimizer state!
        optimizer.load_state_dict(optimizer_dict)
    #training begins...
    t_start = time.time()
    #for early stopping!
    #start with huge number!
    highest_val_score = -1e5
    stop_counter = 1
    epoch_stop = 0 #keeps track of last epoch of checkpoint...
    #get init scores!!!
    print("Starting evaluation scores:")
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        running_score = 0.0
        running_corr_score = 0.0
        running_val_score = 0.0
        #calculate training losses...
        for i, qa_batch in enumerate(train_loader):
            batch, answers = qa_batch
            batch, answers = batch.to(device), answers.to(device)
            #get corrupted triples
            corrupted = corrupted_answer(model.num_entities, answers.size(), start = 1)
            corrupted = corrupted.to(device)
            #calculate loss...
            loss, score, corr_score = model(batch, answers, corrupted)
            #getting losses...
            running_loss += loss.mean().data.item()
            running_score += score.mean().data.item()
            running_corr_score += corr_score.mean().data.item()
        #calculate val loss...
        for j, qa_batch in enumerate(val_loader):
            #questions and answers!
            batch, answers = qa_batch
            batch, answers = batch.to(device), answers.to(device)
            #calculate validation scores!!!
            running_val_score += model.evaluate(batch, answers).mean().data.item()
    #print results...
    print('Epoch: ', epoch_stop, ', loss: ', "{:.4f}".format(running_loss/i),
        ', score: ', "{:.4f}".format(running_score/i),
        ', corrupted score: ', "{:.4f}".format(running_corr_score/i),
        ', val_score: ', "{:.4f}".format(running_val_score/j),
        ', time: ', "starting...")
    #get metrics
    log_metrics({
        "loss": running_loss/i,
        "golden score": running_score/i,
        "corrupted score": running_corr_score/i,
        "val score": running_val_score/i
        }, 0)
    #make temp dir
    os.makedirs("./temp")
    #training ...
    print('Training begins ...')
    for epoch in range(1 , epochs + 1):
        model.train()
        running_loss = 0.0
        running_score = 0.0
        running_corr_score = 0.0
        # #perform normalizations before entering the mini-batch.
        model.normalize()
        for i, qa_batch in enumerate(train_loader):
            batch, answers = qa_batch
            batch, answers = batch.to(device), answers.to(device)
            #get corrupted triples
            corrupted = corrupted_answer(model.num_entities, answers.size(), start = 1)
            corrupted = corrupted.to(device)
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
        model.eval()
        with torch.no_grad():
            running_val_score = 0.0
            for j, qa_batch in enumerate(val_loader):
                #questions and answers!
                batch, answers = qa_batch
                batch, answers = batch.to(device), answers.to(device)
                #calculate validation scores!!!
                running_val_score += model.evaluate(batch, answers).mean().data.item()
        #print results...
        print('Epoch: ', epoch, ', loss: ', "{:.4f}".format(running_loss/i),
            ', score: ', "{:.4f}".format(running_score/i),
            ', corrupted score: ', "{:.4f}".format(running_corr_score/i),
            ', val_score: ', "{:.4f}".format(running_val_score/j),
            ', time: ', "{:.4f}".format((time.time()-t_start)/60), 'min(s)')
        
        #collecting metrics...
        log_metrics({
            "loss": running_loss/i,
            "golden score": running_score/i,
            "corrupted score": running_corr_score/i,
            "val score": running_val_score/j
        }, epoch)

        #implementation of early stop using val_energy (fastest route (could use mean_rank for example))
        if patience != -1:
            if highest_val_score <= running_val_score/j:
                #setting new score!
                highest_val_score = running_val_score/j
                #save model checkpoint...
                save_checkpoint(model, [model.num_entities, model.num_relationships], 
                    model.kwargs,'./temp')
                epoch_stop = epoch
                #"zero out" counter
                stop_counter = 1
            else:
                if stop_counter >= patience:
                    #no more patience...early stopping!
                    print('Early stopping at epoch:', epoch)
                    print('Loading from epoch:', epoch_stop)
                    #load model from previous checkpoint!
                    model = load_checkpoint('./temp', model.__class__, device)
                    break
                else:
                    #be patient...
                    stop_counter += 1
                    #if in the end of training load from checkpoint!
                    if epoch == epochs + epoch_stop:
                        print('Finished during early stopping...')
                        print('Loading from epoch:', epoch_stop)
                        #load model from previous checkpoint!
                        model = load_checkpoint('./temp', model.__class__, device)
        else:
            epoch_stop = epochs
    ## If checkpoint exists, delete it ##
    if os.path.isfile("./temp/checkpoint.pt"):
        os.remove("./temp/checkpoint.pt")
        os.rmdir("./temp")

    print('Training ends ...')
    #normalize the embeddings (to be exactly norm2 == 1)
    model.normalize()
    #putting model weights to cpu (only useful when we have gpu training...)
    model.to(torch.device('cpu'))
    #returning model as well as optimizer and actual last epoch (early stopping)...
    return model, epoch_stop, optimizer
