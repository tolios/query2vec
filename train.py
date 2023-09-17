import time
import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch.optim import Adagrad
from graph import *
from utils import save_checkpoint, load_checkpoint
from mlflow import log_metrics
from metrics import hits_at_N_Grouped

def training(model: torch.nn.Module, optimizer_dict:dict, scheduler_dict:dict,
    train: Dataset, val: Dataset,
    epochs = 50, batch_size = 1024, val_batch_size = 1024, num_negs = 1,
    lr = 0.1, weight_decay = 0.0005, patience = -1, pretrained=False, filter=None, 
    scheduler_patience=3, scheduler_factor=0.1, scheduler_threshold=0.1,
    device=torch.device('cpu')):
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", 
        patience=scheduler_patience, factor=scheduler_factor, threshold=scheduler_threshold, verbose=True)
    if pretrained:
        #load also optimizer state!
        scheduler.load_state_dict(scheduler_dict)
    #training begins...
    t_start = time.time()
    #for early stopping!
    #start with huge number!
    highest_val_hitsAt3 = 0.0
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
        for qa_batch in train_loader:
            batch, answers = qa_batch
            batch, answers = batch.to(device), answers.to(device)
            #get corrupted triples
            corrupted = corrupted_answer(model.num_entities, answers.size(), num_negs=num_negs,start = 1)
            corrupted = corrupted.to(device)
            #calculate loss...
            loss, score, corr_score = model(batch, answers, corrupted)
            #getting losses...
            running_loss += loss.sum().data.item()
            running_score += score.sum().data.item()
            running_corr_score += corr_score.sum().data.item()
        #calculate val loss...
        for qa_batch in val_loader:
            #questions and answers!
            batch, answers = qa_batch
            batch, answers = batch.to(device), answers.to(device)
            #calculate validation scores!!!
            running_val_score += model.evaluate(batch, answers).sum().data.item()
        hitsATN = hits_at_N_Grouped(val, model, N=3, filter=filter, device=device,disable=True)
    #print results...
    print('Epoch: ', epoch_stop, ',loss:', "{:.4f}".format(running_loss/(len(train))),
        ',score:', "{:.4f}".format(running_score/(len(train))),
        ',corr_score:', "{:.4f}".format(running_corr_score/(len(train))),
        ',val_score:', "{:.4f}".format(running_val_score/(len(val))),
        ',val hits@3:', "{:.2f}".format(hitsATN*100),
        ',time:', "starting...")
    #get metrics
    log_metrics({
        "loss": running_loss/(len(train)),
        "golden score": running_score/(len(train)),
        "corr score": running_corr_score/(len(train)),
        "val score": running_val_score/(len(val)),
        "hitsAt3": hitsATN*100,
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
        for qa_batch in train_loader:
            batch, answers = qa_batch
            batch, answers = batch.to(device), answers.to(device)
            #get corrupted triples
            corrupted = corrupted_answer(model.num_entities, answers.size(), num_negs=num_negs, start = 1)
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
            running_loss += loss.sum().data.item()
            running_score += score.sum().data.item()
            running_corr_score += corr_score.sum().data.item()
        #calculating val energy....
        model.eval()
        with torch.no_grad():
            running_val_score = 0.0
            for qa_batch in val_loader:
                #questions and answers!
                batch, answers = qa_batch
                batch, answers = batch.to(device), answers.to(device)
                #calculate validation scores!!!
                running_val_score += model.evaluate(batch, answers).sum().data.item()
            hitsATN = hits_at_N_Grouped(val, model, N=3, filter=filter, device=device,disable=True)
        
        # will make lr smaller if hitsATN doesn't improve
        scheduler.step(hitsATN*100)

        #print results...
        print('Epoch: ', epoch, ',loss:', "{:.4f}".format(running_loss/(len(train))),
            ',score:', "{:.4f}".format(running_score/(len(train))),
            ',corrupted score:', "{:.4f}".format(running_corr_score/(len(train))),
            ',val_score:', "{:.4f}".format(running_val_score/(len(val))),
            ',val hits@3:', "{:.2f}".format(hitsATN*100),
            ',time:', "{:.4f}".format((time.time()-t_start)/60), 'min(s)')
        
        #collecting metrics...
        log_metrics({
            "loss": running_loss/(len(train)),
            "golden score": running_score/(len(train)),
            "corr score": running_corr_score/(len(train)),
            "val score": running_val_score/(len(val)),
            "hitsAt3": hitsATN*100,
        }, epoch)

        #implementation of early stop using val_energy (fastest route (could use mean_rank for example))
        if patience != -1:
            if highest_val_hitsAt3 < hitsATN:
                #setting new score!
                highest_val_hitsAt3 = hitsATN
                #save model checkpoint...
                save_checkpoint(model, optimizer, scheduler,
                    [model.num_entities, model.num_relationships], model.kwargs,'./temp')
                epoch_stop = epoch
                #"zero out" counter
                stop_counter = 1
            else:
                if stop_counter >= patience:
                    #no more patience...early stopping!
                    print('Early stopping at epoch:', epoch)
                    print('Loading from epoch:', epoch_stop)
                    #load model from previous checkpoint!
                    model, optimizer, scheduler = load_checkpoint('./temp', model.__class__, optimizer, scheduler, device)
                    break
                else:
                    #be patient...
                    stop_counter += 1
                    #if in the end of training load from checkpoint!
                    if epoch == epochs + epoch_stop:
                        print('Finished during early stopping...')
                        print('Loading from epoch:', epoch_stop)
                        #load model from previous checkpoint!
                        model, optimizer, scheduler = load_checkpoint('./temp', model.__class__, optimizer, scheduler, device)
        else:
            epoch_stop = epochs
    ## If checkpoint exists, delete it ##
    if os.path.isfile("./temp/checkpoint.pt"):
        os.remove("./temp/checkpoint.pt")
        os.remove("./temp/optimizer.pt")
        os.remove("./temp/scheduler.pt")
        os.rmdir("./temp")

    print('Training ends ...')
    #normalize the embeddings (to be exactly norm2 == 1)
    model.normalize()
    #putting model weights to cpu (only useful when we have gpu training...)
    model.to(torch.device('cpu'))
    #returning model as well as optimizer and actual last epoch (early stopping)...
    return model, epoch_stop, optimizer, scheduler
