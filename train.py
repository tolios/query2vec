import time
import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch.optim import AdamW
from graph import *
from utils import save_checkpoint, load_checkpoint
from mlflow import log_metrics
from metrics import hits_at_N

def training(model: torch.nn.Module, optimizer_dict:dict, scheduler_dict:dict,
    train: Dataset, val: Dataset,
    epochs = 50, batch_size = 1024, val_batch_size = 1024, num_negs = 1,
    lr = 0.1, weight_decay = 0.0005, patience = -1, pretrained=False, filter=None, 
    scheduler_patience=3, scheduler_factor=0.1, scheduler_threshold=0.1, val_every=10,
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
    optimizer = AdamW(model.parameters(), lr = lr, weight_decay = weight_decay) #FIXME - AdamW or something else?
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
    highest_val_hitsAt3 = -0.1
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
        running_val_loss = 0.0
        running_val_corr_score = 0.0
        #calculate training losses...
        q_norms = 0
        a_norms = 0
        for qa_batch in tqdm(train_loader, desc=f"Epoch (train) 0"):
            batch, answers = qa_batch
            batch, answers = batch.to(device), answers.to(device)
            #get corrupted triples
            #corrupted = filter.negatives(batch.hash, model.num_entities, num_negs=num_negs, start = 1)
            corrupted = corrupted_answer(model.num_entities, answers.size(), num_negs=num_negs, start = 1)
            corrupted = corrupted.to(device)
            #calculate loss...
            loss, score, corr_score = model(batch, answers, corrupted)
            #getting losses...
            running_loss += loss.sum().data.item()
            running_score += score.sum().data.item()
            running_corr_score += corr_score.sum().data.item()
            q_embs = model.embed_query(batch)
            a_embs = model.embed_entities(answers)
                
            q_norms += q_embs.norm(p=2, dim = -1).sum().item()
            a_norms += a_embs.norm(p=2, dim = -1).sum().item()
        print("q_norms:", q_norms/len(train), "a_norms:", a_norms/len(train))
        #calculate val loss...
        for qa_batch in val_loader:
            #questions and answers!
            batch, answers = qa_batch
            batch, answers = batch.to(device), answers.to(device)
            #get corrupted triples
            corrupted = corrupted_answer(model.num_entities, answers.size(), num_negs=num_negs, start = 1)
            #corrupted = filter.test_negatives(batch.hash, model.num_entities, num_negs=num_negs, start = 1)
            corrupted = corrupted.to(device)
            #calculate validation scores!!!
            #calculate loss...
            loss, score, corr_score = model(batch, answers, corrupted)
            #getting losses...
            running_val_loss += loss.sum().data.item()
            running_val_score += score.sum().data.item()
            running_val_corr_score += corr_score.sum().data.item()
        hitsATN = hits_at_N(val, model, N=3, filter=filter, device=device)
    #print results...
    print('Epoch: ', epoch_stop, ',loss:', "{:.4f}".format(running_loss/(len(train))),
        ",val loss:", "{:.4f}".format(running_val_loss/(len(val))),
        ',score:', "{:.4f}".format(running_score/(len(train))),
        ',val score:', "{:.4f}".format(running_val_score/(len(val))),
        ',corr score:', "{:.4f}".format(running_corr_score/(len(train))),
        ',val corr:', "{:.4f}".format(running_val_corr_score/(len(val))),
        ',val hits@3:', "{:.2f}".format(hitsATN*100),
        ',time:', "starting...")
    #get metrics
    log_metrics({
        "loss": running_loss/(len(train)),
        "val loss": running_val_loss/(len(val)),
        "golden score": running_score/(len(train)),
        "val score": running_val_score/(len(val)),
        "corr score": running_corr_score/(len(train)),
        "val corr": running_val_corr_score/(len(val)),
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
        running_val_score = 0.0
        running_val_loss = 0.0
        running_val_corr_score = 0.0
        # #perform normalizations before entering the mini-batch.
        q_norms = 0
        a_norms = 0

        for qa_batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
            #zero out gradients...
            optimizer.zero_grad()
            batch, answers = qa_batch
            batch, answers = batch.to(device), answers.to(device)
            #get corrupted triples
            corrupted = corrupted_answer(model.num_entities, answers.size(), num_negs=num_negs, start = 1)
            #corrupted = filter.negatives(batch.hash, model.num_entities, num_negs=num_negs, start = 1)
            corrupted = corrupted.to(device)
            #calculate loss...
            loss, score, corr_score = model(batch, answers, corrupted)
            q_embs = model.embed_query(batch)
            a_embs = model.embed_entities(answers)
            with torch.no_grad():
                q_norms += q_embs.norm(p=2, dim = -1).sum().item()
                a_norms += a_embs.norm(p=2, dim = -1).sum().item()
            #loss backward
            (loss.mean()+0.000001*(q_embs.norm(p=2)**2) + 0.000001*(a_embs.norm(p=2)**2)).backward() #FIXME - might need .sum
            # loss.mean().backward()
            #update parameters!
            optimizer.step()
            #getting losses...
            running_loss += loss.sum().data.item()
            running_score += score.sum().data.item()
            running_corr_score += corr_score.sum().data.item()
        print("q_norms:", q_norms/len(train), "a_norms:", a_norms/len(train))
        #calculating val energy....
        model.eval()
        with torch.no_grad():
            for qa_batch in val_loader:
                #questions and answers!
                batch, answers = qa_batch
                batch, answers = batch.to(device), answers.to(device)
                #get corrupted triples
                corrupted = corrupted_answer(model.num_entities, answers.size(), num_negs=num_negs, start = 1)
                #corrupted = filter.test_negatives(batch.hash, model.num_entities, num_negs=num_negs, start = 1)
                corrupted = corrupted.to(device)
                #calculate validation scores!!!
                #calculate loss...
                loss, score, corr_score = model(batch, answers, corrupted)
                #getting losses...
                running_val_loss += loss.sum().data.item()
                running_val_score += score.sum().data.item()
                running_val_corr_score += corr_score.sum().data.item()
            
            if epoch % val_every == 0:
                hitsATN = hits_at_N(val, model, N=3, filter=filter, device=device)
                # will make lr smaller if hitsATN doesn't improve
                scheduler.step(hitsATN*100)

        #print results...
        if epoch % val_every == 0:
            print('Epoch: ', epoch, ',loss:', "{:.4f}".format(running_loss/(len(train))),
                ',val loss:', "{:.4f}".format(running_val_loss/(len(val))),
                ',score:', "{:.4f}".format(running_score/(len(train))),
                ',val score:', "{:.4f}".format(running_val_score/(len(val))),
                ',corr score:', "{:.4f}".format(running_corr_score/(len(train))),
                ',val corr:', "{:.4f}".format(running_val_corr_score/(len(val))),
                ',val hits@3:', "{:.2f}".format(hitsATN*100),
                ',time:', "{:.4f}".format((time.time()-t_start)/60), 'min(s)')
        else:
            print('Epoch: ', epoch, ',loss:', "{:.4f}".format(running_loss/(len(train))),
            ',val loss:', "{:.4f}".format(running_val_loss/(len(val))),
            ',score:', "{:.4f}".format(running_score/(len(train))),
            ',val score:', "{:.4f}".format(running_val_score/(len(val))),
            ',corr score:', "{:.4f}".format(running_corr_score/(len(train))),
            ',val corr:', "{:.4f}".format(running_val_corr_score/(len(val))),
            ',time:', "{:.4f}".format((time.time()-t_start)/60), 'min(s)')
        
        #collecting metrics...
        log_metrics({
            "loss": running_loss/(len(train)),
            "val loss": running_val_loss/(len(val)),
            "golden score": running_score/(len(train)),
            "val score": running_val_score/(len(val)),
            "corr score": running_corr_score/(len(train)),
            "val corr": running_val_corr_score/(len(val)),
        }, epoch)

        if epoch % val_every == 0:
            #collecting metrics...
            log_metrics({
                "hitsAt3": hitsATN*100,
            }, epoch)

        #implementation of early stop using val_energy (fastest route (could use mean_rank for example))
        if patience != -1 and epoch % val_every == 0:
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
    if os.path.exists("./temp"):
        if os.path.isfile("./temp/checkpoint.pt"):
            os.remove("./temp/checkpoint.pt")
        if os.path.isfile("./temp/optimizer.pt"):
            os.remove("./temp/optimizer.pt")
        if os.path.isfile("./temp/scheduler.pt"):
            os.remove("./temp/scheduler.pt")
        os.rmdir("./temp")

    print('Training ends ...')
    #putting model weights to cpu (only useful when we have gpu training...)
    model.to(torch.device('cpu'))
    #returning model as well as optimizer and actual last epoch (early stopping)...
    return model, epoch_stop, optimizer, scheduler
