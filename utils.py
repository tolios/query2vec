import torch

def save_checkpoint(model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        args: list, kwargs:dict, path_dir:str)->None:
    '''
    General save checkpoint function for gnn Model(s). 
    It expects the model is defined ONLY by n_entities, n_relationships. Rest are kwargs...


    '''
    _dict = {
        'state_dict' : model.state_dict(),
        'args': args,
        'kwargs': kwargs,
    }
    #use mlflow log for saving the model, 
    torch.save(_dict, path_dir+"/checkpoint.pt")
    torch.save(optimizer.state_dict(), path_dir+"/optimizer.pt")
    torch.save(scheduler.state_dict(), path_dir+"/scheduler.pt")

def load_checkpoint(path_dir:str, model_class:torch.nn.Module, 
        optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        device: torch.device)->tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    '''
    General load checkpoint function for gnn Model(s).
    It receives a path and corresponding model type, and returns loaded model!
    '''
    #get info
    _dict = torch.load(path_dir+"/checkpoint.pt", map_location=device)
    #create model with specified architecture
    model = model_class(*_dict['args'], **_dict['kwargs'])
    #load weights...
    model.load_state_dict(_dict['state_dict'])
    optimizer_state_dict = torch.load(path_dir+"/optimizer.pt", map_location=device)
    scheduler_state_dict = torch.load(path_dir+"/scheduler.pt", map_location=device)
    #return both model and updated optimizer as well as scheduler
    optimizer.load_state_dict(optimizer_state_dict)
    scheduler.load_state_dict(scheduler_state_dict)
    
    return model, optimizer, scheduler
