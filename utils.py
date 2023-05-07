import torch

def save_checkpoint(model: torch.nn.Module, args: list, kwargs:dict, path_dir:str)->None:
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

def load_checkpoint(path_dir:str, model_class:torch.nn.Module, device: torch.device)->tuple[torch.nn.Module, dict]:
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
    
    return model
