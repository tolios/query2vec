import torch

def save(model: torch.nn.Module, args: list, kwargs:dict, path:str)->None:
    '''
    General save function for gnn Model(s). 
    It expects the model is defined ONLY by n_entities, n_relationships. Rest are kwargs...
    '''
    _dict = {
        'state_dict' : model.state_dict(),
        'args': args,
        'kwargs': kwargs
    }

    torch.save(_dict, path)

def load(path:str, model_class:torch.nn.Module)->torch.nn.Module:
    '''
    General load function for gnn Model(s).
    It receives a path and corresponding model type, and returns loaded model!
    '''
    #get info
    _dict = torch.load(path)
    #create model with specified architecture
    model = model_class(*_dict['args'], **_dict['kwargs'])
    #load weights...
    model.load_state_dict(_dict['state_dict'])
    
    return model, _dict
