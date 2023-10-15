import os
from numpy import dtype, int64, float32
import torch
from torch_geometric.seed import seed_everything
from train import training
from graph import qa_dataset
import argparse
import importlib
import inspect
import warnings
import json
from config import DEVICE, DEVICE_COUNT, URI
from mlflow import log_params, set_tag, start_run, set_tracking_uri, set_experiment, log_param
from mlflow.pytorch import log_model, log_state_dict, load_model, load_state_dict
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from metrics import Filter

#! UNDER DEVELOPMENT CHECK ALL

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore")

#parser for all arguments!
parser = argparse.ArgumentParser(description='Training knowledge graph embeddings...',
                                 epilog='''
                                    NOTE: You can also add as arguments the kwargs in the Model class,
                                    defined inside the algorithms folder. For example, if --algorithm=transe,
                                    then all kwargs defined in the transe.Model class, can be changed i.e --norm=1
                                        ''')
parser.add_argument("json_config",
                    type=str, help="Training configuration json file...")
parser.add_argument("json_model",
                    type=str, help="Model architecture json file...")
parser.add_argument("--val_filter",
                    type=bool,
                    default=True, help="val filter path")

#finds all arguments...
args = parser.parse_args()

#load model.json
with open(args.json_model, 'r') as f:
    updated_args = json.load(f)
    algorithm = updated_args['model_type']
    updated_args = updated_args['model_params']

#custom parsed arguments from Model kwargs!!!
#given module... algorithm argument
module = importlib.import_module('algorithms.'+algorithm, ".")
#module.Model keyword args!
spec_args = inspect.getfullargspec(module.Model)
values = spec_args.defaults
model_args = spec_args.args[-len(values):]
#make arg dictionary
model_args = {x:updated_args[x] if x in updated_args else y for x, y in zip(model_args, values)}

with open(args.json_config, 'r') as f:
    config = json.load(f)

#configs
SEED = config["config"].get("seed", 42)
TRAIN_PATH = config["config"].get("train_data", './datasets/FB15k_237/qa/train_qa.txt')
VAL_PATH = config["config"].get("val_data", './datasets/FB15k_237/qa/val_qa.txt')
EPOCHS = config["config"].get("epochs", 5)
BATCH_SIZE = config["config"].get("train_batch_size", 1024)
VAL_BATCH_SIZE = config["config"].get("val_batch_size", 1024)
LEARNING_RATE = config["config"].get("lr", 0.01)
WEIGHT_DECAY = config["config"].get("wd", 0.001)
PATIENCE = config["config"].get("patience", -1)
NUM_NEGS = config["config"].get("num_negs", 1)
VAL_EVERY = config["config"].get("val_every", 10)
scheduler_patience = config["config"].get("scheduler_patience", 3) 
scheduler_factor = config["config"].get("scheduler_factor", 0.1)
scheduler_threshold = config["config"].get("scheduler_threshold", 0.1)
pretrained = config["config"].get("pretrained", False)

#seeds
seed_everything(SEED)

set_tracking_uri(URI) #sets uri for mlflow!

set_experiment(config["experiment"])

#directory where qas are stored...
id_dir=os.path.dirname(TRAIN_PATH)

if pretrained:
    #load model!    
    #pretrained should end in .../
    model = load_model("runs:/"+pretrained+"/model")
    optimizer_dict = load_state_dict("runs:/"+pretrained+"/optimizer")
    scheduler_dict = load_state_dict("runs:/"+pretrained+"/scheduler")

else:

    #define trainable embeddings!
    with open(os.path.join(id_dir, "info.json"), "r") as file:
        info = json.load(file)

    num_entities = info["num_entities"]
    num_relationships = info["num_relationships"]

    model = module.Model(num_entities, num_relationships, **model_args)
    optimizer_dict = {}
    scheduler_dict = {}


if args.val_filter:
    filter_path = os.path.join(id_dir, "val_filter.pkl")
    filter = Filter(None, None, VAL_PATH, model.num_entities, load_path=filter_path, delete=True)

else:
    filter = None

#data
#training
train_qa = qa_dataset(TRAIN_PATH)
val_qa = qa_dataset(VAL_PATH)

#training begins...
with start_run(run_name=config["run"], experiment_id=config["experiment_id"]) as run:
    set_tag("algorithm", algorithm)
    log_params(model_args)
    log_params(config["config"])
    log_param("val_filter", args.val_filter)
    log_param("num_entites", model.num_entities)
    log_param("num_relationships", model.num_relationships)
    model, final_epoch, optimizer, scheduler = training(model, optimizer_dict, scheduler_dict, 
                train_qa, val_qa, device=DEVICE, device_count=DEVICE_COUNT, epochs = EPOCHS,
                batch_size = BATCH_SIZE, val_batch_size = VAL_BATCH_SIZE, num_negs=NUM_NEGS,
                lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY, patience = PATIENCE, filter=filter, val_every=VAL_EVERY,
                scheduler_patience = scheduler_patience, scheduler_factor = scheduler_factor, scheduler_threshold = scheduler_threshold)
    log_param("final_epoch", final_epoch)
    
    #! move somewhere relevant to the model?
    input_schema = Schema(
        [
            TensorSpec(dtype(int64), (-1, 1), "x"),
            TensorSpec(dtype(int64), (2, -1), "edge_index"),
            TensorSpec(dtype(int64), (-1, 1), "edge_attr"),
            TensorSpec(dtype(int64), (-1,), "batch"),
            TensorSpec(dtype(int64), (-1,), "ptr")
        ]
    )
    output_schema = Schema([TensorSpec(dtype(float32), (-1, model.kwargs["emb_dim"]))])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    #logging model!    
    log_model(
            model, 
            "model",
            signature=signature
        )
    #logging optimizer!
    log_state_dict(
        optimizer.state_dict(), "optimizer"
    )
    #logging scheduler
    log_state_dict(
        scheduler.state_dict(), "scheduler"
    )

    #in the end write to a run.json file with the run id!
    run_id = run.info.run_id
    with open("./run.json", "w") as f:
        json.dump({"run_id": run_id, "dataset":  os.path.dirname(TRAIN_PATH)}, f)
