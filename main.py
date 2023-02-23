import time
import os
import torch
from torch_geometric.seed import seed_everything
from train import training
from utils import save, load
from graph import qa_dataset
import argparse
import importlib
import inspect
from torch.utils.tensorboard.writer import SummaryWriter
import warnings
import glob
import json
from config import DEVICE

#TODO USE COLAB GPU !!! QA FOLDER IS LIGHT!!!
#TODO Implement nDCG metric
#! UNDER DEVELOPMENT CHECK ALL

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

#parser for all arguments!
parser = argparse.ArgumentParser(description='Training knowledge graph embeddings...',
                                 epilog='''
                                    NOTE: You can also add as arguments the kwargs in the Model class,
                                    defined inside the algorithms folder. For example, if --algorithm=transe,
                                    then all kwargs defined in the transe.Model class, can be changed i.e --norm=1
                                        ''')
#requirement arguments...
parser.add_argument("save_path",
                    type=str, help="Directory where model is saved")
#optional arguments...
parser.add_argument("--pretrained",action=argparse.BooleanOptionalAction,
                    help="If True, it means we will train again!")
parser.add_argument("--algorithm",
                    default='experiment',
                    type=str, help="Embedding algorithm (stored in algorithms folder!)")
parser.add_argument("--seed",
                    default=42,
                    type=int, help="Seed for randomness")
parser.add_argument("--train_data",
                    default='./datasets/FB15k_237/qa/train_qa.txt',
                    type=str, help="Path to training data")
parser.add_argument("--val_data",
                    default='./datasets/FB15k_237/qa/val_qa.txt',
                    type=str, help="Path to validation data")
parser.add_argument("--epochs",
                    default=5,
                    type=int, help="Number of training epochs")
parser.add_argument("--train_batch_size",
                    default=1024,
                    type=int, help="Training data batch size")
parser.add_argument("--val_batch_size",
                    default=1024,
                    type=int, help="Validation data batch size")
parser.add_argument("--lr",
                    default=0.1,
                    type=float, help="Learning rate")
parser.add_argument("--weight_decay",
                    default=0.0,
                    type=float, help="Weight decay")
parser.add_argument("--patience",
                    default=-1,
                    type=int, help="Patience for early stopping")
parser.add_argument("--json_model",
                    default="",
                    type=str, help="Model architecture json file...")

#finds all arguments...
args = parser.parse_args()

#load model.json
if args.json_model:
    with open(args.json_model, 'r') as f:
        updated_args = json.load(f)
        algorithm = updated_args['model_type']
        updated_args = updated_args['model_params']
else:
    #no updates! Preset architecture...
    algorithm = 'experiment'
    updated_args = {}
#custom parsed arguments from Model kwargs!!!
#given module... algorithm argument
module = importlib.import_module('algorithms.'+algorithm, ".")
#module.Model keyword args!
spec_args = inspect.getfullargspec(module.Model)
values = spec_args.defaults
model_args = spec_args.args[-len(values):]
#make arg dictionary
model_args = {x:updated_args[x] if x in updated_args else y for x, y in zip(model_args, values)}

#seeds
seed_everything(args.seed)

#configs
TRAIN_PATH = args.train_data
VAL_PATH = args.val_data
EPOCHS = args.epochs
BATCH_SIZE = args.train_batch_size
VAL_BATCH_SIZE = args.val_batch_size
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay
PATIENCE = args.patience
SAVE_PATH = args.save_path
algorithm = args.algorithm
pretrained = args.pretrained

cwd = os.getcwd()

#directory where qas are stored...
id_dir=os.path.dirname(TRAIN_PATH)

#data
#training
train_qa = qa_dataset(TRAIN_PATH)
val_qa = qa_dataset(VAL_PATH)

if pretrained:
    #load model!
    model, model_dict = load(SAVE_PATH+'/model.pth.tar', module.Model)
else:
    #define new model!

    #create save_path containing everything, unless pretrained (already exists!!)!
    os.makedirs(SAVE_PATH)

    #define trainable embeddings!
    #! ACCESS info.json to get these!!!

    num_entities = 14505
    num_relationships = 237

    model = module.Model(num_entities, num_relationships, **model_args)

#! fix, train (q, a) where q contains many!
#writer.add_graph(model, (train[:10], train[10:20]))

#change to the model directory...
os.chdir(SAVE_PATH)

#set #of runs!
if not pretrained:
    n = 1
else:
    n = len(glob.glob('./run_*'))+1

#init SummaryWriter
writer = SummaryWriter(log_dir=f'./run_{n}')

start = time.time()
#training begins...
model, writer, actual_epochs = training(model, train_qa, val_qa, writer, device=DEVICE,
                epochs = EPOCHS, batch_size = BATCH_SIZE, val_batch_size = VAL_BATCH_SIZE,
                lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY, patience = PATIENCE)

writer.close()
end = time.time()

#go back...
os.chdir(cwd)

#save model!
#create folder containing embeddings
if not pretrained:
    save(model, [num_entities, num_relationships], model_args, SAVE_PATH+'/model.pth.tar')
else: #! maybe add a replace or not utility! (not lose the old one...)
    save(model, model_dict['args'], model_dict['kwargs'], SAVE_PATH+'/model.pth.tar')
# save train configuration!
with open(SAVE_PATH+f'/train_config_{n}.txt', 'w') as file:
    file.write(f'SEED: {args.seed}\n')
    file.write(f'TRAIN_PATH: {TRAIN_PATH}\n')
    file.write(f'VAL_PATH: {VAL_PATH}\n')
    #! ACCESS info.json (contained in FB15k_237/qa/ )
    # file.write(f'NUM_EMBEDDINGS_OBJECT: {train.n_objects}\n')
    # file.write(f'NUM_EMBEDDINGS_RELATIONSHIP: {train.n_relationships}\n')
    file.write(f'EPOCHS: {EPOCHS}\n')
    file.write(f'ACTUAL_EPOCHS: {actual_epochs}\n')
    file.write(f'PATIENCE: {PATIENCE}\n')
    file.write(f'BATCH_SIZE: {BATCH_SIZE}\n')
    file.write(f'VAL_BATCH_SIZE: {VAL_BATCH_SIZE}\n')
    file.write(f'LEARNING_RATE: {LEARNING_RATE}\n')
    file.write(f'WEIGHT_DECAY: {WEIGHT_DECAY}\n')
    file.write(f'Training time: {"{:.4f}".format((end-start)/60)} min(s)')

#model architecture...
if not pretrained:
    with open(SAVE_PATH+'/model_arch.json', 'w') as f:
        model_args = {
            'model_type': algorithm,
            'model_params': model_args
        }
        json.dump(model_args, f, sort_keys=True, indent=2)