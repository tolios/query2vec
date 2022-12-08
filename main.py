import time
import os
import torch
from torch_geometric.seed import seed_everything
from train import training
from graph import qa_dataset
import argparse
import importlib
import inspect
from torch.utils.tensorboard.writer import SummaryWriter
import warnings

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
                    default=0.001,
                    type=float, help="Learning rate")
parser.add_argument("--weight_decay",
                    default=0.0,
                    type=float, help="Weight decay")
parser.add_argument("--patience",
                    default=-1,
                    type=int, help="Patience for early stopping")

#parse known and unknown args!!!
args, unknown = parser.parse_known_args()

#custom parsed arguments from Model kwargs!!!
#given module... algorithm argument
module = importlib.import_module('algorithms.'+args.algorithm, ".")
#module.Model keyword args!
spec_args = inspect.getfullargspec(module.Model)
values = spec_args.defaults
custom_args = spec_args.args[-len(values):]
#make arg dictionary
model_args = {x:y for x, y in zip(custom_args, values)}
for arg in model_args:
    #adding Model keyword arguments to the parser!!!
    parser.add_argument("--"+arg,default=model_args[arg],
                            type = type(model_args[arg]))

#finds all arguments...
args = parser.parse_args()

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

cwd = os.getcwd()

#directory where qas are stored...
id_dir=os.path.dirname(TRAIN_PATH)

#create save_path containing everything!
os.makedirs(SAVE_PATH)

#now update model dictionary with possible given values!
for arg in model_args:
    model_args[arg] = vars(args)[arg]

#data
#training
train_qa = qa_dataset(TRAIN_PATH)
val_qa = qa_dataset(VAL_PATH)

#define trainable embeddings!
#! ACCESS info.json
model = module.Model(14505, 237, **model_args)

#! fix, train (q, a) where q contains many!
#writer.add_graph(model, (train[:10], train[10:20]))

#change to the model directory...
os.chdir(SAVE_PATH)

#init SummaryWriter
writer = SummaryWriter(log_dir='./run')

start = time.time()
#training begins...
model, writer, actual_epochs = training(model, train_qa, val_qa, writer,
                epochs = EPOCHS, batch_size = BATCH_SIZE, val_batch_size = VAL_BATCH_SIZE,
                lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY, patience = PATIENCE)

writer.close()
end = time.time()

#go back...
os.chdir(cwd)

#save model!
#create folder containing embeddings
model.save(SAVE_PATH+'/model.pth.tar')
#save train configuration!
with open(SAVE_PATH+'/train_config.txt', 'w') as file:
    file.write(f'ALGORITHM: {algorithm}\n')
    file.write(f'SEED: {args.seed}\n')
    file.write(f'TRAIN_PATH: {TRAIN_PATH}\n')
    file.write(f'VAL_PATH: {VAL_PATH}\n')
    #! ACCESS info.json
    # file.write(f'NUM_EMBEDDINGS_OBJECT: {train.n_objects}\n')
    # file.write(f'NUM_EMBEDDINGS_RELATIONSHIP: {train.n_relationships}\n')
    file.write(f'EPOCHS: {EPOCHS}\n')
    file.write(f'ACTUAL_EPOCHS: {actual_epochs}\n')
    file.write(f'PATIENCE: {PATIENCE}\n')
    file.write(f'BATCH_SIZE: {BATCH_SIZE}\n')
    file.write(f'VAL_BATCH_SIZE: {VAL_BATCH_SIZE}\n')
    file.write(f'LEARNING_RATE: {LEARNING_RATE}\n')
    file.write(f'WEIGHT_DECAY: {WEIGHT_DECAY}\n')
    file.write('Model args:\n')
    for arg in model_args:
        file.write(f'{arg}: {model_args[arg]}\n')
    file.write(f'Training time: {"{:.4f}".format((end-start)/60)} min(s)')
