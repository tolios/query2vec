import argparse
import os
import json
from metrics import *
from graph import qa_dataset
from config import DEVICE, URI
from torch_geometric.seed import seed_everything
from mlflow import set_tracking_uri, log_dict, start_run
from mlflow.pytorch import load_model

#parser for all arguments!
parser = argparse.ArgumentParser(description='Testing query embeddings...')

#requirement arguments...
parser.add_argument("run_id",
                    type=str, help="Run id of a model")
parser.add_argument("test_dict",
                    type=str, help="Test dict containing results")
parser.add_argument("metric",
                    choices=['mean_rank', 'hits@', 'mrr', 'ndcg'],
                    type=str, help="Metric to be used for testing!")
#optional requirements!
parser.add_argument("--algorithm",
                    default='rgcn',
                    type=str, help="Embedding algorithm (stored in algorithms folder!)")
parser.add_argument("--N",
                    default=10,
                    type=int, help="hits@N N. Only used when hits@ is used as argument in metric")
parser.add_argument("--test_data",
                    default='./datasets/FB15k_237/qa/test_qa.txt',
                    type=str, help="Path to test data")
parser.add_argument("--train_data",
                    default=None,
                    type=str, help="Path to train data")
parser.add_argument("--val_data",
                    default=None,
                    type=str, help="Path to val data")
parser.add_argument("--filtering",
                    default = False,
                    type=bool, help="Filter out true answers, that artificially lower the scores...")
parser.add_argument("--big",
                    default='10e5',
                    type=float, help="Value of mask, so as to filter out golden triples")
parser.add_argument("--batch_size",
                    default=64,
                    type=int, help="Test batch size")
parser.add_argument("--seed",
                    default=42,
                    type=int, help="Seed for randomness")

#finds all arguments...
args = parser.parse_args()

SEED = args.seed
MODEL_URI = "runs:/"+args.run_id+"/model"
ALGORITHM = args.algorithm
METRIC = args.metric
filtering = args.filtering
test_data = args.test_data
N = args.N
batch_size = args.batch_size

#seeds
seed_everything(SEED)

test = qa_dataset(test_data) #get test data

set_tracking_uri(URI) #sets uri for mlflow!

#load model...
model = load_model(MODEL_URI)
#put model to device 
model.to(DEVICE)

if METRIC == 'ndcg':
    filtering = None
    train = qa_dataset(args.train_data)
    val = qa_dataset(args.val_data)
    result = NDCG(train, val, test, model)*100
else:
    if filtering:
        if not (args.train_data and args.val_data):
            print("train data and val data REQUIRED when filtering!!!")
            raise 

        train = qa_dataset(args.train_data)
        val = qa_dataset(args.val_data)

        #directory where qas are stored...
        id_dir=os.path.dirname(args.train_data)

        with open(os.path.join(id_dir, "info.json"), "r") as file:
            info = json.load(file)

        num_entities = info["num_entities"]

        print("creating filter...")
        filter = Filter(train, val, test, num_entities, big = args.big)
        print("filter made successfully!")
    else:
        filter = None

    if METRIC == 'mean_rank':
        result = mean_rank(test, model, batch_size = batch_size, filter=filter, device=DEVICE)
    elif METRIC == 'hits@':
        result = hits_at_N(test, model, N=N, batch_size = batch_size, filter=filter, device=DEVICE)*100
    elif METRIC == 'mrr':
        result = mean_reciprocal_rank(test, model, batch_size = batch_size, filter=filter, device=DEVICE)*100
    else:
        raise KeyError('No such metric available. Use: "mean_rank" or "hits@"')

with start_run(run_id=args.run_id):
    log_dict({
        "metric": METRIC,
        "filtering": filtering,
        "result": result,
        "test_data": test_data,
        "N": N if METRIC == "hits@" else None,
        "train_data": args.train_data if (filtering or (METRIC=='ndcg')) else None,
        "val_data": args.val_data if (filtering or (METRIC=='ndcg')) else None
    }, artifact_file="tests/"+args.test_dict)
