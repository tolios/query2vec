import argparse
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
                    choices=['mean_rank', 'hits@'],
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

if filtering:
    # filter = Filter(train, val, test, big = args.big)
    pass
else:
    filter = None

if METRIC == 'mean_rank':
    result = mean_rank(test, model, batch_size = batch_size, device=DEVICE)
elif METRIC == 'hits@':
    result = hits_at_N(test, model, N=N, batch_size = batch_size, device=DEVICE)*100
else:
    raise KeyError('No such metric available. Use: "mean_rank" or "hits@"')

with start_run(run_id=args.run_id):
    log_dict({
        "metric": METRIC,
        "filtering": filtering,
        "result": result,
        "test_data": test_data
    }, artifact_file="tests/"+args.test_dict)
