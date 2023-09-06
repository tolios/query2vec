import argparse
import os
import json
from metrics import *
from graph import qa_dataset
from config import DEVICE, URI
from torch_geometric.seed import seed_everything
from mlflow import set_tracking_uri, log_dict, start_run
from mlflow.pytorch import load_model
import ast

#parser for all arguments!
parser = argparse.ArgumentParser(description='Testing query embeddings...')

#requirement arguments...
parser.add_argument("run_id",
                    type=str, help="Run id of a model")
parser.add_argument("test_dict",
                    type=str, help="Test dict containing results")
#optional requirements!
parser.add_argument("--N",
                    default=10,
                    type=int, help="hits@N N. Only used when hits@ is used as argument in metric")
parser.add_argument("--tests",
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
                    default=1000,
                    type=int, help="Test batch size")
parser.add_argument("--seed",
                    default=42,
                    type=int, help="Seed for randomness")

#finds all arguments...
args = parser.parse_args()

SEED = args.seed
MODEL_URI = "runs:/"+args.run_id+"/model"
filtering = args.filtering
test_data = ast.literal_eval(args.tests)
N = args.N
batch_size = args.batch_size

#seeds
seed_everything(SEED)

set_tracking_uri(URI) #sets uri for mlflow!

#load model...
model = load_model(MODEL_URI)
#put model to device 
model.to(DEVICE)

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

else:
    filter = None

logs = {}

for i, test_file in enumerate(test_data):

    print(f"Loading {test_file} ...")
    test = qa_dataset(test_file) #get test data
    if filtering:
        if i == 0:
            print("creating filter...")
            filter = Filter(train, val, test, num_entities, big = args.big)
            del train, val #not needed anymore!
            print("filter made successfully!")
        else:
            # update to new test
            print("updating filter...")
            filter.change_test(test)
            print("done!")

    logs[test_file] = {}
    result1 = mean_rank(test, model, batch_size = batch_size, filter=filter, device=DEVICE)
    logs[test_file]["mean_rank"] = {
        "result": result1,
        "N": None,
    }
    result2 = hits_at_N(test, model, N=N, batch_size = batch_size, filter=filter, device=DEVICE)*100
    logs[test_file]["hits@"] = {
        "result": result2,
        "N": N,
    }
    result3 = mean_reciprocal_rank(test, model, batch_size = batch_size, filter=filter, device=DEVICE)*100
    logs[test_file]["mrr"] = {
        "result": result3,
        "N": None,
    }
    print(f"Finished {test_file}")

logs["utils"] = {}
logs["utils"]["filtering"] = filtering
logs["utils"]["train_data"] = args.train_data if (filtering) else None
logs["utils"]["val_data"] = args.val_data if (filtering) else None

with start_run(run_id=args.run_id):
    log_dict(logs, artifact_file="tests/"+args.test_dict)
