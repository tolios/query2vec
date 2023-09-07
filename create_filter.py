from metrics import Filter
import argparse
from graph import qa_dataset
import pickle

#parser for all arguments!
parser = argparse.ArgumentParser(description='Creating filter for loading in tests...')
parser.add_argument("save",
                    type=str, help="Path to save json...")
parser.add_argument("train",
                    type=str, help="Training path...")
parser.add_argument("valid",
                    type=str, help="Validation path...")

#finds all arguments...
args = parser.parse_args()

train = qa_dataset(args.train)
val = qa_dataset(args.valid)

print("Creating filter for train and val ...")
dict_ = Filter._create_stable_dict(train, val)

with open(args.save, "wb") as f:
    pickle.dump(dict_, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved as pickle ...")