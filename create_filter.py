from metrics import Filter
import argparse
import pickle

#parser for all arguments!
parser = argparse.ArgumentParser(description='Creating filter for loading in tests...')
parser.add_argument("save",
                    type=str, help="Path to save json...")
parser.add_argument("train",
                    type=str, help="Training path...")
parser.add_argument("valid",
                    type=str, help="Validation path...")
parser.add_argument("save_val",
                    type=str,
                    default="", help="If given, creates filter only with the training set")

#finds all arguments...
args = parser.parse_args()

if args.save_val:
    print("Creating filter for val using training data!")
    dict_ = Filter._create_train_dict(args.train)
    with open(args.save_val, "wb") as f:
        pickle.dump(dict_, f, protocol=pickle.HIGHEST_PROTOCOL)

del dict_

print("Creating filter for test using train and val ...")
dict_ = Filter._create_stable_dict(args.train, args.valid)

with open(args.save, "wb") as f:
    pickle.dump(dict_, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved as pickle ...")
