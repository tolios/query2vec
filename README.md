# query2vec

A small repo for producing query embeddings. Under development fo my thesis...

## HowTo

First one needs the train, valid and test triplets for a dataset. For example FB15k_237: [https://www.microsoft.com/en-us/download/details.aspx?id=52312]

To generate data for training and testing, as well as the appropriate filters,
one can prepare the following script:

```[bash]
#!/usr/bin/env bash

dataset_path="./datasets/FB15k_237/"
qa_folder="qa_example" #149689
train_path=$dataset_path"train.txt"
val_path=$dataset_path"valid.txt"
test_path=$dataset_path"test.txt"
train_query_orders="[[(1, -1), (2, 50000), (2, 50000), (3, 50000), (3, 50000)]]"
val_query_orders="[[(1, -1), (2, 5000), (2, 5000), (3, 5000), (3, 5000)]]"
test_query_orders="[[], [(2, 5000)], [(2, 5000)], [(3, 5000)], [(3, 5000)], [(3, 5000)], [(3, 5000)]]"
include_train='[["1p", "2p", "2i", "3p", "3i"]]'
include_val='[["1p", "2p", "2i", "3p", "3i"]]'
include_test='[["1p"], ["2p"], ["2i"], ["3p"], ["3i"], ["ip"], ["pi"]]'

python ./query2vec/graph.py $train_path $val_path $test_path \
            --qa_folder=$qa_folder --train_query_orders="$train_query_orders" \
            --val_query_orders="$val_query_orders" --test_query_orders="$test_query_orders" \
            --include_train="$include_train" --include_val="$include_val" --include_test="$include_test" \
            --add_inverse=true

test_filter=$dataset_path$qa_folder"/filter.pkl"
val_filter=$dataset_path$qa_folder"/val_filter.pkl"
train_ds=$dataset_path$qa_folder"/train_qa_1.txt"
val_ds=$dataset_path$qa_folder"/val_qa_1.txt"

python ./query2vec/create_filter.py $test_filter $train_ds $val_ds $val_filter
```

Then to run an experiment, one can simply prepare a model.json and train_config.json (example can be found in the example folder) and simply run:

```[shell]
python ./query2vec/run.py ./query2vec/example --all_tests
```

The "--all_tests flag is used for testing all metrics!"

Also, if one can prepare multiple runs, by preparing a folder (runs) that contains folders of model.json and train_config.json

To run them all:

```[shell]
python ./query2vec/run.py -f ./runs --all_tests
```

To monitor the experiments, one can use mlflow command (in the location where mlruns is located):

```[shell]
mlflow server
```

This creates a server on localhost:5000
