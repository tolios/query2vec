import subprocess
import sys
import os
from time import time
import json

if sys.argv[1] == "-f" or sys.argv[1] == "--folder":
    run_folder = sys.argv[2]
    # Extract the absolute path
    run_folder = os.path.abspath(run_folder)
    dir_list = os.listdir(run_folder)
    if len(sys.argv) > 3:
        extra = sys.argv[3:]
    else:
        extra = []
else:
    # since no -f flag (folder), then
    # Get the full path to the script being executed
    run_path = sys.argv[1]
    # Extract the directory path from the script path
    run_folder = os.path.abspath(os.path.join(os.path.abspath(run_path), os.pardir))
    dir_list = [os.path.basename(os.path.abspath(run_path))]
    if len(sys.argv) > 2:
        extra = sys.argv[2:]
    else:
        extra = []

print(f"Starting execution of {len(dir_list)} run(s) ...")
print("--------------------------------------------------------------------------------------------------------------")

if len(extra) >= 2:
    extra_train = [extra.pop(0)]
else:
    extra_train = []

for i, dir in enumerate(dir_list):
    print(f"Executing run : {dir} ... ({i}/{len(dir_list)}) Done")
    run_dir = os.path.join(run_folder, dir)

    model_json = os.path.join(run_dir, "model.json")
    config_json = os.path.join(run_dir, "train_config.json")

    # Specify the command to execute the script
    command = ["python", f"./query2vec/main.py", config_json, model_json] + extra_train

    # Execute the script and capture its output in real-time
    # Wait for the script to finish
    # Check the exit code of the script
    start = time()
    exit_code = subprocess.call(command)
    end = time()

    if exit_code != 0:
        print(f"Script exited with non-zero exit code: {exit_code}")
    else:
        print(f"Run {dir} finished successfully in {end - start}s")

        print("Continue with testing of model")
        with open("run.json", "r") as f:
            info = json.load(f)
            id = info["run_id"]
            dataset = info["dataset"]

        print("**************************************************************************************************************")
        string_list = []
        for test in range(1, 8): # NOTE - make the other tests for all the good models
            if os.path.exists(f"{dataset}/test_qa_{test}.txt"):
                string_list.append(f"{dataset}/test_qa_{test}.txt")

        command = ["python", f"./query2vec/test.py",
            id, f"tests.yml", "--N=3", "--filtering=true",
            "--tests="+f"{string_list}",
            f"--train_data={dataset}/train_qa_1.txt", 
            f"--val_data={dataset}/val_qa_1.txt"] + extra
        
        exit_code = subprocess.call(command)
        print("finished!")
        print("**************************************************************************************************************")

    print("--------------------------------------------------------------------------------------------------------------")
