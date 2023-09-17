import subprocess
import sys

dataset = sys.argv[1]
filter_path = sys.argv[2]
id_list = sys.argv[3:]

print(f"Starting execution of tests ...")
print("--------------------------------------------------------------------------------------------------------------")

for id in id_list:
    print(f"Executing tests for run : {id}")
    #TODO - make run_tests.py script for full automated testing of a given run id model!
    string_list = []
    for test in range(1, 8): # NOTE - make the other tests for all the good models
        string_list.append(f"{dataset}/test_qa_{test}.txt")

    command = ["python", f"./query2vec/test.py",
        id, f"tests_full.yml", "--N=3", "--filtering=true",
        "--tests="+f"{string_list}",
        f"--train_data={dataset}/train_qa_1.txt", 
        f"--val_data={dataset}/val_qa_1.txt", 
        f"--filter_path={filter_path}",
        "--all_tests"]
    
    exit_code = subprocess.call(command)
    print("finished!")
    print("**************************************************************************************************************")

    print("--------------------------------------------------------------------------------------------------------------")
