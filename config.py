from torch import device, cuda

DEVICE = device("cuda" if cuda.is_available() else "cpu")
DEVICE_COUNT = cuda.device_count()
URI = "./mlruns" #mlflow
