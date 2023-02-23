from torch import device, cuda
DEVICE = device("cuda:0" if cuda.is_available() else "cpu")