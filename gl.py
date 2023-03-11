import torch
from icecream import ic

ic("icecream installed")
print("hello gl")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
