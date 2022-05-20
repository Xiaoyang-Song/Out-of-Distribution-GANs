from config import *


# TODO: Finish implementing this utility function and incoporate it in the notebook
def visualize_img(loader: DataLoader):
    for img, label in loader:
        ic(img.shape)
        ic(label)
        break
