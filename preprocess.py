from PIL import Image
import numpy as np
import sys
import os
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def get_arr_from_image(path):
    img = Image.open(path).convert("L")
    arr = preprocess(img).unsqueeze(0).cpu().detach().numpy()
    return arr


# [1][3][320][320]
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_image.py abc.jpg")
    img_path = sys.argv[1]
    arr = get_arr_from_image(img_path)
    npy_path = os.path.splitext(img_path)[0] + ".npy"
    np.save(npy_path, arr)