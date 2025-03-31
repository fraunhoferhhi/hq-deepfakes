import os

path = "your_path"

for f in os.listdir(path):
    f_new = os.path.join(path, f.split("---")[-1])
    os.rename(os.path.join(path, f), f_new)
