import os

def download_weights_if_necessary():
    path = './models'
    weight_present = os.path.isdir(path)

    if not weight_present:
        import subprocess
        subprocess.call(['sh', './download_weights.sh'])