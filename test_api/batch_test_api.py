import base64
from io import BytesIO
from PIL import Image
import requests
import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='../reorg_val/images',
                    help='input')
parser.add_argument('--mask', type=str, default='../reorg_val/masks',
                    help='path to mask')
parser.add_argument('--output', type=str, default='../val_output_k3mix7_s_up_merge',
                    help='path to save')

opt, _ = parser.parse_known_args()


path_img = opt.image#"./988.jpg"
path_msk = opt.mask#"./988.png"
path_out = opt.output

for name in tqdm(os.listdir(path_img)):
    _name = ".".join(name.split(".")[:-1])
    img = Image.open(os.path.join(path_img, name)).convert("RGB")
    msk = Image.open(os.path.join(path_msk, _name+".png")).convert("L")


    mode_img = img.mode
    mode_msk = msk.mode

    W, H = img.size
    str_img = img.tobytes().decode("latin1")
    str_msk = msk.tobytes().decode("latin1")

    data = {'str_img': str_img, 'str_msk': str_msk, 'width':W, 'height':H, 
            'mode_img':mode_img, 'mode_msk':mode_msk}

    r = requests.post('http://localhost:2334/api', json=data)

    str_result = r.json()['str_result']

    result = str_result.encode("latin1")
    result = Image.frombytes('RGB', (W, H), result, 'raw')
    result.save(os.path.join(path_out, name))
