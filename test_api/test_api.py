import base64
from io import BytesIO
from PIL import Image
import requests
import pdb

img = Image.open("4.jpg")
mask = Image.open("4.png")

mode_img = img.mode
mode_msk = mask.mode

W, H = img.size
str_img = img.tobytes().decode("latin1")
str_msk = mask.tobytes().decode("latin1")

data = {'str_img': str_img, 'str_msk': str_msk, 'width':W, 'height':H, 
        'mode_img':mode_img, 'mode_msk':mode_msk}

r = requests.post('http://localhost:2334/api', json=data)

r = r.json()
str_result = r['str_result']
WW = r['W']
HH = r['H']

result = str_result.encode("latin1")
result = Image.frombytes('RGB', (WW, HH), result, 'raw')
result.save("result.jpg")
