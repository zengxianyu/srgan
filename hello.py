import os
import requests
from flask import Flask, url_for, render_template, request, redirect, send_from_directory
from flask import jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import base64
import io
import numpy as np
import random
import pdb
import timeit
import tensorflow as tf
import tensorlayer as tl
from model import get_G
from batch_test import SR_PIL

from datetime import datetime


app = Flask(__name__)

checkpoint_dir = "./models/g.npz"
G = get_G([1, None, None, 3])
G.load_weights(checkpoint_dir)
G.eval()

@app.route('/api', methods=['POST'])
def api_seq():
    now = datetime.now()
    data = request.json
    str_img = data['str_img']
    W = data['width']
    H = data['height']
    mode_img = data['mode_img']
    img = str_img.encode("latin1")
    img = Image.frombytes(mode_img, (W, H), img, 'raw')
    print("super res {}x{}".format(W,H))
    result = SR_PIL(G, img)
    WW, HH = result.size
    str_result = result.tobytes().decode("latin1")
    data_result = {'str_result':str_result, 'W': WW, 'H': HH}
    return jsonify(data_result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=1818,
                                help='web service port')
    opt = parser.parse_args()

    app.run(host='0.0.0.0', debug=False, port=opt.port, threaded=False)
