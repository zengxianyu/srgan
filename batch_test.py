import os
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import pdb
import tensorflow as tf
import tensorlayer as tl
from model import get_G

def SR_PIL(net, img):
    valid_lr_img = np.array(img)
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
    valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
    size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

    out = net(valid_lr_img).numpy()
    out = (out+1)*127.5
    out = out[0].astype(np.uint8)
    out = Image.fromarray(out)

    #print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    #print("[*] save images")
    #tl.vis.save_image(out[0], os.path.join(save_dir, name))
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='./test_api/images',
                        help='input')
    parser.add_argument('--output', type=str, default='./results',
                        help='path to save')
    parser.add_argument('--test_iter', type=int, default=4,
                        help='path to save')
    parser.add_argument('--dilate', type=int, default=0,
                        help='dilate mask')
    parser.add_argument('--param', type=str, default='./models/g.npz',
                        help='path to parameters file')
    parser.add_argument('--nogpu', action='store_true')

    opt, _ = parser.parse_known_args()


    checkpoint_dir = opt.param#"../files/seqsalskip_305000_g.pth"
    path_img = opt.image#"./988.jpg"
    save_dir = opt.output
    nogpu = opt.nogpu#False

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    G = get_G([1, None, None, 3])
    G.load_weights(checkpoint_dir)
    G.eval()

    for name in tqdm(os.listdir(path_img)):
        img = Image.open(os.path.join(path_img, name)).convert("RGB")
        out = SR_PIL(G, img)
        out.save(os.path.join(save_dir, name))
