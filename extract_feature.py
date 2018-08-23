import os
import h5py
import argparse
from imageretrievalnet import init_network, extract_vectors
import numpy as np


parser = argparse.ArgumentParser(description='CNN Image Retrieval Extract Feature')

# options
parser.add_argument('--image-size', '-imsize', default=320, type=int, metavar='N',
                    help='size of longer image side used for extracting feature (default: 480)')
parser.add_argument('--model-choose', '-model', default='vgg16', type=str,
                    help='model for extracting feature (default: vgg16)')
parser.add_argument('--feature-path', '-spath', default='./', type=str,
                    help='path for save feature (default: ./feature/)')
parser.add_argument('--image-path', '-impath', default='C:/Users/MS/Documents/datasets/trademark', type=str,
                    help='path for image (default: ./data/)')

def get_imlist(path):
    imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    imlist.sort()
    print(len(imlist))
    return imlist

def save_feature(images, model_choose, image_size, feature_path, istest):


    model = init_network(model=model_choose)

    vecs, vecs_MAC, vecs_SPoC, vecs_RMAC, vecs_RAMAC, name_list = extract_vectors(model, images, image_size, print_freq=1)

    print(name_list)

    for i,_ in enumerate(vecs):
        vecs[i]=vecs[i].detach().numpy()
        vecs_MAC[i] = vecs_MAC[i].detach().numpy()
        vecs_SPoC[i] = vecs_SPoC[i].detach().numpy()
        vecs_RMAC[i] = vecs_RMAC[i].detach().numpy()
        vecs_RAMAC[i] = vecs_RAMAC[i].detach().numpy()


    feats= np.vstack(vecs)
    feats_MAC = np.vstack(vecs_MAC)
    feats_SPoC =  np.vstack(vecs_SPoC)
    feats_RMAC =  np.vstack(vecs_RMAC)
    feats_RAMAC = np.vstack(vecs_RAMAC)

    name_list=[os.path.basename(n).encode("ascii", "ignore") for n in name_list]



    print(feats.shape)
    print(feats_MAC.shape)
    print(feats_SPoC.shape)
    print(feats_RMAC.shape)
    print(feats_RAMAC.shape)
    print(name_list)



    if os.path.exists(feature_path) == False:
        os.mkdir(feature_path)
    if istest == 0:
        name = feature_path+'feat_'+model_choose+'.h5'
    else:
        name = feature_path+'feat_test_'+model_choose+'.h5'

    h5f = h5py.File(name, 'w')
    h5f.create_dataset('Pool5', data=feats)
    h5f.create_dataset('MAC', data = feats_MAC)
    h5f.create_dataset('SPoC', data = feats_SPoC)
    h5f.create_dataset('RMAC', data = feats_RMAC)
    h5f.create_dataset('RAMAC', data = feats_RAMAC)
    h5f.create_dataset('name_list', data = name_list)
    h5f.close()
    print('\r>>>> save to {}.'.format(name))


def main():
    args = parser.parse_args()
    images = get_imlist(args.image_path)

    save_feature(images, args.model_choose, args.image_size, args.feature_path, 0)


if __name__ == '__main__':
    main()
