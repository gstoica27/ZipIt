import pickle

from torchvision.models.resnet import resnet50

def load_rn50_rebasin(a, b, device):
    L = [0, 1, 2, 3, 4]

    vals = {}

    for i, fi in enumerate(L):
        for j, fj in enumerate(L[i+1:]):
            vals[(min(fi, fj), max(fi, fj))] = (fj, fi, fi)
    

    vals = vals[(a, b)]

    with open('./checkpoints/imnet200rebasin/resnet50_{}_to_{}_and_{}.pkl'.format(*vals), 'rb') as f:
        sd = pickle.load(f)
    

    for k, v in list(sd.items()):
        del sd[k]
        # k = k.replace('batch_stats.', '').replace('params.', '').replace('linear', 'fc').replace('shortcut', 'downsample')
        # k = k.replace('scale', 'weight')
        k = k.replace('module.', '')
        sd[k] = v



    model = resnet50()
    model.load_state_dict(sd)

    return model.to(device)
