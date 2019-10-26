#!/usr/bin/env python3
from charades_eval import charades_eval

aps = []
corlocs = []
detpath = imdb._data_path+'/results/ws_det_' +imdb._image_set +'_{:s}.txt'
annopath = imdb._data_path+'/ImageSets/annotations.pkl'
imagelistfile= imdb._data_path+'/ImageSets/charades_action_cls_test.txt'
with open(annopath, 'rb') as f:
    try:
        recs = pickle.load(f)
    except:
        recs = pickle.load(f, encoding='bytes')
for cls in imdb._obj_classes:
    if cls == "__BACKGROUND__": continue
    rec, prec, ap, corloc = charades_eval(detpath, recs, imagelistfile, cls, ovthresh=0.5)
    aps += [ap]
    corlocs += [corloc]
    print('AP for {} = {:.4f}'.format(cls, ap))
    print('CorLoc for {} = {:.4f}'.format(cls, corloc))
print('Mean AP = {:.4f}'.format(np.mean(aps)))
print('Mean CorLoc = {:.4f}'.format(np.mean(corlocs)))
