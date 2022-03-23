import os

def genDatasetForTrojaiFromTorchDataset(dataset, datasetDesc, outf, datasetdir, train=True, description='generate dataset for trojai'):
    os.makedirs(datasetdir, exist_ok=True)
    o = open(os.path.join(datasetdir, outf), 'w')
    o.write('file,label\n')
    for i in tqdm(range(len(dataset)), desc=description):
        if train:
            os.makedirs(datasetdir + '/data/train',exist_ok=True)
            filename = datasetdir + '/data/train/' + datasetDesc+'_'+str(i) + '.jpg'
        else:
            os.makedirs(datasetdir + '/data/test', exist_ok=True)
            filename = datasetdir + '/data/test/' + datasetDesc+'_'+str(i) + '.jpg'
        label = str(dataset[i][1])
        mp.imsave(filename, dataset[i][0])
        if train:
            o.write('data/train/' + datasetDesc+'_'+str(i) + '.jpg')
        else:
            o.write('data/test/' + datasetDesc+'_'+str(i) + '.jpg')
        o.write(',')
        o.write(label)
        o.write('\n')
    o.close()
    return os.path.join(datasetdir, outf)
