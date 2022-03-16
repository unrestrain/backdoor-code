def convert_nparray_to_csv(image_list, label_list, outf, description='image_convert'):
    if np.max(image_list[0])<=1:
        flat_to_255 = True
    o = open(outf, 'a+')
    for i in tqdm(range(len(label_list)), desc=description):
        o.write(str(label_list[i]))
        o.write(',')
        if flat_to_255:
            o.write(','.join(str(int(x)) for x in (image_list[i]*255).reshape(-1).tolist()))
        else:
            o.write(','.join(str(x) for x in image_list[i].reshape(-1).tolist()))
        o.write('\n')
    o.close()
