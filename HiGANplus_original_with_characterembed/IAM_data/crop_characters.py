import cv2, h5py, os, pickle
import numpy as np
data_path = '/data/sidra/handwriting_generation_data/iam_data_higanplus/'
file_path = data_path + 'trnvalset_words64_OrgSz.hdf5'
plot_original_image = '/data/sidra/handwriting_generation_data/iam_data_higanplus/trnval_plot_crop_characters/'

if os.path.exists(file_path):
    h5f = h5py.File(file_path, 'r')
    imgs, lbs = h5f['imgs'][:], h5f['lbs'][:]
    img_seek_idxs, lb_seek_idxs = h5f['img_seek_idxs'][:], h5f['lb_seek_idxs'][:]
    img_lens, lb_lens = h5f['img_lens'][:], h5f['lb_lens'][:]
    wids = h5f['wids'][:]

total_idx = len(img_lens)
hash_char_boxes = {}
len_char_in_word = 15
read_src_mask_char = []
for idx in range(total_idx):
    print(idx, total_idx)
    img_seek_idx, img_len = img_seek_idxs[idx], img_lens[idx]
    lb_seek_idx, lb_len = lb_seek_idxs[idx], lb_lens[idx]
    img = imgs[:, img_seek_idx: img_seek_idx + img_len]
    image = img

    #'/home/tug85766/Text_Detection/CRAFT-Reimplementation-master/IAM_data/original_test_images_IAM_char/'
    w, h = img.shape[1], img.shape[0]
    total_patches = lb_len
    each_path_width = int(w/lb_len)
    char_images = []
    for p in range(total_patches):
        if (p+1)*each_path_width > w:
            patch_img = img[:, p * each_path_width:w]
        else:
            patch_img = img[:,p*each_path_width:(p+1)*each_path_width]
        hash_char_boxes[idx] = []
        ### todo: resize and crop
        patch_img = cv2.resize(patch_img, (32, 64), interpolation=cv2.INTER_CUBIC)
        char_images.append(patch_img)
        cv2.imwrite(plot_original_image + str(idx) + '_' + str(p) + '.png', patch_img)
    hash_char_boxes[idx] = char_images
    L = [np.zeros(len(char_images), dtype=float).tolist() + (np.NINF * np.ones(len_char_in_word - len(char_images), dtype=float)).tolist()] * len(char_images)
    # L = np.zeros(2, dtype=float).tolist() + (np.NINF * np.ones(len_char_in_word-2, dtype=float)).tolist()
    L1 = [(np.NINF * np.ones(len_char_in_word, dtype=float)).tolist()] * (len_char_in_word - len(char_images))
    read_src_mask_char.append(L + L1)
with open(data_path  + 'trnval_charboxes_IAM.pickle', 'wb') as handle:
    pickle.dump(hash_char_boxes, handle, protocol=pickle.HIGHEST_PROTOCOL)
