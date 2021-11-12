import os.path as osp
import os
import numpy as np
import cv2

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


if __name__ == '__main__':
    # data_path = '/export/wei.zhang/datasets/mixdata/MOT17/images/train'
    data_name_path = '/export/wei.zhang/PycharmProjects/FairMOT/src/data/mot17.train'
    output_dir = '/export/wei.zhang/datasets/generated_reid_datasets/MOT17/'

    dataset_path = '/export/wei.zhang/datasets/mixdata'

    with open(data_name_path, 'r') as file:
        img_files = file.readlines()
        img_files = [x.replace('\n', '') for x in img_files]
        img_files = list(filter(lambda x: len(x) > 0, img_files))

    label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                        for x in img_files]

    for files_index in range(len(img_files)):
        img_path = img_files[files_index]
        label_path = label_files[files_index]

        seq_name = img_path.split('/')[-3]
        img_path = osp.join(dataset_path, img_path)
        label_path = osp.join(dataset_path, label_path)
        img = cv2.imread(img_path)  # BGR
        h, w, _ = img.shape

        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4] / 2)
            labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5] / 2)
            labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4] / 2)
            labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5] / 2)

            for item in labels:

                x1 = np.clip(int(item[2]), 0, w)
                y1 = np.clip(int(item[3]), 0, h)
                x2 = np.clip(int(item[4])+1, 0, w)
                y2 = np.clip(int(item[5])+1, 0, h)



                person_id = int(item[1])

                img_save_path = osp.join(output_dir, seq_name, str(person_id))
                mkdirs(img_save_path)
                img_save_name = osp.join(img_save_path, img_path.split('/')[-1])
                crop_img = img[y1:y2, x1:x2, :]
                print(img_path)

                try:
                    cv2.imwrite(img_save_name, crop_img)
                except:
                    pass
        else:
            assert 0


