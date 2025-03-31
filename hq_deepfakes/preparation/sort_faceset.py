import os
import logging 
import argparse


import cv2
import numpy as np
import torch
from torchvision import transforms, models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class Sorter():
    def __init__(self, data_path, method):
        self.data_path = data_path
        self.method = method
        self.batch_size = 32

        self.save_path = self.data_path

    # core
    def process(self):
        img_list = self._load_images()
        sort_method = getattr(Sorter, "_sort_" + self.method)
        sorted = sort_method(self, img_list)
        self._rename_images(sorted)
        logging.info("Done.")

    # load save stuff
    def _load_images(self):
        logging.info("Loading images")

        flag = cv2.IMREAD_UNCHANGED if self.method in ["face", "vgg_face"] else cv2.IMREAD_GRAYSCALE
        img_list= [(f, cv2.imread(os.path.join(self.save_path, f), flag)) for f in os.listdir(self.save_path) if f.endswith(".png")]
        return img_list

    def _rename_images(self, sorted):
        logging.info("Renaming files...")
        if self.method == "face":
            names, labels = sorted
            for label, path in zip(labels, names):
                label = str(label)
                if len(label) == 1:
                    label = str(0) + label
                new_name = str(label) + "---" + path
                os.rename(os.path.join(self.save_path, path), os.path.join(self.save_path, new_name))
        else:
            for i in range(len(sorted)):
                new_name = str(i) + "---" + sorted[i][0]
                os.rename(os.path.join(self.save_path, sorted[i][0]), os.path.join(self.save_path, new_name))

    # sorting methods
    def _sort_hist(self, img_list):
        logging.info("Computing hists...")
        for i in range(len(img_list)):
            img = img_list[i][1]
            name = img_list[i][0]
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            img_list[i] = (name, hist)

        # compare hists
        img_list_len = len(img_list)

        logging.info("Comparing hists...")
        for i in range(0, img_list_len-1):
            min_score = float("inf")
            j_min_score = i + 1
            for j in range(i+1, img_list_len):
                score = cv2.compareHist(img_list[i][1], img_list[j][1], cv2.HISTCMP_BHATTACHARYYA)
                if score < min_score:
                    min_score = score
                    j_min_score = j
            (img_list[i + 1], img_list[j_min_score]) = (img_list[j_min_score], img_list[i + 1])

        return img_list

    def _sort_blur(self, img_list):
        logging.info("Computing blur scores...")
        for i in range(len(img_list)):
            img = img_list[i][1]
            name = img_list[i][0]
            crop = slice(img.shape[0] // 2 - 75, img.shape[0] // 2 + 75)
            img = img[crop, crop]
            blur_map = cv2.Laplacian(img, cv2.CV_32F)
            score = np.var(blur_map) / np.sqrt(img.shape[0] * img.shape[1])

            img_list[i] = (name, score)

        # sort
        img_list = sorted(img_list, key=lambda x: x[1], reverse=True)
        return img_list

    def _sort_blur_fft(self, img_list):
        logging.info("Computing blur fft scores...")
        for i in range(len(img_list)):
            img = img_list[i][1]
            name = img_list[i][0]
            crop = slice(img.shape[0] // 2 - 200 //2, img.shape[0] // 2 + 200 // 2)
            img = img[crop, crop]

            height, width = img.shape
            c_height, c_width = (int(height / 2.0), int(width / 2.0))
            fft = np.fft.fft2(img)
            fft_shift = np.fft.fftshift(fft)
            fft_shift[c_height - 75:c_height + 75, c_width - 75:c_width + 75] = 0
            ifft_shift = np.fft.ifftshift(fft_shift)
            shift_back = np.fft.ifft2(ifft_shift)
            magnitude = np.log(np.abs(shift_back))
            score = np.mean(magnitude)    

            img_list[i] = (name, score)

        # sort
        img_list = sorted(img_list, key=lambda x: x[1], reverse=True)
        return img_list


    def _sort_face(self, img_list, num_clusters=10):
        logging.info("Computing vgg face features...")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        net = models.vgg16(pretrained=True)
        net.classifier = net.classifier[:5]
        net.to("cuda")
        net.eval()

        data = {}

        with torch.no_grad():
            # compute features
            num_images = len(img_list)
            for i in range(0, num_images, self.batch_size):
                start = i
                end = i+self.batch_size if i+self.batch_size < num_images else num_images

                batch_list = img_list[start:end]
                batch = self._get_batch(batch_list, transform)
                batch = batch.to("cuda")

                features = net(batch).cpu().numpy()
                
                for j in range(len(batch_list)):
                    key = batch_list[j][0]
                    val = features[j]

                    data[key] = val

        names = list(data.keys())
        feats = np.array(list(data.values()))

        # pca
        logging.info("Start PCA...")
        pca = PCA(n_components=100, random_state=42)
        pca.fit(feats)
        x = pca.transform(feats)

        logging.info("Start kmeans...")
        # clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(x)
        labels = kmeans.labels_

        return names, labels


    # helper
    def _get_batch(self, batch_list, transform):
        batch = torch.zeros(len(batch_list), 3, 224 , 224)

        for i in range(len(batch_list)):
            img = batch_list[i][1]
            batch[i] = transform(img)

        return batch




parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="specify path to the data directory")
parser.add_argument("--method", type=str, help="specify method to sort with (one of: hist, face, blur, blur_fft")
parser.add_argument("--log", default="INFO", type=str)

args = parser.parse_args()

if __name__ == '__main__':
    numeric_logging_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(level=numeric_logging_level)

    input_dict = vars(args)
    del input_dict["log"]

    sorter = Sorter(**input_dict)
    sorter.process()