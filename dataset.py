import os
from random import randint

import cv2
import re
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from matplotlib import animation
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# helper function to make getting another batch of data easier


# from diffusion_training import output_img


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def make_pngs_anogan():
    dir = {
        "Train":     "./DATASETS/Train", "Test": "./DATASETS/Test",
        "Anomalous": "./DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1"
        }
    slices = {
        "17904": range(165, 205), "18428": range(177, 213), "18582": range(160, 190), "18638": range(160, 212),
        "18675": range(140, 200), "18716": range(135, 190), "18756": range(150, 205), "18863": range(130, 190),
        "18886": range(120, 180), "18975": range(170, 194), "19015": range(158, 195), "19085": range(155, 195),
        "19275": range(184, 213), "19277": range(158, 209), "19357": range(158, 210), "19398": range(164, 200),
        "19423": range(142, 200), "19567": range(160, 200), "19628": range(147, 210), "19691": range(155, 200),
        "19723": range(140, 170), "19849": range(150, 180)
        }
    center_crop = 235
    import os
    try:
        os.makedirs("./DATASETS/AnoGAN")
    except OSError:
        pass
    # for d_set in ["Train", "Test"]:
    #     try:
    #         os.makedirs(f"./DATASETS/AnoGAN/{d_set}")
    #     except OSError:
    #         pass
    #
    #     files = os.listdir(dir[d_set])
    #
    #     for volume_name in files:
    #         try:
    #             volume = np.load(f"{dir[d_set]}/{volume_name}/{volume_name}.npy")
    #         except (FileNotFoundError, NotADirectoryError) as e:
    #             continue
    #         for slice_idx in range(40, 120):
    #             image = volume[:, slice_idx:slice_idx + 1, :].reshape(256, 192).astype(np.float32)
    #             image = (image * 255).astype(np.int32)
    #             empty_image = np.zeros((256, center_crop))
    #             empty_image[:, 21:213] = image
    #             image = empty_image
    #             center = (image.shape[0] / 2, image.shape[1] / 2)
    #
    #             x = center[1] - center_crop / 2
    #             y = center[0] - center_crop / 2
    #             image = image[int(y):int(y + center_crop), int(x):int(x + center_crop)]
    #             image = cv2.resize(image, (64, 64))
    #             cv2.imwrite(f"./DATASETS/AnoGAN/{d_set}/{volume_name}-slice={slice_idx}.png", image)

    try:
        os.makedirs(f"./DATASETS/AnoGAN/Anomalous")
    except OSError:
        pass
    try:
        os.makedirs(f"./DATASETS/AnoGAN/Anomalous-mask")
    except OSError:
        pass
    files = os.listdir(f"{dir['Anomalous']}/raw_cleaned")
    center_crop = (175, 240)
    for volume_name in files:
        try:
            volume = np.load(f"{dir['Anomalous']}/raw_cleaned/{volume_name}")
            volume_mask = np.load(f"{dir['Anomalous']}/mask_cleaned/{volume_name}")
        except (FileNotFoundError, NotADirectoryError) as e:
            continue
        temp_range = slices[volume_name[:-4]]
        for slice_idx in np.linspace(temp_range.start + 5, temp_range.stop - 5, 4).astype(np.uint16):
            image = volume[slice_idx, ...].reshape(volume.shape[1], volume.shape[2]).astype(np.float32)
            image = (image * 255).astype(np.int32)
            empty_image = np.zeros((max(volume.shape[1], center_crop[0]), max(volume.shape[2], center_crop[1])))

            empty_image[9:165, :] = image
            image = empty_image
            center = (image.shape[0] / 2, image.shape[1] / 2)

            x = center[1] - center_crop[1] / 2
            y = center[0] - center_crop[0] / 2
            image = image[int(y):int(y + center_crop[0]), int(x):int(x + center_crop[1])]
            image = cv2.resize(image, (64, 64))
            cv2.imwrite(f"./DATASETS/AnoGAN/Anomalous/{volume_name}-slice={slice_idx}.png", image)

            image = volume_mask[slice_idx, ...].reshape(volume.shape[1], volume.shape[2]).astype(np.float32)
            image = (image * 255).astype(np.int32)
            empty_image = np.zeros((max(volume.shape[1], center_crop[0]), max(volume.shape[2], center_crop[1])))

            empty_image[9:165, :] = image
            image = empty_image
            center = (image.shape[0] / 2, image.shape[1] / 2)

            x = center[1] - center_crop[1] / 2
            y = center[0] - center_crop[0] / 2
            image = image[int(y):int(y + center_crop[0]), int(x):int(x + center_crop[1])]
            image = cv2.resize(image, (64, 64))
            cv2.imwrite(f"./DATASETS/AnoGAN/Anomalous-mask/{volume_name}-slice={slice_idx}.png", image)




def main(save_videos=True, bias_corrected=False, verbose=0):
    DATASET = "./DATASETS/CancerousDataset/EdinburghDataset"
    patients = os.listdir(DATASET)
    for i in [f"{DATASET}/Anomalous-T1/raw_new", f"{DATASET}/Anomalous-T1/mask_new"]:
        try:
            os.makedirs(i)
        except OSError:
            pass
    if save_videos:
        for i in [f"{DATASET}/Anomalous-T1/raw_new/videos", f"{DATASET}/Anomalous-T1/mask_new/videos"]:
            try:
                os.makedirs(i)
            except OSError:
                pass

    for patient in patients:
        try:
            patient_data = os.listdir(f"{DATASET}/{patient}")
        except:
            if verbose:
                print(f"{DATASET}/{patient} Not a directory")
            continue
        for data_folder in patient_data:
            if "COR_3D" in data_folder:
                try:
                    T1_files = os.listdir(f"{DATASET}/{patient}/{data_folder}")
                except:
                    if verbose:
                        print(f"{patient}/{data_folder} not a directory")
                    continue
                try:
                    mask_dir = os.listdir(f"{DATASET}/{patient}/tissue_classes")
                    for file in mask_dir:
                        if file.startswith("cleaned") and file.endswith(".nii"):
                            mask_file = file
                except:
                    if verbose:
                        print(f"{DATASET}/{patient}/tissue_classes dir not found")
                    return
                for t1 in T1_files:
                    if bias_corrected:
                        check = t1.endswith("corrected.nii")
                    else:
                        check = t1.startswith("anon")
                    if check and t1.endswith(".nii"):
                        # try:
                        # use slice 35-55
                        img = nib.load(f"{DATASET}/{patient}/{data_folder}/{t1}")
                        mask = nib.load(f"{DATASET}/{patient}/tissue_classes/{mask_file}").get_fdata()
                        image = img.get_fdata()
                        if verbose:
                            print(image.shape)
                        if bias_corrected:
                            # image.shape = (256, 156, 256)
                            image = np.rot90(image, 3, (0, 2))
                            image = np.flip(image, 1)
                            # image.shape = (256, 156, 256)
                        else:
                            image = np.transpose(image, (1, 2, 0))
                        mask = np.transpose(mask, (1, 2, 0))
                        if verbose:
                            print(image.shape)
                        image_mean = np.mean(image)
                        image_std = np.std(image)
                        img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
                        image = np.clip(image, img_range[0], img_range[1])
                        image = image / (img_range[1] - img_range[0])

                        np.save(
                                f"{DATASET}/Anomalous-T1/raw_new/{patient}.npy", image.astype(
                                        np.float32
                                        )
                                )
                        np.save(
                                f"{DATASET}/Anomalous-T1/mask_new/{patient}.npy", mask.astype(
                                        np.float32
                                        )
                                )
                        if verbose:
                            print(f"Saved {DATASET}/Anomalous-T1/mask/{patient}.npy")

                        if save_videos:
                            fig = plt.figure()
                            ims = []
                            for i in range(image.shape[0]):
                                tempImg = image[i:i + 1, :, :]
                                im = plt.imshow(
                                        tempImg.reshape(image.shape[1], image.shape[2]), cmap='gray', animated=True
                                        )
                                ims.append([im])

                            ani = animation.ArtistAnimation(
                                    fig, ims, interval=50, blit=True,
                                    repeat_delay=1000
                                    )

                            ani.save(f"{DATASET}/Anomalous-T1/raw_new/videos/{patient}.mp4")
                            if verbose:
                                print(f"Saved {DATASET}/Anomalous-T1/raw/videos/{patient}.mp4")
                            fig = plt.figure()
                            ims = []
                            for i in range(mask.shape[0]):
                                tempImg = mask[i:i + 1, :, :]
                                im = plt.imshow(
                                        tempImg.reshape(mask.shape[1], mask.shape[2]), cmap='gray', animated=True
                                        )
                                ims.append([im])

                            ani = animation.ArtistAnimation(
                                    fig, ims, interval=50, blit=True,
                                    repeat_delay=1000
                                    )

                            ani.save(f"{DATASET}/Anomalous-T1/mask_new/videos/{patient}.mp4")
                            if verbose:
                                print(mask.shape)
                                print(f"Saved {DATASET}/Anomalous-T1/raw/videos/{patient}.mp4")


def checkDataSet():
    resized = False
    mri_dataset = AnomalousMRIDataset(
            "DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1/raw", img_size=(256, 256),
            slice_selection="iterateUnknown", resized=resized
            # slice_selection="random"
            )

    dataset_loader = cycle(
            torch.utils.data.DataLoader(
                    mri_dataset,
                    batch_size=22, shuffle=True,
                    num_workers=2, drop_last=True
                    )
            )

    new = next(dataset_loader)

    image = new["image"]

    print(image.shape)
    from helpers import gridify_output
    print("Making Video for resized =", resized)
    fig = plt.figure()
    ims = []
    for i in range(0, image.shape[1], 2):
        tempImg = image[:, i, ...].reshape(image.shape[0], 1, image.shape[2], image.shape[3])
        im = plt.imshow(
                gridify_output(tempImg, 5), cmap='gray',
                animated=True
                )
        ims.append([im])

    ani = animation.ArtistAnimation(
            fig, ims, interval=50, blit=True,
            repeat_delay=1000
            )

    ani.save(f"./CancerousDataset/EdinburghDataset/Anomalous-T1/video-resized={resized}.mp4")


def output_videos_for_dataset():
    folders = os.listdir("/Users/jules/Downloads/19085/")
    folders.sort()
    print(f"Folders: {folders}")
    for folder in folders:
        try:
            files_folder = os.listdir("/Users/jules/Downloads/19085/" + folder)
        except:
            print(f"{folder} not a folder")
            exit()

        for file in files_folder:
            try:
                if file[-4:] == ".nii":
                    # try:
                    # use slice 35-55
                    img = nib.load("/Users/jules/Downloads/19085/" + folder + "/" + file)
                    image = img.get_fdata()
                    image = np.rot90(image, 3, (0, 2))
                    print(f"{folder}/{file} has shape {image.shape}")
                    outputImg = np.zeros((256, 256, 310))
                    for i in range(image.shape[1]):
                        tempImg = image[:, i:i + 1, :].reshape(image.shape[0], image.shape[2])
                        img_sm = cv2.resize(tempImg, (310, 256), interpolation=cv2.INTER_CUBIC)
                        outputImg[i, :, :] = img_sm

                    image = outputImg
                    print(f"{folder}/{file} has shape {image.shape}")
                    fig = plt.figure()
                    ims = []
                    for i in range(image.shape[0]):
                        tempImg = image[i:i + 1, :, :]
                        im = plt.imshow(tempImg.reshape(image.shape[1], image.shape[2]), cmap='gray', animated=True)
                        ims.append([im])

                    ani = animation.ArtistAnimation(
                            fig, ims, interval=50, blit=True,
                            repeat_delay=1000
                            )

                    ani.save("/Users/jules/Downloads/19085/" + folder + "/" + file + ".mp4")
                    plt.close(fig)

            except:
                print(
                        f"--------------------------------------{folder}/{file} FAILED TO SAVE VIDEO ------------------------------------------------"
                        )



def load_datasets_for_test():
    args = {'img_size': (256, 256), 'random_slice': True, 'Batch_Size': 20}
    training, testing = init_datasets("./", args)

    ano_dataset = AnomalousMRIDataset(
            ROOT_DIR=f'./dataset/test/FLAIR_test', MASK_DIR='dataset/test/mask_test', img_size=args['img_size'],
            slice_selection="random", resized=False
            )

    train_loader = init_dataset_loader(training, args)
    ano_loader = init_dataset_loader(ano_dataset, args)

    train_loader_iter = iter(train_loader)
    ano_loader_iter = iter(ano_loader)
    
    for i in range(5):
        train_data_object = next(train_loader_iter)
        ano_data_object = next(ano_loader_iter)
        
        output = torch.cat((train_data_object["image"][:10], ano_data_object["image"][:10]))
        plt.imshow(helpers.gridify_output(output, 5), cmap='gray')
        plt.show()
        plt.pause(0.0001)


def init_datasets(ROOT_DIR, args):
    training_dataset = MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}dataset/train/hc_FLAIR', img_size=args['img_size'], random_slice=args['random_slice']
            )
    testing_dataset = MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}dataset/test/FLAIR_test', img_size=args['img_size'], random_slice=args['random_slice']
            )
    return training_dataset, testing_dataset


def init_dataset_loader(mri_dataset, args, shuffle=True):
    dataset_loader = cycle(
            torch.utils.data.DataLoader(
                    mri_dataset,
                    batch_size=args['Batch_Size'], shuffle=shuffle,
                    num_workers=0, drop_last=True
                    )
            )
    # dataset_loader = torch.utils.data.DataLoader(
    #                     mri_dataset,
    #                     batch_size=args['Batch_Size'], shuffle=shuffle,
    #                     num_workers=0, drop_last=True
    #                  )
            

    return dataset_loader

class ResizeWithPadding:
    def __init__(self, target_size=256):
        self.target_size = target_size

    def __call__(self, image):
        width, height = image.size
        
        # 计算缩放比例
        if width > height:
            new_width = self.target_size
            new_height = int(height * (self.target_size / width))
        else:
            new_height = self.target_size
            new_width = int(width * (self.target_size / height))
        
        # 缩放图像
        resize_transform = transforms.Resize(
            (new_height, new_width), 
            transforms.InterpolationMode.BILINEAR
        )
        resized_image = resize_transform(image)
        
        # 计算填充尺寸
        pad_left = (self.target_size - new_width) // 2
        pad_top = (self.target_size - new_height) // 2
        pad_right = self.target_size - new_width - pad_left
        pad_bottom = self.target_size - new_height - pad_top
        
        # 填充图像
        pad_transform = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0)
        padded_image = pad_transform(resized_image)
        
        return padded_image

class MRIDataset(Dataset):
    """Healthy MRI dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=256, random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transforms.Compose(
            [
                # transforms.RandomAffine(3, translate=(0.02, 0.09)),
                # transforms.CenterCrop(235),
                # transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(256),
                ResizeWithPadding(target_size=256),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ]
        ) if not transform else transform

        self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.ROOT_DIR, self.filenames[idx])
        if os.path.exists(img_path):
            image = Image.open(img_path)
        else:
            image = None
            print(f"Error: Invalid image path {img_path}")

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, "filenames": self.filenames[idx]}
        return sample


class AnomalousMRIDataset(Dataset):
    """Anomalous MRI dataset."""

    def __init__(
            self, ROOT_DIR, MASK_DIR, transform=None, img_size=256, slice_selection="random", resized=False,
            cleaned=True
            ):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            img_size: size of each 2D dataset image
            slice_selection: "random" = randomly selects a slice from the image
                             "iterateKnown" = iterates between ranges of tumour using slice data
                             "iterateUnKnown" = iterates through whole MRI volume
        """
        self.transform = transforms.Compose([
                 # transforms.CenterCrop((175, 240)),
                 # transforms.RandomAffine(0, translate=(0.02, 0.1)),
                 # transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                 # transforms.CenterCrop(256),
                 ResizeWithPadding(target_size=256),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5), (0.5))
                ]) if not transform else transform
        self.img_size = img_size
        self.resized = resized

        self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        self.ROOT_DIR = ROOT_DIR
        self.MASK_DIR = MASK_DIR
        self.slice_selection = slice_selection

    def __len__(self):
        return len(self.filenames)
 
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.ROOT_DIR, self.filenames[idx])
        if os.path.exists(img_path):
            image = Image.open(img_path)
        else:
            image = None
            print(f"Error: Invalid image path {img_path}")

        if self.transform:
            image = self.transform(image)

        sample = {}
        # mask: 
        # "D:\Datasets\OPMED_proc\test\mask_test\sub-00003_acq-T2sel_FLAIR_roi_88.png"
        # test_sample: 
        # "D:\Datasets\OPMED_proc\test\FLAIR_test\sub-00003_acq-T2sel_FLAIR_88.png"
        match = re.search(r"([A-Za-z]+)_(\d+\.png)$", self.filenames[idx])
        if match:
            mask_filename = self.filenames[idx][:match.start(2)] + "roi_" + self.filenames[idx][match.start(2):]
            mask_path = os.path.join(self.MASK_DIR, mask_filename)
            if os.path.exists(img_path):
                mask = Image.open(mask_path)
                
                if self.transform:
                    mask = self.transform(mask)
                    mask = (mask > 0).float()
            else:
                mask = None
            
        # missing slices, wonder if it is necessary
        sample["image"] = image
        sample["filenames"] = self.filenames[idx]
        sample["mask"] = mask
        return sample


def load_CIFAR10(args, train=True):
    return torch.utils.data.DataLoader(
            datasets.CIFAR10(
                    "./DATASETS/CIFAR10", train=train, download=True, transform=transforms
                        .Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

                                ]
                            )
                    ),
            shuffle=True, batch_size=args["Batch_Size"], drop_last=True
            )


if __name__ == "__main__":
    
    # get_segmented_labels(True)
    # main(False, False, 0)
    # make_pngs_anogan()
    import matplotlib.pyplot as plt
    import helpers

    load_datasets_for_test()
    # d_set = AnomalousMRIDataset(
    #         ROOT_DIR='./DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1', img_size=(256, 256),
    #         slice_selection="iterateKnown_restricted", resized=False
    #         )
    # loader = init_dataset_loader(d_set, {"Batch_Size": 16})

    # for i in range(4):
    #     new = next(loader)
    #     plt.imshow(helpers.gridify_output(new["image"], 4), cmap="gray")
    #     plt.show()
    #     plt.imshow(helpers.gridify_output(new["mask"], 4), cmap="gray")
    #     plt.show()
    #     plt.pause(1)
