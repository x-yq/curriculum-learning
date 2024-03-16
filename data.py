from PIL import Image
import torch.utils.data as data
import torch
import torchvision.transforms as transforms


def prepare_dataset(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    global transform_train
    transform_train = transform_test

    global all_train_set, train0_70, train71_80, train81_90, train91_100
    all_train_set, train0_70, train71_80, train81_90, train91_100 = [], [], [], [], []

    with open("41-80_anno.txt", 'r') as f:
        all_train_set = f.readlines()
        size = len(all_train_set)
        index = int(size * 0.7)
        length = int(size * 0.1)
        train0_70 = all_train_set[:index+1]
        train71_80 = all_train_set[index+1: index+1+length]
        train81_90 = all_train_set[index + 1 + length:index + 1 + 2*length]
        train91_100 = all_train_set[index + 1 + 2*length:index + 1 + 3*length]

    global all_val_set, all_test_set
    all_val_set, all_test_set = [], []

    with open("01-40_anno.txt", 'r') as f:
        for line in f.readlines():
            line = line.strip()
            temp = [int(x) for x in line.split()]
            if temp[0] <= 30:
                all_test_set.append(temp)
            else:
                all_val_set.append(temp)

def prepare_dataloader_anticl(args, stage):

    '''
    TRAIN: 4 Stage，4 interval of difficulty：
      0-70p 71-80p 81-90p 91-100p /all     
    0  5%    35%    35%    35%    0.14
    1  15%    25%    25%    30%    0.185
    2  30%    25%    25%    25%   0.285
    3  50%     15%    15%    10%   0.39
    '''

    if stage == "test":
        test_loader = data.DataLoader(ImageDataset(all_test_set, args.width, args.height, transform=transform_train),
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.j,
                                     pin_memory=args.pin_memory)
        return test_loader

    train_patch = [train0_70, train71_80, train81_90, train91_100]
    t0_s, t1_s, t2_s, t3_s = len(train_patch[0]), len(train_patch[1]), len(train_patch[2]), len(train_patch[3])

    percentile = [
        [0.05*t0_s, 0.35*t1_s, 0.35*t2_s, 0.35*t3_s],
        [0.2*t0_s, 0.6*t1_s, 0.6*t2_s, 0.65*t3_s],
        [0.5*t0_s, 0.85*t1_s, 0.85*t2_s, 0.9*t3_s],
        [t0_s, t1_s, t2_s, t3_s]
    ]

    train_set = []
    for i in range(4):
        train_set = train_set + train_patch[i][: int(percentile[stage][i])]
    print('Stage: ' , stage)
    print('Presenting train set size (proportion):  ', len(train_set)/ len(all_train_set))

    train_list = []
    for i in range(len(train_set)):
        train_set[i] = train_set[i].strip()
        temp = [int(x) for x in train_set[i].split()]
        train_list.append(temp)
    train_set = train_list

    train_loader = data.DataLoader(ImageDataset(train_set, args.width, args.height, transform=transform_train),
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.j,
                                   pin_memory=args.pin_memory)


    val_interval = int(0.25 * len(all_val_set))
    val_set = all_val_set[:(stage+1) * val_interval]
    print('Stage: ' , stage)
    print('Presenting val set size (proportion): ', len(val_set)/ len(all_val_set))

    val_loader = data.DataLoader(ImageDataset(val_set, args.width, args.height, transform=transform_train),
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.j,
                                   pin_memory=args.pin_memory)

    return train_loader, val_loader


def prepare_dataloader(args, stage):

    '''
    TRAIN: 5 Stage，4 interval of difficulty：
      0-70p 71-80p 81-90p 91-100p /all
    0  40%    5%     0%     0%    0.285
    1  30%    10%    10%    0%    0.23
    2  20%    20%    20%    20%   0.2
    3  10%    30%    30%    30%   0.16
    4  0%     35%    40%    50%   0.125
    '''

    '''
    TRAIN: 6 Stage，4 interval of difficulty：
      0-70p 71-80p 81-90p 91-100p /all
    0  40%    5%     0%     0%    0.285
    1  30%    10%    10%    0%    0.23
    2  15%    15%    15%    20%   0.155
    3  10%    20%    20%    25%   0.135
    4  5%     25%    25%    25%   0.11
    5  0%     25%    30%    30%   0.08
    '''

    '''
    TRAIN: 4 Stage，4 interval of difficulty：
      0-70p 71-80p 81-90p 91-100p /all     
    0  50%    15%    15%    10%    0.39
    1  30%    25%    25%    25%    0.285
    2  15%    25%    25%    30%   0.185
    3  5%     35%    35%    35%   0.14
    '''

    if stage == "test":
        test_loader = data.DataLoader(ImageDataset(all_test_set, args.width, args.height, transform=transform_train),
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.j,
                                     pin_memory=args.pin_memory)
        return test_loader

    train_patch = [train0_70, train71_80, train81_90, train91_100]
    t0_s, t1_s, t2_s, t3_s = len(train_patch[0]), len(train_patch[1]), len(train_patch[2]), len(train_patch[3])

    # ## 4 stages
    # percentile = [
    #     [0.5*t0_s, 0.15*t1_s, 0.15*t2_s, 0.1*t3_s],
    #     [0.8*t0_s, 0.40*t1_s, 0.4*t2_s, 0.35*t3_s],
    #     [0.95*t0_s, 0.65*t1_s, 0.65*t2_s, 0.65*t3_s],
    #     [t0_s, t1_s, t2_s, t3_s]
    # ]

    ## 5 stages
    percentile = [
        [0.4*t0_s, 0.05*t1_s, 0*t2_s, 0*t3_s],
        [0.7*t0_s, 0.15*t1_s, 0.1*t2_s, 0*t3_s],
        [0.9*t0_s, 0.35*t1_s, 0.3*t2_s, 0.2*t3_s],
        [t0_s, 0.65*t1_s, 0.6*t2_s, 0.5*t3_s],
        [t0_s, t1_s, t2_s, t3_s]
    ]

    # ## 6 stages
    # percentile = [
    #     [0.4*t0_s, 0.05*t1_s, 0*t2_s, 0*t3_s],
    #     [0.7*t0_s, 0.15*t1_s, 0.1*t2_s, 0*t3_s],
    #     [0.85*t0_s, 0.3*t1_s, 0.25*t2_s, 0.2*t3_s],
    #     [0.95*t0_s, 0.5*t1_s, 0.45*t2_s, 0.45*t3_s],
    #     [t0_s, 0.75*t1_s, 0.7*t2_s, 0.7*t3_s],
    #     [t0_s, t1_s, t2_s, t3_s]
    # ]

    train_set = []
    for i in range(4):
        train_set = train_set + train_patch[i][: int(percentile[stage][i])]
    print('Stage: ' , stage)
    print('Presenting train set size (proportion):  ', len(train_set)/ len(all_train_set))

    train_list = []
    for i in range(len(train_set)):
        train_set[i] = train_set[i].strip()
        temp = [int(x) for x in train_set[i].split()]
        train_list.append(temp)
    train_set = train_list

    train_loader = data.DataLoader(ImageDataset(train_set, args.width, args.height, transform=transform_train),
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.j,
                                   pin_memory=args.pin_memory)


    val_interval = int(0.2 * len(all_val_set))
    val_set = all_val_set[:(stage+1) * val_interval]
    print('Stage: ' , stage)
    print('Presenting val set size (proportion): ', len(val_set)/ len(all_val_set))

    val_loader = data.DataLoader(ImageDataset(val_set, args.width, args.height, transform=transform_train),
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.j,
                                   pin_memory=args.pin_memory)

    return train_loader, val_loader

class ImageDataset(data.Dataset):
    def __init__(self, anno, width, height, transform=None):
        super(ImageDataset).__init__()
        self.anno = anno
        self.transform = transform
        self.width = width
        self.height = height

    def __getitem__(self, idx):
        anno = self.anno[idx]
        home_path = "/mnt/ceph/tco/TCO-Students/Projects/KP_curriculum_learning/cholec80/frames_1fps/"
        folder_path = "%02d" % anno[0]
        frame_path = "%08d" % anno[1]
        path = home_path + folder_path + "/" + frame_path + ".jpg"
        label = torch.tensor(anno[2:])
        img = self.load_image(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def load_image(self, img_file):

        im = Image.open(img_file)
        w = im.width
        h = im.height
        height2 = int(self.width * (h / w))
        offset_y = (self.height - height2) // 2

        img_y = im.resize((self.width, height2))
        img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        img.paste(img_y, box=(0, offset_y))
        return img

    def __len__(self):
        return len(self.anno)


