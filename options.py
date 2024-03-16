import argparse

parser = argparse.ArgumentParser( description="Model parameters for surgical instruments classification.")

parser.add_argument("--data_folder", type=str, default="", help="Folder containing the frames at 1 fps")
parser.add_argument("--trial_name", type=str, default="", help="name for training run")
parser.add_argument("--annotation_folder", type=str, default="", help="Folder containing the frames annotations")
parser.add_argument("--output_folder", type=str, default="", help="Folder to store the outputs of the training process")

parser.add_argument("--width", type=int, default=256, help="image width")
parser.add_argument("--height", type=int, default=256, help="image height")
parser.add_argument("--epochs", type=int, default=100, help="epochs to train")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--learning_rate", type=int, default=1e-4, help="learning rate")
parser.add_argument("--weight_decay", type=int, default=1e-5, help="weight decay")
parser.add_argument("--j", type=int, default=4, help="Num workers")
parser.add_argument("--pin_memory", type=int, default=True, help="Pin memory for dataloader")
 
