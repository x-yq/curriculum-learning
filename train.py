import datetime
import os
import random
import time
from sys import stderr

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

from data import prepare_dataset, prepare_dataloader, prepare_dataloader_anticl
from options import parser


def forward_data(data_loader, mode, model, optimizer, criterion, args, device):
    assert (mode in {"train", "val", "test"})

    if mode == "train":
        model.train()
    elif mode in {"val", "test"}:
        model.eval()

    cumulative_loss = 0

    y_score = None
    y_true = None

    for index, (images, labels_cpu) in enumerate(data_loader):

        # Forward Pass
        outputs = model(images.to(device))

        loss = criterion(outputs, labels_cpu.to(device).float())
        cumulative_loss += loss.item() * labels_cpu.size(0)

        if mode == "train":
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if index == 0:
            y_true = labels_cpu.numpy()
            y_score = torch.round(torch.nn.Sigmoid()(outputs)).detach().cpu().numpy()
        else:
            y_true = np.concatenate((y_true, labels_cpu.numpy()))
            y_score = np.concatenate((y_score, torch.round(torch.nn.Sigmoid()(outputs)).detach().cpu().numpy()))

    f1 = f1_score(y_true, y_score, average="macro")
    if mode == "test":
       f1 = []
       for i in range(0, 7):
           f1.append(f1_score(y_true[:, i], y_score[:, i], average="macro"))
    else:
       f1 = f1_score(y_true, y_score, average="macro")

    cumulative_loss /= len(data_loader.dataset)

    return f1, cumulative_loss, model, optimizer


def logging(mode, writer, log, epoch, f1, loss):

    if mode == "test":
        log(f"Epoche {epoch} {mode}: (loss {loss:.4f}, F1 {np.asarray(f1)})")
    else:
        log(f"Epoche {epoch} {mode}: (loss {loss:.4f}, F1 {f1:.4f})")

    if writer:
        writer.add_scalar(f"{mode}/Loss", loss, epoch)
        writer.add_scalar(f"{mode}/F1-Score", f1, epoch)


def main(args):
    NUM_INS = 7
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## make output directory
    date_ = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(args.output_folder, f"{date_}_{args.trial_name}")
    os.makedirs(output_path)

    ## logging in text_file
    log_file = open(os.path.join(output_path, "log.txt"), "a")

    def log(msg):
        print(time.strftime("[%d.%m.%Y %H:%M:%S]: "), msg, file=stderr)
        log_file.write(time.strftime("[%d.%m.%Y %H:%M:%S]: ") + msg + os.linesep)
        log_file.flush()
        os.fsync(log_file)

    log('Output directory: ' + output_path)

    log("Used parameters...")
    for arg in sorted(vars(args)):
        log("\t" + str(arg) + " : " + str(getattr(args, arg)))

    args_dict = {}
    for arg in vars(args):
        args_dict[str(arg)] = getattr(args, arg)

    ## tensorboard writer 
    writer = SummaryWriter(log_dir=output_path)
    writer.add_text("Args", str(args_dict), global_step=0)

    ## Model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=NUM_INS, bias=True)
    model.to(device)

    ## Loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    ## optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    ## seeding
    SEED = 42
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    best_f1 = 0
    best_loss = 1

    '''
    TODO:
    1. Anti-cl
    2. best model selection: instead of f1 score, choose loss?
    '''

    prepare_dataset(args)
    stage = 0
    train_loader, val_loader = prepare_dataloader(args, stage)
    #train_loader, val_loader = prepare_dataloader_anticl(args, stage)
    log("----Stage 0----")

    for epoch in range(1, args.epochs + 1):

        ### Switch DataLoader according to stage
        if epoch == 21:
            del train_loader, val_loader
            stage = 1
            train_loader, val_loader = prepare_dataloader(args, stage)
            # train_loader, val_loader = prepare_dataloader_anticl(args, stage)
            log("----Stage 1----")
        elif epoch == 41:
            del train_loader, val_loader
            stage = 2
            train_loader, val_loader = prepare_dataloader(args, stage)
            # train_loader, val_loader = prepare_dataloader_anticl(args, stage)
            log("----Stage 2----")
        elif epoch == 61:
            del train_loader, val_loader
            stage = 3
            train_loader, val_loader = prepare_dataloader(args, stage)
            # train_loader, val_loader = prepare_dataloader_anticl(args, stage)
            log("----Stage 3----")
        elif epoch == 81:
            del train_loader, val_loader
            stage = 4
            train_loader, val_loader = prepare_dataloader(args, stage)
            # train_loader, val_loader = prepare_dataloader_anticl(args, stage)
            log("----Stage 4----")
        # elif epoch == 91:
        #     del train_loader, val_loader
        #     stage = 5
        #     train_loader, val_loader = prepare_dataloader(args, stage)
        #     # train_loader, val_loader = prepare_dataloader_anticl(args, stage)
        #     log("----Stage 5----")

        ## training
        f1_train, loss_train, model, optimizer = forward_data(train_loader, "train", model, optimizer, criterion, args,
                                                              device)
        logging("train", writer, log, epoch, f1_train, loss_train)

        with torch.no_grad():
            ##Validation
            f1_val, loss_val, model, optimizer = forward_data(val_loader, "val", model, optimizer, criterion, args,
                                                              device)
            logging("val", writer, log, epoch, f1_val, loss_val)

        ## save Checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            current_state = {'epoch': epoch,
                             'model_weights': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'args': args_dict,
                             }

            model_path = os.path.join(output_path, f"checkpoint.pth.tar")
            torch.save(current_state, model_path)
            log(f"Saved checkpoint to: {model_path}")

        ### save best model on validation set, f1 score
        if stage >= 3 and f1_val > best_f1:
            best_f1 = f1_val

            current_state = {'epoch': epoch,
                             'model_weights': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'args': args_dict,
                             'f1_val': f1_val,
                             }
            model_path = os.path.join(output_path, f"model_best_f1.pth.tar")
            torch.save(current_state, model_path)
            log(f"Saved Model with best Validation F1 to: {model_path}" )

        ### save best model on validation set, loss value
        if stage >= 3 and loss_val < best_loss:
            best_loss = loss_val

            current_state = {'epoch': epoch,
                             'model_weights': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'args': args_dict,
                             'f1_val': f1_val,
                             }
            model_path = os.path.join(output_path, f"model_best_loss.pth.tar")
            torch.save(current_state, model_path)
            log(f"Saved Model with best Validation loss value to: {model_path}" )


    prepare_dataset(args)
    ## test best model, f1 score:
    log("Testing best Validation Model with f1 score")
    test_loader = prepare_dataloader(args, "test")
    # test_loader = prepare_dataloader_anticl(args, 4)
    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join("/mnt/ceph/tco/TCO-Students/Projects/KP_curriculum_learning/cholec80/results/Approach6/Stage_6_Epoch_140_saveInWholeStage/", f"model_best_f1.pth.tar"))['model_weights'])
        f1_test, loss_test, model, optimizer = forward_data(test_loader, "test", model, optimizer, criterion, args,
                                                            device)
        log(f"Test: (loss {loss_test:.4f}, F1 {np.asarray(f1_test)})")

    # ## test best model, loss value:
    # log("Testing best Validation Model with loss value")
    # with torch.no_grad():
    #     model.load_state_dict(torch.load(os.path.join(output_path, f"model_best_loss.pth.tar"))['model_weights'])
    #     f1_test, loss_test, model, optimizer = forward_data(test_loader, "test", model, optimizer, criterion, args,
    #                                                         device)
    #     log(f"Test: (loss {loss_test:.4f}, F1 {np.asarray(f1_test)})")

    log_file.close()
    writer.flush()
    writer.close()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
