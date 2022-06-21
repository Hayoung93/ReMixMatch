import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from copy import copy
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from torch_ema import ExponentialMovingAverage
from randaugment import RandAugment
from ctaugment import CTAugment
from models.wideresnet import WideResNet
from utils.misc import load_full_checkpoint, SharpenSoftmax, Div255, LogWeight


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="stl10")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model_arch", default="wideresnet")
    parser.add_argument("--model_name", default="wideresnet-28-2")
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--results_dir", type=str, default="/data/weights/hayoung/mixmatch/t1")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--datadir", type=str, default="/data/data/stl10")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--tensorboard_path", type=str, default="./runs/mixmatch/t1")
    parser.add_argument("--tsa", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--weight", type=str, default="/data/weights/hayoung/mixmatch/t1/model_last.pth")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--aug_type", help="randaug or ctaug", default="ctaug")
    parser.add_argument("--k", type=int, help="number of strong augmentation for unlabeled data", default=8)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--unsup_weight", type=float, default=1.5)
    parser.add_argument("--use_disk", action="store_true")
    parser.add_argument("--temp_disk", type=str, default="./temp")
    parser.add_argument("--cosine_tmax", type=int, default=300)
    args = parser.parse_args()

    return args


def get_loaders(args, resolution, train_transform=None, test_transform=None):
    # transforms
    train_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(resolution)
    ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if args.dataset_name == "stl10":
        valset = datasets.STL10(args.datadir, "test", transform=test_transform, download=True)
        trainset = datasets.STL10(args.datadir, "train", transform=train_transform, download=True)
        unlabelset = datasets.STL10(args.datadir, "unlabeled", transform=train_transform, download=True)
    else:
        raise Exception("Not supported dataset")
    # loader
    trainloader = DataLoader(trainset, 1, True, num_workers=0, pin_memory=True)
    valloader = DataLoader(valset, args.batch_size, False, num_workers=0, pin_memory=True)
    trainloader_u = DataLoader(unlabelset, args.batch_size, True, num_workers=0, pin_memory=True)

    return trainloader, valloader, trainloader_u


def get_model(args, device):
    if args.model_arch == "efficientnet":
        model = EfficientNet.from_name(args.model_name)
        model._fc = nn.Linear(model._fc.in_features, args.num_classes)
    elif args.model_arch == "wideresnet":
        depth = int(args.model_name.split("-")[1])
        w_factor = int(args.model_name.split("-")[2])
        model = WideResNet(depth, args.num_classes, w_factor)
    else:
        raise Exception("Not supported network architecture")

    # if args.ema:
    #     ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    #     ema = ema.to(device)
    # else:
    #     ema = None

    model.to(device)
    return model

def main(args):
    writer = SummaryWriter(args.tensorboard_path)
    if not os.path.exists(args.tensorboard_path):
        os.mkdir(args.tensorboard_path)
    
    resolution = (args.resolution, args.resolution)
    weak_transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.Pad(12),
        transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip(),
        Div255(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if args.aug_type == "randaug":
        strong_transform = RandAugment(1, 2)
    elif args.aug_type == "ctaug":
        strong_transform = CTAugment()
    else:
        raise Exception("Augmentation type not supported")
    trainloader, valloader, trainloader_u = get_loaders(args, resolution)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, device)
    
    supcriterion = nn.BCELoss(reduction='mean')
    unsupcriterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    if args.resume:
        model, optimizer, scheduler, last_epoch, best_val_loss, best_val_acc, p_label, p_pred = \
            load_full_checkpoint(model, optimizer, scheduler, args.weight)
        print("Loaded checkpoint from: {}".format(args.weight))
        start_epoch = last_epoch + 1
    else:
        start_epoch = 0
        best_val_acc = 0.
    
    p_label, p_pred = torch.zeros(args.batch_size, args.num_classes).cuda(), torch.zeros(args.batch_size*(args.k+2), args.num_classes).cuda()
    for ep in range(start_epoch, args.num_epochs):
        scheduler.step()
        # val_loss, val_acc = eval_model(ep, model, valloader, supcriterion, writer, args.num_epochs)
        print("Epoch {} --------------------------------------------".format(ep + 1))
        train_loss, train_acc, model, optimizer, p_label, p_pred = \
            train(args, ep, model, trainloader, trainloader_u, weak_transform, strong_transform,
                  supcriterion, unsupcriterion, optimizer, writer, args.num_epochs, p_label, p_pred)
        print("Train loss: {}\tTrain acc: {}".format(train_loss, train_acc))
        val_loss, val_acc = eval_model(ep, model, valloader, supcriterion, writer, args.num_epochs)
        print("Val loss: {}\tVal acc: {}".format(val_loss, val_acc))
        print("--------------------------------------------")
        # scheduler.step(val_loss)
        print("{}".format(optimizer.state_dict))
        if ep == 0:
            best_val_loss = val_loss
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                save_checkpoint(ep, model, optimizer, scheduler, args.results_dir, best_val_loss, best_val_acc, True)
        save_checkpoint(ep, model, optimizer, scheduler, args.results_dir, best_val_loss, best_val_acc, False)
    print("Best Val Loss: {} / Acc: {}".format(best_val_loss, best_val_acc))


def train(args, ep, model, loader, loader_u, weak_transform, strong_transform, sup_criterion, unsup_criterion, optimizer, writer, eps,
            p_label, p_pred):
    model.train()
    train_loss = 0.
    train_acc = 0.
    softmax = nn.Softmax(dim=1)
    sharpen_softmax = SharpenSoftmax(0.5, dim=1)
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    beta = torch.distributions.beta.Beta(args.alpha, args.alpha)
    labeled_generator = iter(loader)
    unsup_weight_coeff = LogWeight(10, args.num_epochs)
    w = [1, 1.5, 0.5, 0.5]  # sup, unsup, premix, rotation

    pbar = tqdm(loader_u)
    for i, (inputs_u, _) in enumerate(pbar):
        optimizer.zero_grad()
        # construct labeled batch
        labeled_inputs = []
        labeled_labels = []
        for _ in range(args.batch_size):
            try:
                _inputs, _labels = labeled_generator.next()
                labeled_inputs.append(_inputs)
                labeled_labels.append(_labels)
            except StopIteration:
                labeled_generator = iter(loader)
                _inputs, _labels = labeled_generator.next()
                labeled_inputs.append(_inputs)
                labeled_labels.append(_labels)
        labeled_inputs = torch.cat(labeled_inputs, dim=0).cuda()
        labeled_labels = torch.cat(labeled_labels, dim=0).cuda()
        labeled_inputs, aug_index, bin_index = strong_transform.aug_batch(labeled_inputs)
        labeled_inputs = normalize(labeled_inputs / 255.)

        # construct unlabeled batch
        unlabeled_inputs = []
        guessed_labels = []
        with torch.no_grad():
            inputs_u = inputs_u.cuda()
            # weakly augmented samples
            inputs_u_weak = weak_transform(inputs_u)
            # strongly augmented samples
            for _ in range(args.k):
                inputs_u_strong, _, _ = strong_transform.aug_batch(inputs_u)
                unlabeled_inputs.append(normalize(inputs_u_strong / 255.))
            # label guess
            pred = model(inputs_u_weak)
            pred = pred * (p_label.sum(0) + 1e-6).cuda() / (p_pred.sum(0) + 1e-6).cuda()
            pred = pred / pred.sum()
            pred = pred ** 2
            pred = pred / pred.sum(1)[:, None]
            guessed_labels = guessed_labels + [pred] * (args.k + 1)
            unlabeled_inputs.append(inputs_u_weak)

        # construct W
        w_inputs = torch.cat([labeled_inputs] + unlabeled_inputs, dim=0)
        w_labels = torch.cat([nn.functional.one_hot(labeled_labels, args.num_classes)] + guessed_labels, dim=0)
        w_index = list(range(len(w_inputs)))
        np.random.shuffle(w_index)

        # mixup
        lam = beta.sample(sample_shape=torch.Size([len(w_index)])).cuda()
        lam_p = torch.max(lam, 1-lam)
        inputs = w_inputs * lam_p[:, None, None, None] + w_inputs[w_index] * (1-lam_p)[:, None, None, None]
        labels = w_labels * lam_p[:, None] + w_labels[w_index] * (1-lam_p)[:, None]

        in_index = list(range(len(inputs)))
        np.random.shuffle(in_index)
        in_index = torch.tensor(in_index)
        labeled_mask = in_index < args.batch_size

        # forward
        outputs = model(inputs[in_index]).softmax(1)
        strong_transform.update(outputs[labeled_mask], labels[labeled_mask], aug_index, bin_index)
        p_label = p_label * 0.999 + nn.functional.one_hot(labeled_labels, args.num_classes) * 0.001
        p_pred = p_pred * 0.999 + outputs * 0.001  # !! TODO: 128 moving average
        
        # rotation forward
        rot_inputs = []
        angles = [0, 90, 180, 270]
        angle = np.random.choice(angles, len(unlabeled_inputs[0]), replace=True)
        angle = angle.tolist()
        for ui in unlabeled_inputs[0]:
            rot_inputs.append(transforms.functional.rotate(ui, angle[0]))
        rot_inputs = torch.stack(rot_inputs, dim=0)
        rot_outputs = model(rot_inputs, rotation=True).softmax(1)

        # backward
        sup_loss = sup_criterion(outputs[labeled_mask], labels[labeled_mask])
        unsup_loss = unsup_criterion(outputs[~labeled_mask], labels[~labeled_mask])
        premix_loss = unsup_criterion(model(unlabeled_inputs[0]).softmax(1), guessed_labels[0])
        rotation_loss = unsup_criterion(rot_outputs, nn.functional.one_hot(torch.tensor([angles.index(ang) for ang in angle]), 4).float().cuda())
        total_loss = w[0] * sup_loss + w[1] * unsup_weight_coeff(ep) * unsup_loss + w[2] * premix_loss + w[3] * rotation_loss
        total_loss.backward()
        optimizer.step()

        # acc (labeled data)
        running_acc = (outputs[labeled_mask].argmax(1) == labels[labeled_mask].argmax(1)).sum().item()
        train_acc += running_acc
        writer.add_scalar("train acc", running_acc / labeled_mask.sum().item(), ep * len(loader_u) * i)

        pbar.set_description("Epoch: {}/{}|Loss: {}".format(ep, eps-1, total_loss))

    train_loss /= len(loader)
    train_acc /= len(loader.dataset)
    return train_loss, train_acc, model, optimizer, p_label, p_pred


def eval_model(ep, model, loader, criterion, writer, eps):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            running_acc = (outputs.argmax(1) == labels).sum().item()
            val_acc += running_acc
            loss = criterion(outputs.softmax(1), torch.nn.functional.one_hot(labels, 10).float())
            val_loss += loss.item()
            writer.add_scalar("val loss", loss.item(), ep * len(loader) + i)
            writer.add_scalar("val acc", running_acc, ep * len(loader) + i)
    val_loss /= len(loader)
    val_acc /= len(loader.dataset)
    return val_loss, val_acc


def save_checkpoint(ep, model, optimizer, scheduler, savepath, best_val_loss, best_val_acc, isbest, p_label, p_pred):
    save_dict = {
        "epoch": ep,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "p_label": p_label,
        "p_pred": p_pred
    }
    if isbest:
        torch.save(save_dict, os.path.join(savepath, "model_best.pth"))
    else:
        torch.save(save_dict, os.path.join(savepath, "model_last.pth"))


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    main(args)