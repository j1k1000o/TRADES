from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models.magnet_resnet import ResNet18
from trades import trades_loss
from pgd_attack_generic import eval_adv_test_whitebox

def print_to_log(text, txt_file_path):
    with open(txt_file_path, 'a') as text_file:
        print(text, file=text_file)

def update_log(optimizer, epoch, train_loss, train_acc, test_loss, test_acc,
        pgd_acc, log_path):
    lr = get_lr(optimizer)
    print_to_log(
        f'{epoch+1}\t {lr:1.0E}\t {train_loss:5.4f} \t '
        f'{train_acc:4.3f}\t\t {test_loss:5.4f}\t\t {test_acc:4.3f}\t\t '
        f'{pgd_acc:4.3f}',
        log_path
    )

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

parser = argparse.ArgumentParser(description='PyTorch SVHN TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=76, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', required=True,
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset to train in')
parser.add_argument('--pretrained-path', default=None, type=str,
                    help='path to pretrained weights')

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Logging stuff
log_path = os.path.join(model_dir, 'log.txt')
log_headers = ['Epoch', 'LR', 'Train loss', 'Train acc.', 'Test loss', \
        'Test acc.', 'PGD acc.']
print_to_log('\t '.join(log_headers), log_path)

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == 'svhn':
    num_classes = 10
    trainset = torchvision.datasets.SVHN(root='../data', split='train', 
        download=True, transform=transform_train)
    testset = torchvision.datasets.SVHN(root='../data', split='test', 
        download=True, transform=transform_test)
elif args.dataset == 'cifar10':
    num_classes = 10
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, 
        download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, 
        download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    num_classes = 100
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, 
        download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, 
        download=True, transform=transform_test)
else:
    print(f'Dataset "{args.dataset}" not implemented. Exiting...')
    sys.exit()

train_loader = torch.utils.data.DataLoader(trainset, 
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, 
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class MagnetModelWrapper(nn.Module):
    def __init__(self, model):
        super(MagnetModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        scores, _ = self.model(x)
        return scores


def main():
    model = ResNet18(num_classes=num_classes)
    if args.pretrained_path is not None:
        print(f'Load pretrained weights from {args.pretrained_path}', end='...')
        ckpt = torch.load(args.pretrained_path)
        model.load_state_dict(ckpt['state_dict'])
        print('done.')

    model = MagnetModelWrapper(model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
        momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        train_loss, train_acc = eval_train(model, device, train_loader)
        test_loss, test_acc = eval_test(model, device, test_loader)
        print(f'Train acc: {100.*train_acc:4.3f} | Test acc: {100.*test_acc:4.3f}')
        print('================================================================')
        # estimate PGD accuracy
        clean_acc, pgd_acc = eval_adv_test_whitebox(model, device, test_loader, 
            epsilon=8/255, num_steps=5, step_size=2/255)
        print(f'Clean acc: {clean_acc:4.3f} | (est.) PGD acc: {pgd_acc:4.3f}')
        update_log(optimizer, epoch, train_loss, 100.*train_acc, test_loss, 100.*test_acc,
            pgd_acc, log_path)

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, f'model-wideres-epoch{epoch}.pt'))


if __name__ == '__main__':
    main()
