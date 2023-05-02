# gpu_info = !nvidia-smi -i 0
# gpu_info = '\n'.join(gpu_info)
# print(gpu_info)

#!pip install info-nce-pytorch
from info_nce import InfoNCE


from datetime import datetime
from functools import partial
from PIL import Image,ImageEnhance
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset
import cv2
# from copy_paste import CopyPaste
# from coco import CocoDetectionCP
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')

# parser.add_argument('-a', '--arch', default='resnet18')
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)


# lr: 0.06 for batch 512 (or 0.03 for batch 256)
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

# moco specific configs:
parser.add_argument(
    "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--moco-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)

parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')

# knn monitor
parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

# utils
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')

parser.add_argument('--background_dir', default='', type=str, metavar='PATH', help='path to background images (default: none)')
parser.add_argument('--root_dir', default='', type=str, metavar='PATH', help='path to root of foreground images (default: none)')

'''
args = parser.parse_args()  # running in command line
'''
args = parser.parse_args('')  # running in ipynb

# set command line arguments here when running in ipynb
args.epochs = 200
args.cos = True
args.schedule = []  # cos in use
args.symmetric = False
if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")

args.background_dir = '/home1/rshashan/imagenette2-320/train/n01440764/'
args.root_dir = '/home1/rshashan/coco-segmentations/'


print(args)

'''
Assumed dir structure:
root dir
|
-this notebook
|
-tiny-imagenet-200-20230328T011153Z-001
    |
    - tiny-imagenet-200
        |
        - images
            |
            - xxx.jpeg
            
'''
'''
Assumed dir structure:
root dir
|
-this notebook
|
-tiny-imagenet-200-20230328T011153Z-001
    |
    - tiny-imagenet-200
        |
        - images
            |
            - xxx.jpeg
            
'''
def get_background():
  ''' Get random imagenet (224, 224) image '''
  tiny_imagenet = random.choice(os.listdir(args.background_dir))
  image = Image.open(args.background_dir + tiny_imagenet)

  return image.crop((0,0,224,224))
def copypaste(fg, seed=[0,0,0], im=0, loc=[0,0]):
  ''' Pastes the foreground into a random background '''

  bg_size = (224, 224)
  '''Apply random scaling to foreground'''
  scale = (seed[2] - 0.5) + 1
  new_fg_dim = int(128 * scale)
  fg_size = (128, 128)

  image = np.array(fg)
  if im == 0:
    bg_img = get_background()
  else:
    bg_img = Image.new(mode="RGB", size=bg_size, color="white")

  # if input_box is not None:
  #   dst = segment(image, input_box)
  # else:
  dst = image

  im_pil = Image.fromarray(dst)
  image = im_pil.resize(fg_size)
  bg_img = bg_img.resize(bg_size)

  max_loc = bg_size[0] - fg_size[0]
  x = loc[0]
  y = loc[1]

  x1, y1, x2, y2 = x, y, x+fg_size[0], y+fg_size[0]
  '''Horizontal flip'''
  if seed[0] >= 0.5:
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
  '''Vertical flip'''
  if seed[1] >= 0.5:
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
  
  # image = image.rotate(360 * seed[2], Image.NEAREST, expand = 1)
  
  # display(image)
  bg_img.paste(image, (x1, y1, x2, y2), image.convert('RGBA'))
  # bg_img = bg_img.resize((32,32))
  return np.array(bg_img)
    
    
# cifar = '0001.png'
# cifar = Image.open(cifar)

# # cifar = np.asarray(cifar)
# augmented = copypaste(cifar)
# augmented.show()

class MyLambda(transforms.Lambda):
    def __init__(self, lambd, seed, img, loc):
        super().__init__(lambd)
        self.seed = seed
        self.img = img
        self.loc=  loc
    def __call__(self, img):
        return self.lambd(img, self.seed, self.img, self.loc)
class COCOPairDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        transform=None,
        train: bool = True,     # optional if we decide to separate dataset into train and test
    ):
        self.data1 = []
        self.data2 = []
        self.labels = []
        self.transform = transform
        classes = [c for c in os.listdir(root_dir) if os.path.isdir(root_dir + c)]
        classes = {cls: i for i, cls in enumerate(classes)}
        print(classes)
        for class_dir in os.listdir(root_dir):
          if os.path.isdir(root_dir + class_dir):
            images = [f for f in os.listdir(root_dir + class_dir) if f.endswith(('.jpeg', '.png'))]
            
            #  skip this class if it has less than 20 images
            # if len(images) < 20:
            #   continue

            # select 20 images per class
            for _ in range(1):
              for img in os.listdir(root_dir + class_dir):
                  if img.endswith(('.jpeg', '.png')):
                    image = Image.open(f'{root_dir}{class_dir}/{img}')
                    data = np.asarray(image)

                    if self.transform is not None:
                        seed = [random.random(),random.random(),random.random()]
                        self.transform.transforms[-1].seed = seed
                        x = random.randint(0, 224-128)
                        y = random.randint(0, 224-128)
                        self.transform.transforms[-1].loc = [x,y]
                        self.transform.transforms[-1].img = 0
                        im_1 = self.transform(image)
                        self.transform.transforms[-1].img = 1
                        im_2 = self.transform(image)
                        im_1 = np.asarray(im_1)
                        im_2 = np.asarray(im_2)
                        if len(im_1.shape) < 3 or len(im_2.shape) < 3:
                            continue
                        #print(im_1.shape)
                        #print(img)
                        im_1 = np.transpose(im_1, axes=[2,0,1])
                        im_2 = np.transpose(im_2, axes=[2,0,1])
                        
                        self.data1.append(im_1)
                        self.data2.append(im_2)

                    self.labels.append(classes[class_dir])

    def __getitem__(self, index):
        label = self.labels[index]
        img1= self.data1[index]
        img2 = self.data2[index]

        return img1, img2, label

    def __len__(self):
      return len(self.data1)




transform = transforms.Compose([
  # transforms.ToPILImage(),
  MyLambda(copypaste, [0,0,0], 0, [0,0])
  ])

# data prepare
# transform=train_transform
train_data = COCOPairDataset(root_dir=args.root_dir, train=True, transform = transform)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
# memory_data = COCOPairDataset(root_dir=coco_dir, train=True, transform=transform)
# memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)

class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, feature_dim=128, arch=None, bn_splits=16):
        super(ModelBase, self).__init__()

        # use split batchnorm
        bn_splits = 1
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        # for name, module in net.named_children():
        #     if name == 'conv1':
                
        #         module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #         '''If input image is 64x64'''
        #         # module = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        #     if isinstance(module, nn.MaxPool2d):
        #         continue
        #     if isinstance(module, nn.Linear):
        #         self.net.append(nn.Flatten(1))
        #     self.net.append(module)

        # self.net = nn.Sequential(*self.net)
        self.net = net

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x

class ModelMoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, T=0.1, arch='resnet50', bn_splits=8, symmetric=True):
        super(ModelMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric
        base_encoder = models.__dict__[args.arch]
        # create the encoders
        self.encoder_q = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)
        self.encoder_k = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)
        # self.encoder_q = base_encoder(num_classes=dim)
        # self.encoder_k = base_encoder(num_classes=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # loss = nn.CrossEntropyLoss().cuda()(logits, labels)
        #print(labels.shape, logits.shape)
        #print(logits.unsqueeze(1).shape, labels.unsqueeze(1).shape)
        loss = InfoNCE().cuda()(q, k)


        return loss, q, k

    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        return loss

# create model
model = ModelMoCo(
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        arch=args.arch,
        bn_splits=args.bn_splits,
        symmetric=args.symmetric,
    ).cuda()
print(model.encoder_q)


# train for one epoch
def train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_2,_ in train_bar:
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)
        im_1 = im_1.float()
        im_2 = im_2.float()
        loss = net(im_1, im_2)
        
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num

# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(net, memory_data_loader, test_data_loader, epoch, args):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

# define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

# load model if resume


#model.load_state_dict(torch.load('/home1/rshashan/cache-2023-03-31-08-14-23-moco/model_last.pth')['state_dict'])

# logging
results = {'train_loss': [], 'test_acc@1': []}
if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)
# dump args
with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)

# training loop
for epoch in range(0, args.epochs):
    train_loss = train(model, train_loader, optimizer, epoch, args)
    results['train_loss'].append(train_loss)
    # test_acc_1 = test(model.encoder_q, memory_loader, test_loader, epoch, args)
    # results['test_acc@1'].append(test_acc_1)
    # save statistics
    data_frame = pd.DataFrame(data=results['train_loss'], index=range(0, epoch + 1))
    data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
    # save model
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')

# model.load_state_dict(torch.load('/home1/rshashan/cache-2023-04-19-07-55-09-moco/model_last.pth')['state_dict'])
# print(torch.load('/home1/rshashan/cache-2023-04-01-21-26-17-moco/model_last.pth')['epoch'])

num_classes = 10
batch_size = 64
test_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(train_data)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

cp_train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
cp_test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=test_sampler)

#cache-2023-04-30-15-53-25-moco
checkpoint = torch.load('/home1/rshashan/cache-2023-04-30-15-53-25-moco/model_last.pth', map_location="cpu")
#checkpoint = torch.load('/home1/rshashan/cache-2023-04-27-15-55-28-moco/model_last.pth', map_location="cpu")
model.load_state_dict(checkpoint['state_dict'])


model.eval()
# phead.eval()
# phead.cuda()
outputs = None
labels = None
with torch.no_grad():
    for images,_,label in tqdm(train_loader):
        #images, label = data
        label = label.cuda()
        output = model.encoder_q(images.cuda().float())
        if outputs is None:
            outputs = output.cpu().numpy()
            labels = label.cpu().numpy()
        else:
#             print(outputs.shape)
            outputs = np.concatenate((outputs, output.cpu().numpy()), axis=0)
            labels = np.concatenate((labels, label.cpu().numpy()), axis=0)
import pickle as pk
with open('labels.pkl', 'wb') as f:
    pk.dump(labels, f)
with open('embeddings.pkl', 'wb') as f:
    pk.dump(outputs ,f)










#lin = model.encoder_q.net.fc
#new_lin = nn.Sequential(
#    lin,
#    nn.Linear(lin.out_features, 128),
#    nn.BatchNorm1d(128),
#    nn.ReLU(),
#    nn.Linear(128, num_classes)
#)
#model.encoder_q.net.fc = new_lin

#for name, param in model.encoder_q.named_parameters():
#    if name not in ["net.fc.0.weight","net.fc.1.weight","net.fc.3.weight", "net.fc.0.bias","net.fc.1.bias","net.fc.3.bias"]:
#        pi1 = param.requires_grad = False

for name, param in model.encoder_q.named_parameters():
        if name not in ["net.fc.weight", "net.fc.bias"]:
            param.requires_grad = False
# init the fc layer
model.encoder_q.net.fc.weight.data.normal_(mean=0.0, std=0.01)
model.encoder_q.net.fc.bias.data.zero_()

print('model')
model.encoder_q.cuda()
print('model end')
'''
Training projection head
'''
num_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1.0, momentum=0.9)
sch = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=25)

#model.encoder_q.train()

total_loss = 0
for ep in range(num_epochs):

    for data,_, label in tqdm(cp_train_loader):
        #model.zero_grad()
        optimizer.zero_grad()
        data = data.cuda().float()
        label = label.cuda()
        # emb = model.encoder_q(data)
        output = model.encoder_q(data)
#         print(output)
#         print(label)
        loss = criterion(output, label)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        sch.step()
    total_loss /= len(cp_train_loader)
    print('Epoch: ' + str(ep) + ' Loss: ' + str(total_loss.item()) )
    torch.save(model.state_dict(), '/home1/rshashan/phead-new.pth')
classes = ('boat','dog','zebra','train','cat','sheep','pizza','person3','cow','elephant')

correct = 0
total = 0
model.encoder_q.eval()
# model.eval()
# phead.cuda()
with torch.no_grad():
    for images,_,labels in tqdm(cp_test_loader):
        images = images.cuda().float()
        labels = labels.cuda()
        outputs = model.encoder_q(images)
        # outputs = phead(outputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for images,_ ,labels in cp_test_loader:
        images = images.cuda().float()
        labels = labels.cuda()
        outputs = model.encoder_q(images)
        # outputs = phead(outputs)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(c)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
