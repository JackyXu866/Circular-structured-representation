# This code is written by Jingyuan Yang @ XD

"""Train Emotion_LDL with Pytorch"""

import torch
import argparse
import torchvision
import torchvision.transforms as transforms
import os
import random
# from data_LDL import Emotion_LDL
from dataloader_customize import EmoData
# from models import *
from model_baseline import model_baseline
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from adamw import AdamW
# from torch.autograd import Variable
import utils
import math
import numpy as np
from torchvision import models
from tensorboardX import SummaryWriter
# from models.TL import Triplet
# from models.CE_loss_softmax import CELoss_softmax
from MSE_loss_theta import MSE_Loss_theta
from Polarloss import PolarLoss
# from models.CE_loss_weighed import CELoss_weighed
from polar_coordinates import Polar_coordinates
from evaluation_metric import Evaluation_metrics
from PIL import ImageFile
from da_model import CNNModel

ImageFile.LOAD_TRUNCATED_IMAGES = True
cuda = True

# random seed
def set_seed(seed):
    # random.seed(seed) ##
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.benchmark = False                   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True            # cudnn
    # os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    # Parameters
    parser = argparse.ArgumentParser(description='PyTorch Emotion_LDL CNN Training')
    parser.add_argument('--img_path', type=str, default='/home/yjy/Dataset/Emotion_LDL/Twitter_LDL/images/')
    parser.add_argument('--source_train_csv_file', type=str,
                        default='./dataset/fer2013/train/train.csv')
    parser.add_argument('--target_train_csv_file', type=str,
                        default='./dataset/facialRecon/train/train.csv')
    # FLICKR csv_7 TWITTER csv_6
    parser.add_argument('--source_test_csv_file', type=str,
                        default='./dataset/fer2013/test/test.csv')
    parser.add_argument('--target_test_csv_file', type=str,
                        default='./dataset/facialRecon/test/test.csv')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt_da')
    parser.add_argument('--model', type=str, default='ResNet_50', help='CNN architecture')
    parser.add_argument('--dataset', type=str, default='Emotion_LDL', help='Dataset')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkposint')

    parser.add_argument('--if_decay', default=1, type=int, help='decay lr every 5 epochs')
    parser.add_argument('--decay', default=0.1, type=float, help='decay value every 5 epochs')
    parser.add_argument('--start', default=10, type=float, help='decay value every 5 epochs')
    parser.add_argument('--every', default=10, type=float, help='decay value every 5 epochs')
    parser.add_argument('--lr_adam', default=1e-5, type=float, help='learning rate for adam|5e-4|1e-5|smaller')
    parser.add_argument('--lr_sgd', default=1e-3, type=float,  help='learning rate for sgd|1e-3|5e-4')
    parser.add_argument('--wd', default=5e-5, type=float, help='weight decay for adam|1e-4|5e-5')
    parser.add_argument('--optimizer', default='adamw', type=str, help='sgd|adam|adamw')
    parser.add_argument('--gpu', default=0, type=int, help='0|1|2|3')

    parser.add_argument('--seed', default=66, type=int, help='just a random seed')
    opt = parser.parse_args()

    # set gpu device
    torch.cuda.set_device(opt.gpu)

    set_seed(seed=opt.seed)

    writer = SummaryWriter()

    best_test_acc = 0
    best_test_acc_epoch = 0
    best_test_loss = 20
    best_test_loss_epoch = 0
    start_epoch = 0

    learning_rate_decay_start = opt.start
    learning_rate_decay_every = opt.every
    learning_rate_decay_rate = opt.decay

    total_epoch = 25

    path = os.path.join(opt.dataset + '_' + opt.model)

    # Data
    print('==> Preparing data..')

    transform_train_source = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(48),  # resize the short side to 480, and resize the long side proportionally
        transforms.RandomCrop(48),  # different from resize, randomcrop will crop a square of 448*448, disproportionally
        transforms.RandomHorizontalFlip(),
    ])

    transform_test_source = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(48),
            transforms.RandomCrop(48),
        ])
    
    transform_train_target = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(48),  # resize the short side to 480, and resize the long side proportionally
        transforms.RandomCrop(48),  # different from resize, randomcrop will crop a square of 448*448, disproportionally
        transforms.RandomHorizontalFlip(),
    ])

    transform_test_target = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(48),
            transforms.RandomCrop(48),
        ])

    # trainset = Emotion_LDL(csv_file=opt.train_csv_file, root_dir=opt.img_path, transform=transform_train)
    # testset= Emotion_LDL(csv_file=opt.test_csv_file, root_dir=opt.img_path, transform=transform_test)
    source_trainset = EmoData(csv_file=opt.source_train_csv_file, transform=transform_train_source)
    source_testset = EmoData(csv_file=opt.source_test_csv_file, transform=transform_test_source)
    
    target_trainset = EmoData(csv_file=opt.target_train_csv_file, transform=transform_train_target)
    target_testset = EmoData(csv_file=opt.target_test_csv_file, transform=transform_test_target)

    source_trainloader = torch.utils.data.DataLoader(source_trainset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    source_testloader = torch.utils.data.DataLoader(source_testset, batch_size=opt.batch_size, shuffle=False, num_workers=8)
    
    target_trainloader = torch.utils.data.DataLoader(target_trainset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    target_testloader = torch.utils.data.DataLoader(target_testset, batch_size=opt.batch_size, shuffle=False, num_workers=8)

    # Model
    # if opt.model == 'ResNet_50':
    #     base_model = models.resnet50(pretrained=True) ###
    #     net = model_baseline(base_model)
    # elif opt.model == 'ResNet_101':
    #     base_model = models.resnet101(pretrained=True) ###
    #     net = model_baseline(base_model)
    # elif opt.model == 'VGG_19':
    #     base_model = models.vgg19(pretrained=True)
    #     net = model_baseline(base_model)

    net = CNNModel()
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    param_num = 0
    for param in net.parameters():
        param_num = param_num + int(np.prod(param.shape))

    print('==> Trainable params: %.2f million' % (param_num / 1e6))
    #print(np.prod(net.lstm.parameters()[1].shape))

    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('./ckpts1/epoch-19.pkl', map_location="cuda:0")
        net.load_state_dict(checkpoint)
    else:
        print('==> Building model..')
    if torch.cuda.is_available():
        net.cuda()

    CEloss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()
    KLloss = nn.KLDivLoss(size_average=False, reduce=True)
    
    # KLloss = nn.KLDivLoss(reduction='batchmean')
    MSELoss_theta = MSE_Loss_theta()
    Polarloss = PolarLoss()
    # Triploss = Triplet(measure='cosine', max_violation=True) #MARGIN
    # CEloss_weighed = CELoss_weighed()



    if torch.cuda.is_available():
        CEloss = CEloss.cuda()
        MSEloss = MSEloss.cuda()
        KLloss = KLloss.cuda()

    if opt.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr_adam, weight_decay=opt.wd)
    elif opt.optimizer == 'adamw':
        optimizer = AdamW(net.parameters(), lr=opt.lr_adam, weight_decay=opt.wd, amsgrad=False)
    elif opt.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=opt.lr_sgd, momentum=0.9, weight_decay=5e-4)

        # Data
    print('==> Preparing data..')

    for epoch in range(start_epoch, total_epoch):
        print(epoch)
        # set_seed(seed=opt.seed)
        # train(epoch, opt, net, writer, trainloader, optimizer, KLloss, MSEloss, Polar_coordinates, MSELoss_theta, Polarloss)
        train(epoch, opt, net, writer, source_trainloader, target_trainloader, optimizer, loss_class, loss_domain, KLloss, MSEloss, Polar_coordinates, MSELoss_theta, Polarloss, total_epoch= 50)
        test(epoch, net, writer, target_testloader, KLloss, best_test_acc,
                                                  best_test_acc_epoch, path, MSEloss, Polar_coordinates, MSELoss_theta, Polarloss)
        


def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X



# Training
def train(epoch, opt, net, writer, source_trainloader, target_trainloader, optimizer, loss_class, loss_domain, KLloss, MSEloss, Polar_coordinates, MSELoss_theta, Polarloss, total_epoch= 50):
    # set_seed(seed=opt.seed)
    print('\nEpoch: %d' % epoch)
    global train_acc
    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    train_Dist_1 = 0
    train_Dist_2 = 0
    train_Dist_3 = 0
    train_Dist_4 = 0
    train_Sim_1 = 0
    train_Sim_2 = 0
    correct = 0
    total = 0


    len_dataloader = min(len(source_trainloader), len(target_trainloader))
    data_source_iter = iter(source_trainloader)
    data_target_iter = iter(target_trainloader)
    


    i = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / total_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = next(data_source_iter)
        s_img = data_source['image']
        s_label = data_source['dist_emo']

        net.zero_grad()
        batch_size = len(s_label)
        
        # print(s_img.shape[2], s_img.shape[3], s_img.shape[1])

        input_img = torch.FloatTensor(batch_size, s_img.shape[1], s_img.shape[2], s_img.shape[3])
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)

        class_output, domain_output = net(input_data=input_img, alpha=alpha)
        err_s_label = loss_class(class_output, class_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        data_target = data_target_iter.next()
        t_img = data_target["image"]

        batch_size = len(t_img)

        input_img = torch.FloatTensor(batch_size, t_img.shape[1], t_img.shape[2], t_img.shape[3])
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        _, domain_output = net(input_data=input_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        i += 1

        print (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                 err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy())

    torch.save(net, './da_model_epoch_{0}.pth'.format(epoch))


   ################################################################## # 
    # if opt.if_decay == 1:
    #     if epoch >= opt.start:
    #         frac = (epoch - opt.start) // opt.every + 1  # round
    #         decay_factor = opt.decay ** frac  # how many times we have this decay

    #         if opt.optimizer == 'adam':
    #             current_lr = opt.lr_adam
    #         elif opt.optimizer == 'adamw':
    #             current_lr = opt.lr_adam
    #         elif opt.optimizer == 'sgd':
    #             current_lr = opt.lr_sgd
    #         current_lr = current_lr * decay_factor # new learning rate
    #         for rr in range(len(optimizer.param_groups)):
    #             utils.set_lr(optimizer, current_lr, rr)  # set the decayed learning rate
    #     else:
    #         if opt.optimizer == 'adam':
    #             current_lr = opt.lr_adam
    #         elif opt.optimizer == 'adamw':
    #             current_lr = opt.lr_adam
    #         elif opt.optimizer == 'sgd':
    #             current_lr = opt.lr_sgd
    #     print('learning_rate: %s' % str(current_lr))

    # for batch_idx, data in enumerate(trainloader):
    #     images = data['image']
    #     # image_name = data['img_id']
    #     dist_emo = data['dist_emo']

    #     if torch.cuda.is_available():
    #         images = images.cuda()
    #         dist_emo = dist_emo.cuda()

    #     optimizer.zero_grad()
    #     net.train()
    #     emo = net(images)
    #     # print('emo', emo)
    #     # print('dist_emo', dist_emo)
    #     theta_emo, r_emo = Polar_coordinates(emo)
    #     theta_dist_emo, r_dist_emo = Polar_coordinates(dist_emo)
    #     weight = r_dist_emo

    #     loss1 = KLloss(emo.log(), dist_emo)
    #     # loss1 = KLloss(emo, dist_emo)
    #     loss2 = MSELoss_theta(theta_emo, theta_dist_emo, r_dist_emo)
    #     # loss3 = MSEloss(r_emo, r_dist_emo)
    #     loss3 = Polarloss(theta_emo, theta_dist_emo, r_dist_emo)
    #     # loss = loss1 + loss2 ##################
    #     # loss = loss1 + loss3 ##############
    #     loss = loss1 ##############

    #     loss.backward()
    #     # loss.backward(loss.clone().detach())
    #     # for param in net.parameters():
    #     #     print(param.grad)
    #     # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, net.lstm.parameters()), 1.0) #1
    #     optimizer.step()

    #     # print(loss.item())
    #     train_loss1 += loss1.item()
    #     train_loss2 += loss2.item()
    #     train_loss3 += loss3.item()
    #     train_loss += loss.item()

    #     Dis_1, Dis_2, Dis_3, Dis_4, Sim_1, Sim_2 = Evaluation_metrics(emo, dist_emo)
    #     train_Dist_1 += Dis_1
    #     train_Dist_2 += Dis_2
    #     train_Dist_3 += Dis_3
    #     train_Dist_4 += Dis_4
    #     train_Sim_1 += Sim_1
    #     train_Sim_2 += Sim_2

    #     _, predicted = torch.max(emo.data, 1)
    #     _, labeled = torch.max(dist_emo.data, 1)
    #     total += dist_emo.size(0)
    #     correct += predicted.eq(labeled.data).cpu().sum().numpy()
    #     train_acc = 100. * correct / total

    #     utils.progress_bar(batch_idx, len(trainloader),
    #                        'Loss1: %.3f Loss2: %.3f Loss3: %.3f Loss: %.3f '
    #                        '| Chebyshev: %.3f Clark: %.3f Canberra: %.3f KL: %.3f Cosine: %.3f Inter: %.3f Acc: %.3f%%'
    #                        % (train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
    #                           train_loss3 / (batch_idx + 1), train_loss / (batch_idx + 1),
    #                           train_Dist_1 / (batch_idx + 1), train_Dist_2 / (batch_idx + 1),
    #                           train_Dist_3 / (batch_idx + 1), train_Dist_4 / (batch_idx + 1),
    #                           train_Sim_1 / (batch_idx + 1), train_Sim_2 / (batch_idx + 1), train_acc))

    # writer.add_scalar('data/Train_Loss', train_loss, epoch)
    # writer.add_scalar('data/Train_Loss1', train_loss1 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Train_Loss2', train_loss2 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Train_Loss3', train_loss3 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Train_Chebyshev', train_Dist_1 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Train_Clark', train_Dist_2 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Train_Canberra', train_Dist_3 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Train_KL', train_Dist_4 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Train_Cosine', train_Sim_1 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Train_Inter', train_Sim_2 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Train_Acc', train_acc, epoch)
    # print('==> Saving model...')
    # torch.save(net.state_dict(), os.path.join(opt.ckpt_path, 'epoch-%d.pkl' % epoch))

# Test



def test(epoch, net, writer, testloader, KLloss, best_test_acc, best_test_acc_epoch, path, MSEloss, Polar_coordinates, MSELoss_theta, Polarloss):
    # set_seed(seed=opt.seed)
    global test_acc
    # global best_test_acc
    # global best_test_acc_epoch
    # global best_test_loss
    # global best_test_loss_epoch

    test_loss1 = 0
    test_loss2 = 0
    test_loss3 = 0
    test_Dist_1 = 0
    test_Dist_2 = 0
    test_Dist_3 = 0
    test_Dist_4 = 0
    test_Sim_1 = 0
    test_Sim_2 = 0
    test_loss = 0
    correct = 0
    total = 0
    alpha = 0
    
    
    len_dataloader = len(testloader)
    data_target_iter = iter(testloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img = data_target['image']
        t_label = data_target['dist_emo']

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, t_img.shape[1], t_img.shape[2], t_img.shape[3])
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        class_output, _ = net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    print(epoch, "accuracy" , accu)
    

    # for batch_idx, data in enumerate(testloader):
    #     images = data['image']
    #     dist_emo = data['dist_emo']

    #     if torch.cuda.is_available():
    #         images = images.cuda()
    #         dist_emo = dist_emo.cuda()

    #     with torch.no_grad():
    #         net.eval()
    #         emo = net(images)
    #         theta_emo, r_emo = Polar_coordinates(emo)
    #         theta_dist_emo, r_dist_emo = Polar_coordinates(dist_emo)
    #         weight = r_dist_emo

    #         loss1 = KLloss(emo.log(), dist_emo)
    #         # loss1 = KLloss(emo, dist_emo)
    #         loss2 = MSELoss_theta(theta_emo, theta_dist_emo, r_dist_emo)
    #         # loss3 = MSEloss(r_emo, r_dist_emo)
    #         loss3 = Polarloss(theta_emo, theta_dist_emo, r_dist_emo)
    #         # loss = loss1 + loss2 #############
    #         # loss = loss1 + loss3  ##############
    #         loss = loss1 ###########

    #         test_loss1 += loss1.item()
    #         test_loss2 += loss2.item()
    #         test_loss3 += loss3.item()
    #         test_loss += loss.item()

    #         Dis_1, Dis_2, Dis_3, Dis_4, Sim_1, Sim_2 = Evaluation_metrics(emo, dist_emo)
    #         test_Dist_1 += Dis_1
    #         test_Dist_2 += Dis_2
    #         test_Dist_3 += Dis_3
    #         test_Dist_4 += Dis_4
    #         test_Sim_1 += Sim_1
    #         test_Sim_2 += Sim_2

    #         _, predicted = torch.max(emo.data, 1)
    #         _, labeled = torch.max(dist_emo.data, 1)
    #         total += dist_emo.size(0)
    #         correct += predicted.eq(labeled.data).cpu().sum().numpy()
    #         test_acc = 100. * correct / total

    #         utils.progress_bar(batch_idx, len(testloader),
    #                            'Loss1: %.3f Loss2: %.3f Loss3: %.3f Loss: %.3f '
    #                            '| Chebyshev: %.3f Clark: %.3f Canberra: %.3f KL: %.3f Cosine: %.3f Inter: %.3f Acc: %.3f%%'
    #                            % (test_loss1 / (batch_idx + 1), test_loss2 / (batch_idx + 1),
    #                               test_loss3 / (batch_idx + 1), test_loss / (batch_idx + 1),
    #                               test_Dist_1 / (batch_idx + 1), test_Dist_2 / (batch_idx + 1),
    #                               test_Dist_3 / (batch_idx + 1), test_Dist_4 / (batch_idx + 1),
    #                               test_Sim_1 / (batch_idx + 1), test_Sim_2 / (batch_idx + 1), test_acc))

    # writer.add_scalar('data/Test_Loss', test_loss / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Test_Loss1', test_loss1 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Test_Loss2', test_loss2 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Test_Loss3', test_loss3 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Test_Chebyshev', test_Dist_1 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Test_Clark', test_Dist_2 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Test_Canberra', test_Dist_3 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Test_KL', test_Dist_4 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Test_Cosine', test_Sim_1 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Test_Inter', test_Sim_2 / (batch_idx + 1), epoch)
    # writer.add_scalar('data/Test_Acc', test_acc, epoch)

    # # Save checkpoint.
    # if test_acc > best_test_acc:
    # # if (test_loss / (batch_idx + 1)) < best_test_loss:
    #     print('==> Finding best acc..')
    #     # print("best_test_acc: %0.3f" % test_acc)
    #     state = {
    #         'net': net.state_dict() if torch.cuda.is_available() else net,
    #         'acc': test_acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir(path):
    #         os.mkdir(path)
    #     torch.save(state, os.path.join(path,'test_model.t7'))
    #     best_test_acc = test_acc
    #     best_test_acc_epoch = epoch
    # return best_test_acc, best_test_acc_epoch

if __name__ == '__main__':
    main()
    print('Finish training')