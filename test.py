from dataloader_customize import EmoData
from model_baseline import model_baseline
import torch
from torchvision import models
import torchvision.transforms as transforms
from polar_coordinates import Polar_coordinates
from evaluation_metric import Evaluation_metrics
from MSE_loss_theta import MSE_Loss_theta
from Polarloss import PolarLoss
import torch.nn as nn
import utils



# Test
def test(net, testloader, KLloss,  MSEloss, Polar_coordinates, MSELoss_theta, Polarloss):
    # set_seed(seed=opt.seed)

    acc = 0
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

    for batch_idx, data in enumerate(testloader):
        images = data['image']
        dist_emo = data['dist_emo']

        if torch.cuda.is_available():
            images = images.cuda()
            dist_emo = dist_emo.cuda()

        with torch.no_grad():
            net.eval()
            emo = net(images)
            theta_emo, r_emo = Polar_coordinates(emo)
            theta_dist_emo, r_dist_emo = Polar_coordinates(dist_emo)
            weight = r_dist_emo

            loss1 = KLloss(emo.log(), dist_emo)
            # loss1 = KLloss(emo, dist_emo)
            loss2 = MSELoss_theta(theta_emo, theta_dist_emo, r_dist_emo)
            # loss3 = MSEloss(r_emo, r_dist_emo)
            loss3 = Polarloss(theta_emo, theta_dist_emo, r_dist_emo)
            # loss = loss1 + loss2 #############
            # loss = loss1 + loss3  ##############
            # loss = loss2 ###########
            loss = loss1 + loss2 + loss3

            test_loss1 += loss1.item()
            test_loss2 += loss2.item()
            test_loss3 += loss3.item()
            test_loss += loss.item()

            Dis_1, Dis_2, Dis_3, Dis_4, Sim_1, Sim_2 = Evaluation_metrics(emo, dist_emo)
            test_Dist_1 += Dis_1
            test_Dist_2 += Dis_2
            test_Dist_3 += Dis_3
            test_Dist_4 += Dis_4
            test_Sim_1 += Sim_1
            test_Sim_2 += Sim_2

            _, predicted = torch.max(emo.data, 1)
            _, labeled = torch.max(dist_emo.data, 1)
            total += dist_emo.size(0)
            correct += predicted.eq(labeled.data).cpu().sum().numpy()
            acc = 100. * correct / total

            utils.progress_bar(batch_idx, len(testloader),
                               'Loss1: %.3f Loss2: %.3f Loss3: %.3f Loss: %.3f '
                               '| Chebyshev: %.3f Clark: %.3f Canberra: %.3f KL: %.3f Cosine: %.3f Inter: %.3f Acc: %.3f%%'
                               % (test_loss1 / (batch_idx + 1), test_loss2 / (batch_idx + 1),
                                  test_loss3 / (batch_idx + 1), test_loss / (batch_idx + 1),
                                  test_Dist_1 / (batch_idx + 1), test_Dist_2 / (batch_idx + 1),
                                  test_Dist_3 / (batch_idx + 1), test_Dist_4 / (batch_idx + 1),
                                  test_Sim_1 / (batch_idx + 1), test_Sim_2 / (batch_idx + 1), acc))

    return acc

if __name__ == '__main__':

    model_path = './ckpt/epoch-24.pkl'
    test_data_path = './dataset/facialRecon/test/test.csv'

    resNet50 = models.resnet50(pretrained=True)
    net = model_baseline(resNet50)
    # Load the model
    checkpoint = torch.load(model_path, map_location='cuda:0')
    net.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        net.cuda()
        
    # Load the test data
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(240),
            transforms.RandomCrop(224),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
    test_data = EmoData(test_data_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)

    CEloss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()
    KLloss = nn.KLDivLoss(size_average=False, reduce=True)
    # KLloss = nn.KLDivLoss(reduction='batchmean')
    MSELoss_theta = MSE_Loss_theta()
    Polarloss = PolarLoss()


    acc = test(net, testloader, KLloss, MSEloss, Polar_coordinates, MSELoss_theta, Polarloss)
    print('Test Accuracy: %.3f%%' % (acc))