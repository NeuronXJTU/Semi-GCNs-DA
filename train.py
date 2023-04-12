import os
import torch
import numpy as np
import torch.nn as nn
import data
import torch.utils.data
from sklearn.manifold import TSNE
from net import All_net
import torch.nn.functional as F
from test import testModel
from regLoss import cal_reg
from KMeans import generatePoints, kMeans


def train(args):
    isVal = False
    print("load source train dataset")
    train_set = data.BData('train', args)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    print("load target unlabel dataset")
    unlabel_set = data.BData('unlabel_data', args)
    unlabel_loader = torch.utils.data.DataLoader(
        unlabel_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    print("load target label dataset")
    reallabel_set = data.BData('reallabel_data', args)
    reallabel_loader = torch.utils.data.DataLoader(
        reallabel_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    print("load test dataset")
    test_set = data.BData('test', args)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    data_iter_s, data_iter_t_unl, data_iter_t, data_iter_t_test = iter(train_loader), iter(unlabel_loader), iter(
        reallabel_loader), iter(test_loader)
    len_train_source, len_train_target_semi, len_train_target, len_test = len(train_loader), len(unlabel_loader), len(
        reallabel_loader), len(test_loader)

    '''model'''
    print("load the model")
    model = All_net()
    print(torch.cuda.get_device_name(0))
    model = nn.DataParallel(model)
    model = model.cuda()
    for p in model.parameters():
        p.requires_grad = True

    '''    criterion'''

    criterion_cross = torch.nn.CrossEntropyLoss(ignore_index=2)
    criterion_cross = criterion_cross.cuda()
    criterion_l1 = nn.L1Loss(size_average=False)
    criterion_l1 = criterion_l1.cuda()
    criterion_cross1 = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=2)
    criterion_cross1 = criterion_cross1.cuda()
    criterion_l2 = nn.MSELoss(size_average=False)
    criterion_l2 = criterion_l2.cuda()

    optimizer = torch.optim.Adam(
        params=list(model.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    '''scheduler'''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5, verbose=True, min_lr=1e-8
    )

    '''training and evaling'''
    lr_count_s = 0
    lr_count_t = 0
    lr_count_tun = 0
    isFalseUp = 0
    bestTrain = 0
    bestTest = 0
    picrecordpath = args.out_dir
    pth_path = os.path.join(picrecordpath, 'pth')
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)

    print("start training")
    for epoch in range(0, args.epochs):
        param_lr_f = []
        if (epoch % len_train_source == 0) and (epoch != 0):
            data_iter_s = iter(train_loader)
            scheduler.step(lr_count_s)
            lr_count_s = lr_count_s + 1
        if (epoch % len_train_target_semi == 0) and (epoch != 0):
            data_iter_t_unl = iter(unlabel_loader)
            lr_count_tun = lr_count_tun + 1
            scheduler.step(lr_count_tun)
            print("test_acc")
            if (lr_count_tun > args.end_epochs):
                break
            print("lr_count_s")
            print(lr_count_s)
            resT = testModel(model, test_loader, False)
            if (resT > bestTest):
                torch.save(model, os.path.join(pth_path, "model" + str(epoch) + "acc" + str(resT) + '.pth'))

        if epoch % len_train_target == 0:
            data_iter_t = iter(reallabel_loader)
            lr_count_t = lr_count_t + 1
        model.train()

        '''Entropy optimization on source domain data'''
        (images_0, labels_0) = next(data_iter_s)
        images_0 = images_0.cuda()
        labels_0 = labels_0.cuda()
        outputS, GTun_source, _ = model(images_0)
        GTun_source = F.softmax(GTun_source)
        lossS = 0.1 * criterion_cross(outputS[0], labels_0)
        loss_ent = -0.1 * torch.mean(torch.sum(GTun_source *
                                               (torch.log(GTun_source + 1e-5)), 1))

        optimizer.zero_grad()
        loss_ent.backward()
        optimizer.step()

        '''Training on the target domain'''
        isFalseUp = isFalseUp + 1
        (images_1, labels_1) = next(data_iter_t)
        images_1 = images_1.cuda()
        labels_1 = labels_1.cuda()
        outT = model(images_1)
        outputT, ClassOpu = outT[0], outT[1]
        lossT = criterion_cross(outputT[0], labels_1)
        ClassOpu = F.softmax(ClassOpu)
        loss_ent = -0.1 * torch.mean(torch.sum(ClassOpu *
                                               (torch.log(ClassOpu + 1e-5)), 1))
        (images_2, labels_2, target_un_name) = next(data_iter_t_unl)
        images_2 = images_2.cuda()
        labels_2 = labels_2.cuda()
        preTun, _, featureUn = model(images_2)
        loss_unPresuo = criterion_cross1(preTun[0], labels_2)
        mask = ((labels_2 == 0) | (labels_2 == 1)).type(torch.cuda.FloatTensor)
        loss_unPresuo = torch.sum(loss_unPresuo * mask) / (mask.sum() + 1e-8)
        fprelabel = {}
        max_y_unlabeled = torch.max(torch.nn.functional.softmax(preTun[0], dim=1), 1, keepdim=True)
        all_fea = featureUn.cpu()
        tsTwo = TSNE(n_components=2)
        loc = tsTwo.fit_transform(all_fea.detach().numpy())

        '''Generate pseudo labels through K-means'''
        if (lr_count_s > args.begain_Kmeans):
            preLabels = []
            for i in range(0, max_y_unlabeled[0].shape[0]):
                if (max_y_unlabeled[0][i] > 0.8):
                    fprelabel[target_un_name[i]] = max_y_unlabeled[1][i].squeeze().cpu().numpy()
                    preLabels.append(fprelabel[target_un_name[i]])
                else:
                    preLabels.append(2)
            clusterCenterNumber = 2
            numberPoints = max_y_unlabeled[0].shape[0]
            points = generatePoints(numberPoints, loc, target_un_name, preLabels)
            _, clusterCenterTrace = kMeans(points, clusterCenterNumber)
            pointsCluster0 = []
            pointsCluster1 = []
            for i, point in enumerate(points):
                if (point.group == 0):
                    pointsCluster0.append(point)
                elif (point.group == 1):
                    pointsCluster1.append(point)
            pointsNumber0 = 0
            pointsNumber1 = 0
            pointsNumber2 = 0
            for i, point in enumerate(pointsCluster0):
                if (point.preLabel == 0):
                    pointsNumber0 += 1
                elif (point.preLabel == 1):
                    pointsNumber1 += 1
                elif (point.preLabel == 2):
                    pointsNumber2 += 1
            locMager = [pointsNumber0, pointsNumber1, pointsNumber2]
            pointsLabel0 = np.argmax(locMager)

            pointsNumber0 = 0
            pointsNumber1 = 0
            pointsNumber2 = 0
            for i, point in enumerate(pointsCluster1):
                if (point.preLabel == 0):
                    pointsNumber0 += 1
                elif (point.preLabel == 1):
                    pointsNumber1 += 1
                elif (point.preLabel == 2):
                    pointsNumber2 += 1
            locMager = [pointsNumber0, pointsNumber1, pointsNumber2]
            pointsLabel1 = np.argmax(locMager)

            if ((pointsLabel0 + pointsLabel1) == 1):

                fNPreLabel = {}
                for i, point in enumerate(pointsCluster0):
                    fNPreLabel[point.name] = pointsLabel0
                for i, point in enumerate(pointsCluster1):
                    fNPreLabel[point.name] = pointsLabel1
                unlabel_set.update(fNPreLabel)
            else:
                unlabel_set.update(fprelabel)

        if (args.regularization == None):
            loss_reg = 0
        elif (args.regularization == "L2"):
            loss_reg = cal_reg(model, criterion_l2, args.LgCoefficient)
        elif (args.regularization == "L1"):
            loss_reg = cal_reg(model, criterion_l1, args.LgCoefficient)

        if (lr_count_s < args.begain_unPresuo):
            loss_unPresuo = 0

        loss_tun = 0.5 * loss_ent + lossT + loss_unPresuo + loss_reg
        optimizer.zero_grad()
        loss_tun.backward()
        optimizer.step()



