import os
import torch
import numpy as np
import math
import torch.nn as nn
import data
import torch.utils.data
from sklearn.manifold import TSNE
import random
from net import All_net
import torch.nn.functional as F
import copy
import pylab

FLOAT_MAX = 1e100

class Point:
	__slots__ = ["x", "y", "group","name","preLabel"]
	def __init__(self, x = 0, y = 0, group = 0,name="1",preLabel=0):
		self.x, self.y,self.name,self.preLabel, self.group = x, y, group,name,preLabel

def generatePoints(pointsNumber, features,names,preLabels):
	points = [Point() for _ in range(pointsNumber)]
	count = 0
	print(len(points))
	for i,point in enumerate(points):
		# print(str(i))
		points[i].x=features[i][0]
		points[i].y=features[i][1]
		points[i].name=names[i]
		points[i].preLabel=preLabels[i]
	return points

def solveDistanceBetweenPoints(pointA, pointB):
	return (pointA.x - pointB.x) * (pointA.x - pointB.x) + (pointA.y - pointB.y) * (pointA.y - pointB.y)

def getNearestCenter(point, clusterCenterGroup):
	minIndex = point.group
	minDistance = FLOAT_MAX
	for index, center in enumerate(clusterCenterGroup):
		distance = solveDistanceBetweenPoints(point, center)
		if (distance < minDistance):
			minDistance = distance
			minIndex = index
	return (minIndex, minDistance)

def kMeansPlusPlus(points, clusterCenterGroup):
	clusterCenterGroup[0] = copy.copy(random.choice(points))
	distanceGroup = [0.0 for _ in range(len(points))]
	sum = 0.0
	for index in range(1, len(clusterCenterGroup)):
		for i, point in enumerate(points):
			distanceGroup[i] = getNearestCenter(point, clusterCenterGroup[:index])[1]
			sum += distanceGroup[i]
		sum *= random.random()
		for i, distance in enumerate(distanceGroup):
			sum -= distance;
			if sum < 0:
				clusterCenterGroup[index] = copy.copy(points[i])
				break
	for point in points:
		point.group = getNearestCenter(point, clusterCenterGroup)[0]
	return

def kMeans(points, clusterCenterNumber):
	clusterCenterGroup = [Point() for _ in range(clusterCenterNumber)]
	kMeansPlusPlus(points, clusterCenterGroup)
	clusterCenterTrace = [[clusterCenter] for clusterCenter in clusterCenterGroup]
	tolerableError, currentError = 5.0, FLOAT_MAX
	count = 0
	while currentError >= tolerableError:
		count += 1
		countCenterNumber = [0 for _ in range(clusterCenterNumber)]
		currentCenterGroup = [Point() for _ in range(clusterCenterNumber)]
		for point in points:
			currentCenterGroup[point.group].x += point.x
			currentCenterGroup[point.group].y += point.y
			countCenterNumber[point.group] += 1
		for index, center in enumerate(currentCenterGroup):
			center.x /= countCenterNumber[index]
			center.y /= countCenterNumber[index]
		currentError = 0.0
		for index, singleTrace in enumerate(clusterCenterTrace):
			singleTrace.append(currentCenterGroup[index])
			currentError += solveDistanceBetweenPoints(singleTrace[-1], singleTrace[-2])
			clusterCenterGroup[index] = copy.copy(currentCenterGroup[index])
		for point in points:
			point.group = getNearestCenter(point, clusterCenterGroup)[0]
	return clusterCenterGroup, clusterCenterTrace

def cal_reg_l1(model, criterion_l1):
    reg_loss = 0
    np = 0
    for param in model.parameters():
        reg_loss += criterion_l1(param, torch.zeros_like(param))
        np += param.nelement()
    reg_loss = reg_loss / np
    return reg_loss


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
    len_all = (len_train_source, len_train_target_semi, len_train_target, len_test)

    '''model'''

    model = All_net()
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
    '''store'''

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
    for epoch in range(0, args.epochs):

        param_lr_f = []
        if (epoch % len_train_source == 0) and  (epoch!=0):
            data_iter_s = iter(train_loader)
            scheduler.step(lr_count_s)
            lr_count_s = lr_count_s + 1
            print("lr_count_s")
            print(str(lr_count_s))
            print("train_acc")
            res = test(model, train_loader, True)
        if (epoch % len_train_target_semi == 0) and (epoch!=0):
            data_iter_t_unl = iter(unlabel_loader)
            lr_count_tun = lr_count_tun + 1
            scheduler.step(lr_count_tun)
            if (lr_count_tun % 3 == 0):
                print("test_acc")
                if (lr_count_tun > 300):
                    break
                print("lr_count_s")
                print(lr_count_s)
                resT = test(model, test_loader, False)
                if (resT > bestTest):
                    torch.save(model, os.path.join(pth_path, "model" + str(epoch) + "acc" + str(resT) + '.pth'))

        if epoch % len_train_target == 0:
            data_iter_t = iter(reallabel_loader)
            lr_count_t = lr_count_t + 1
        model.train()
        if (lr_count_s < args.pre_epochs):
            (images_0, labels_0) = next(data_iter_s)
            images_0 = images_0.cuda()
            labels_0 = labels_0.cuda()
            outputS, _, _ = model(images_0)
            reg_l1 = 1e-5 * cal_reg_l1(model, criterion_l1)
            lossS = 0.5*criterion_cross(outputS[0], labels_0)
            optimizer.zero_grad()
            lossS.backward()
            optimizer.step()
    # else:
        isFalseUp = isFalseUp + 1
        (images_1, labels_1) = next(data_iter_t)
        images_1 = images_1.cuda()
        labels_1 = labels_1.cuda()
        outT = model(images_1)
        outputT, ClassOpu = outT[0], outT[1]
        lossT = criterion_cross(outputT[0], labels_1)

        (images_2, labels_2, target_un_name) = next(data_iter_t_unl)
        images_2 = images_2.cuda()
        labels_2 = labels_2.cuda()
        preTun, GTun, featureUn = model(images_2)

        loss_unPresuo = criterion_cross1(preTun[0], labels_2)
        max_y_unlabeled = torch.max(torch.nn.functional.softmax(GTun, dim=1), 1, keepdim=True)[0]
        mask = ((labels_2 == 0) | (labels_2 == 1)).type(torch.cuda.FloatTensor)
        loss_unPresuo = torch.sum(loss_unPresuo * mask) / (mask.sum() + 1e-8)

        fprelabel = {}
        max_y_unlabeled = torch.max(torch.nn.functional.softmax(preTun[0], dim=1), 1, keepdim=True)

        all_fea = featureUn.cpu()
        # print(type(all_fea))
        tsTwo = TSNE(n_components=2)
        loc = tsTwo.fit_transform(all_fea.detach().numpy())
        # print(loc)

        preLabels=[]
        for i in range(0, max_y_unlabeled[0].shape[0]):
            if (max_y_unlabeled[0][i] > 0.8):
                fprelabel[target_un_name[i]] = max_y_unlabeled[1][i].squeeze().cpu().numpy()
                preLabels.append(fprelabel[target_un_name[i]])
            else:
                preLabels.append(2)

        if(lr_count_s<10):
            unlabel_set.update(fprelabel)
        else:
            clusterCenterNumber = 2
            numberPoints = max_y_unlabeled[0].shape[0]
            points = generatePoints(numberPoints, loc,target_un_name,preLabels)
            _, clusterCenterTrace = kMeans(points, clusterCenterNumber)
            pointsCluster0 = []
            pointsCluster1 = []
            pointsLabel0 = []
            pointsLabel1 = []
            for i,point in enumerate(points):
                if(point.group==0):
                    pointsCluster0.append(point)
                elif(point.group==1):
                    pointsCluster1.append(point)
            pointsNumber0=0
            pointsNumber1=0
            pointsNumber2=0
            for i,point in enumerate(pointsCluster0):
                if(point.preLabel==0):
                    pointsNumber0+=1
                elif(point.preLabel==1):
                    pointsNumber1+=1
                elif(point.preLabel==2):
                    pointsNumber2+=1
            locMager = [pointsNumber0,pointsNumber1,pointsNumber2]
            pointsLabel0=np.argmax(locMager)

            pointsNumber0=0
            pointsNumber1=0
            pointsNumber2=0
            for i,point in enumerate(pointsCluster1):
                if(point.preLabel==0):
                    pointsNumber0+=1
                elif(point.preLabel==1):
                    pointsNumber1+=1
                elif(point.preLabel==2):
                    pointsNumber2+=1
            locMager = [pointsNumber0,pointsNumber1,pointsNumber2]
            pointsLabel1=np.argmax(locMager)

            if((pointsLabel0+pointsLabel1)==1):

                fNPreLabel={}
                for i,point in enumerate(pointsCluster0):
                    fNPreLabel[point.name] = pointsLabel0
                for i,point in enumerate(pointsCluster1):
                    fNPreLabel[point.name] = pointsLabel1
                unlabel_set.update(fNPreLabel)
            else:
                unlabel_set.update(fprelabel)



        GTun = F.softmax(GTun)
        loss_ent = -0.1 * torch.mean(torch.sum(GTun *
                                               (torch.log(GTun + 1e-5)), 1))
        reg_l2 = cal_reg_l1(model, criterion_l2)
        if(lr_count_s<args.begain_ent):
            loss_ent=0
            loss_unPresuo=0
        loss_tun =0.5*loss_ent+lossT+loss_unPresuo+0.7*reg_l2
        # loss_tun =loss_ent+ lossT+loss_unPresuo
        optimizer.zero_grad()
        loss_tun.backward()
        optimizer.step()



def test(model, data, isTrain):
    model.eval()
    with torch.no_grad():
        TP, FP, TN, FN, denominator, numerator = 0, 0, 0, 0, 0, 0
        if (not isTrain):
            for (
                    i, (images_0, labels_0)
            ) in enumerate(data):
                images_0, labels_0 = images_0.cuda(), labels_0.cuda()
                labeled_count = images_0.size(0)
                outputTest, _, _ = model(images_0)
                # print(outputTest)
                outputMax = torch.max(outputTest[0], 1)[1]
                pred = outputMax.cpu().data.numpy()
                mask = labels_0.cpu().data.numpy()
                numerator += np.sum(pred == mask)
                denominator += mask.shape[0]
                TP += sum((pred == 1)[mask == 1])
                FP += sum((pred == 1)[mask == 0])
                TN += sum((pred == 0)[mask == 0])
                FN += sum((pred == 0)[mask == 1])
        else:
            for (
                    i, (images_0, labels_0)
            ) in enumerate(data):
                images_0, labels_0 = images_0.cuda(), labels_0.cuda()
                labeled_count = images_0.size(0)
                outputTest, _, _ = model(images_0)
                # print(outputTest)
                outputMax = torch.max(outputTest[0], 1)[1]
                pred = outputMax.cpu().data.numpy()
                mask = labels_0.cpu().data.numpy()
                numerator += np.sum(pred == mask)
                denominator += mask.shape[0]
                TP += sum((pred == 1)[mask == 1])
                FP += sum((pred == 1)[mask == 0])
                TN += sum((pred == 0)[mask == 0])
                FN += sum((pred == 0)[mask == 1])
        t_acc = numerator / denominator
        specificity = TN / (FP + TN)
        sensitivity = TP / (TP + FN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1_scall = 2 * precision * recall / (precision + recall)
        print("t_acc")
        # print("\n")
        print(t_acc)
        print("specificity")
        # print("\n")
        print(specificity)
        print("sensitivity")
        # print("\n")
        print(sensitivity)
        print("f1_scall")
        # print("\n")
        print(f1_scall)
        print("\n")
        return t_acc

