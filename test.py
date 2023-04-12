import torch
import numpy as np

def testModel(model, data, isTrain):
    model.eval()
    with torch.no_grad():
        TP, FP, TN, FN, denominator, numerator = 0, 0, 0, 0, 0, 0
        if (not isTrain):
            for (
                    i, (images_0, labels_0)
            ) in enumerate(data):
                images_0, labels_0 = images_0.cuda(), labels_0.cuda()
                outputTest, _, _ = model(images_0)
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
                outputTest, _, _ = model(images_0)
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