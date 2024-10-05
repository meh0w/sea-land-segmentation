import numpy as np
import tensorflow as tf

LAND = 0
WATER = 1

SMOOTH = 1e-9

def IoU(pred, real):

    if len(pred.shape) == 2:
        pred = pred.reshape(1, pred.shape[0], pred.shape[1])
        real = real.reshape(1, real.shape[0], real.shape[1])

    intersect = np.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2

    intersect_land = np.count_nonzero(intersect==LAND, axis=(1,2))
    intersect_water = np.count_nonzero(intersect==WATER, axis=(1,2))
    union_land = np.count_nonzero(pred==LAND, axis=(1,2)) + np.count_nonzero(real==LAND, axis=(1,2)) - intersect_land
    union_water = np.count_nonzero(pred==WATER, axis=(1,2)) + np.count_nonzero(real==WATER, axis=(1,2)) - intersect_water

    IoU_land = (intersect_land + SMOOTH) / (union_land + SMOOTH)
    IoU_water = (intersect_water + SMOOTH) / (union_water + SMOOTH)

    return [IoU_land, IoU_water]

def dice_coeff(pred, real):

    intersect = np.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2

    intersect_land = intersect[intersect==LAND].shape[0]
    intersect_water = intersect[intersect==WATER].shape[0]
    sum_land = pred[pred==LAND].shape[0] + real[real==LAND].shape[0]
    sum_water = pred[pred==WATER].shape[0] + real[real==WATER].shape[0]

    dice_land = 2 * intersect_land / sum_land
    dice_water = 2 * intersect_water / sum_water

    return [dice_land, dice_water]

def accuracy(pred, real):

    correct = np.where(pred==real, 1, 0)
    tp_tn = np.count_nonzero(correct, axis=(1,2))
    pixels_per_img = pred.shape[1] * pred.shape[2]

    acc = tp_tn/pixels_per_img

    return np.mean(acc)

def balanced_accuracy(pred, real):

    # WATER = POSITIVE
    # LAND = NEGATIVE

    intersect = np.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
    exclusion = np.where(pred!=real, pred, 2)

    intersect_land = np.count_nonzero(intersect==LAND, axis=(1,2))
    intersect_water = np.count_nonzero(intersect==WATER, axis=(1,2))

    exclusion_land = np.count_nonzero(exclusion==LAND, axis=(1,2))
    exclusion_water = np.count_nonzero(exclusion==WATER, axis=(1,2))

    # balanced_acc = (TN / (TN+FP) + TP / (TP+FN) ) / 2
    balanced_acc = (intersect_land/(intersect_land+exclusion_water) + intersect_water/(intersect_water+exclusion_land))/2

    return balanced_acc

def precision(pred, real):

    # WATER = POSITIVE
    # LAND = NEGATIVE

    intersect = np.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
    exclusion = np.where(pred!=real, pred, 2)

    intersect_water = np.count_nonzero(intersect==WATER, axis=(1,2))

    exclusion_water = np.count_nonzero(exclusion==WATER, axis=(1,2))

    # precision = TP / (TP + FP)
    prec = intersect_water / (intersect_water + exclusion_water)

    return prec

def recall(pred, real):

    # WATER = POSITIVE
    # LAND = NEGATIVE

    intersect = np.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
    exclusion = np.where(pred!=real, pred, 2)

    intersect_water = np.count_nonzero(intersect==WATER, axis=(1,2))

    exclusion_land = np.count_nonzero(exclusion==LAND, axis=(1,2))

    # recall = TP / (TP + FN)

    rec = intersect_water / (intersect_water + exclusion_land)

    return rec

def f1score(pred, real):

    prec = precision(pred, real)
    rec = recall(pred, real)

    # F1-score = 2 * (precision * recall) / (precision + recall)

    f1 = 2 * (prec * rec) / (prec + rec)

    return f1

def jaccard(pred, real):

    # WATER = POSITIVE
    # LAND = NEGATIVE

    intersect = np.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
    exclusion = np.where(pred!=real, pred, 2)

    intersect_water = np.count_nonzero(intersect==WATER, axis=(1,2))

    exclusion_land = np.count_nonzero(exclusion==LAND, axis=(1,2))
    exclusion_water = np.count_nonzero(exclusion==WATER, axis=(1,2))

    # jaccard = TP / (TP + FP + FN)
    jacc = intersect_water / (intersect_water + exclusion_water + exclusion_land)

    return jacc

class ConfusionMatrix():
    def __init__(self, prefix="", smooth=1e-9):
        self.TP = 0
        self.TN = 0
        self.FN = 0
        self.FP = 0
        self.smooth = smooth

    def calc(self, pred, real):
        if len(pred.shape) == 2:
            pred = pred.reshape(1, pred.shape[0], pred.shape[1])
            real = real.reshape(1, real.shape[0], real.shape[1])

        intersect = np.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
        exclusion = np.where(pred!=real, pred, 2)

        intersect_land = np.count_nonzero(intersect==LAND, axis=(1,2))          #TN
        intersect_water = np.count_nonzero(intersect==WATER, axis=(1,2))        #TP

        union_land = np.count_nonzero(pred==LAND, axis=(1,2)) + np.count_nonzero(real==LAND, axis=(1,2)) - intersect_land
        union_water = np.count_nonzero(pred==WATER, axis=(1,2)) + np.count_nonzero(real==WATER, axis=(1,2)) - intersect_water

        exclusion_land = np.count_nonzero(exclusion==LAND, axis=(1,2))          #FN
        exclusion_water = np.count_nonzero(exclusion==WATER, axis=(1,2))        #FP
        self.add_to_confusion_matrix(intersect_land,intersect_water,exclusion_land,exclusion_water)

    def add_to_confusion_matrix(self, TN, TP, FN, FP):
        self.TP += TP
        self.TN += TN
        self.FN += FN
        self.FP += FP

    def get_IoU_new(self):
        # WATER = POSITIVE
        # LAND = NEGATIVE
        IoU_water = (self.TP + self.smooth) / (self.TP + self.FP + self.FN + self.smooth)
        IoU_land = (self.TN + self.smooth) / (self.TN + self.FN + self.FP + self.smooth)
        IoU_mean = (IoU_water+IoU_land)/2
        return IoU_land, IoU_water, IoU_mean
        

def all_metrics(pred, real, prefix="", smooth=1e-9):

    # WATER = POSITIVE
    # LAND = NEGATIVE

    if len(pred.shape) == 2:
        pred = pred.reshape(1, pred.shape[0], pred.shape[1])
        real = real.reshape(1, real.shape[0], real.shape[1])

    intersect = np.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
    exclusion = np.where(pred!=real, pred, 2)

    intersect_land = np.count_nonzero(intersect==LAND, axis=(1,2))          #TN
    intersect_water = np.count_nonzero(intersect==WATER, axis=(1,2))        #TP

    union_land = np.count_nonzero(pred==LAND, axis=(1,2)) + np.count_nonzero(real==LAND, axis=(1,2)) - intersect_land
    union_water = np.count_nonzero(pred==WATER, axis=(1,2)) + np.count_nonzero(real==WATER, axis=(1,2)) - intersect_water

    exclusion_land = np.count_nonzero(exclusion==LAND, axis=(1,2))          #FN
    exclusion_water = np.count_nonzero(exclusion==WATER, axis=(1,2))        #FP

    IoU_land = (intersect_land + smooth) / (union_land + smooth)
    IoU_water = (intersect_water + smooth) / (union_water + smooth)

    pixels_per_img = pred.shape[1] * pred.shape[2]

    acc = (intersect_land + intersect_water) / pixels_per_img

    # balanced_acc = (TN / (TN+FP) + TP / (TP+FN) ) / 2
    balanced_acc = ((intersect_land+smooth)/(intersect_land+exclusion_water+smooth) + (intersect_water+smooth)/(intersect_water+exclusion_land+smooth))/2

    # precision = TP / (TP + FP)
    prec = (intersect_water+smooth) / (intersect_water + exclusion_water+smooth)

    # recall = TP / (TP + FN)
    rec = (intersect_water+smooth) / (intersect_water + exclusion_land+smooth)

    # F1-score = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (prec * rec) / (prec + rec)

    # jaccard = TP / (TP + FP + FN)
    jacc = (intersect_water+smooth) / (intersect_water + exclusion_water + exclusion_land+smooth)

    # MCC = (TP * TN - FP * FN) / (sqrt[(TP + FP)(TP + FN)(TN + FP)(TN + FN)])
    mcc = ((intersect_water*intersect_land)-(exclusion_water*exclusion_land)+smooth) / (np.sqrt((intersect_water+exclusion_water)*(intersect_water+exclusion_land)*(intersect_land+exclusion_water)*(intersect_land+exclusion_land))+smooth)

    # kappa = 2 * (TP * TN - FP * FN) / ((TP + FP)*(TN + FP) + (TP + FN)*(TN + FN))
    kappa = ((intersect_water*intersect_land)-(exclusion_water*exclusion_land)+smooth) / (((intersect_water+exclusion_water)*(intersect_land+exclusion_water))+((intersect_water+exclusion_land)*(intersect_land+exclusion_land))+smooth)

    return {
        f'{prefix} IoU land': IoU_land,
        f'{prefix} IoU water': IoU_water,
        f'{prefix} IoU mean': tf.reduce_mean([IoU_land, IoU_water], axis=0).numpy() ,
        f'{prefix} Accuracy': acc,
        f'{prefix} Balanced accuracy': balanced_acc,
        f'{prefix} Precision': prec,
        f'{prefix} Recall': rec,
        f'{prefix} F1-score': f1,
        f'{prefix} Jaccard Index': jacc,
        f'{prefix} MCC': mcc,
        f'{prefix} Cohens Kappa': kappa
    }









# test
# for i in range(1000):
#     a = np.random.randint(0, 2, size=(30, 256, 256))
#     b = np.random.randint(0, 2, size=(30, 256, 256))

#     # good: accuracy
#     # bad: rest

#     res1 = accuracy(a, b)
#     res2 = accuracy_(a, b)

#     if res1 != res2:
#         print(f'{res1} \n {res2}')