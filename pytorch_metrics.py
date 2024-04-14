import torch

LAND = 0
WATER = 1

SMOOTH = 1e-9

def IoU(pred, real):

    if len(pred.shape) == 2:
        pred = pred.reshape(1, pred.shape[0], pred.shape[1])
        real = real.reshape(1, real.shape[0], real.shape[1])

    intersect = torch.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2

    intersect_land = torch.count_nonzero(intersect==LAND, dim=(1,2))
    intersect_water = torch.count_nonzero(intersect==WATER, dim=(1,2))
    union_land = torch.count_nonzero(pred==LAND, dim=(1,2)) + torch.count_nonzero(real==LAND, dim=(1,2)) - intersect_land
    union_water = torch.count_nonzero(pred==WATER, dim=(1,2)) + torch.count_nonzero(real==WATER, dim=(1,2)) - intersect_water

    IoU_land = (intersect_land + SMOOTH) / (union_land + SMOOTH)
    IoU_water = (intersect_water + SMOOTH) / (union_water + SMOOTH)

    return [IoU_land, IoU_water]

def dice_coeff(pred, real):

    intersect = torch.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2

    intersect_land = intersect[intersect==LAND].shape[0]
    intersect_water = intersect[intersect==WATER].shape[0]
    sum_land = pred[pred==LAND].shape[0] + real[real==LAND].shape[0]
    sum_water = pred[pred==WATER].shape[0] + real[real==WATER].shape[0]

    dice_land = 2 * intersect_land / sum_land
    dice_water = 2 * intersect_water / sum_water

    return [dice_land, dice_water]

def accuracy(pred, real):

    correct = torch.where(pred==real, 1, 0)
    tp_tn = torch.count_nonzero(correct, dim=(1,2))
    pixels_per_img = pred.shape[1] * pred.shape[2]

    acc = tp_tn/pixels_per_img

    return torch.mean(acc)

def balanced_accuracy(pred, real):

    # WATER = POSITIVE
    # LAND = NEGATIVE

    intersect = torch.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
    exclusion = torch.where(pred!=real, pred, 2)

    intersect_land = torch.count_nonzero(intersect==LAND, dim=(1,2))
    intersect_water = torch.count_nonzero(intersect==WATER, dim=(1,2))

    exclusion_land = torch.count_nonzero(exclusion==LAND, dim=(1,2))
    exclusion_water = torch.count_nonzero(exclusion==WATER, dim=(1,2))

    # balanced_acc = (TN / (TN+FP) + TP / (TP+FN) ) / 2
    balanced_acc = (intersect_land/(intersect_land+exclusion_water) + intersect_water/(intersect_water+exclusion_land))/2

    return balanced_acc

def precision(pred, real):

    # WATER = POSITIVE
    # LAND = NEGATIVE

    intersect = torch.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
    exclusion = torch.where(pred!=real, pred, 2)

    intersect_water = torch.count_nonzero(intersect==WATER, dim=(1,2))

    exclusion_water = torch.count_nonzero(exclusion==WATER, dim=(1,2))

    # precision = TP / (TP + FP)
    prec = intersect_water / (intersect_water + exclusion_water)

    return prec

def recall(pred, real):

    # WATER = POSITIVE
    # LAND = NEGATIVE

    intersect = torch.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
    exclusion = torch.where(pred!=real, pred, 2)

    intersect_water = torch.count_nonzero(intersect==WATER, dim=(1,2))

    exclusion_land = torch.count_nonzero(exclusion==LAND, dim=(1,2))

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

    intersect = torch.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
    exclusion = torch.where(pred!=real, pred, 2)

    intersect_water = torch.count_nonzero(intersect==WATER, dim=(1,2))

    exclusion_land = torch.count_nonzero(exclusion==LAND, dim=(1,2))
    exclusion_water = torch.count_nonzero(exclusion==WATER, dim=(1,2))

    # jaccard = TP / (TP + FP + FN)
    jacc = intersect_water / (intersect_water + exclusion_water + exclusion_land)

    return jacc

def all_metrics(pred, real, prefix="", smooth=1e-9):

    # WATER = POSITIVE
    # LAND = NEGATIVE

    if len(pred.shape) == 2:
        pred = pred.reshape(1, pred.shape[0], pred.shape[1])
        real = real.reshape(1, real.shape[0], real.shape[1])
    
    pred = torch.argmax(pred, dim=1)
    real = torch.argmax(real, dim=1)

    intersect = torch.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
    exclusion = torch.where(pred!=real, pred, 2)

    intersect_land = torch.count_nonzero(intersect==LAND, dim=(1,2))          #TN
    intersect_water = torch.count_nonzero(intersect==WATER, dim=(1,2))        #TP

    union_land = torch.count_nonzero(pred==LAND, dim=(1,2)) + torch.count_nonzero(real==LAND, dim=(1,2)) - intersect_land
    union_water = torch.count_nonzero(pred==WATER, dim=(1,2)) + torch.count_nonzero(real==WATER, dim=(1,2)) - intersect_water

    exclusion_land = torch.count_nonzero(exclusion==LAND, dim=(1,2))          #FN
    exclusion_water = torch.count_nonzero(exclusion==WATER, dim=(1,2))        #FP

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
    mcc = ((intersect_water*intersect_land)-(exclusion_water*exclusion_land)+smooth) / (torch.sqrt((intersect_water+exclusion_water)*(intersect_water+exclusion_land)*(intersect_land+exclusion_water)*(intersect_land+exclusion_land))+smooth)

    # kappa = 2 * (TP * TN - FP * FN) / ((TP + FP)*(TN + FP) + (TP + FN)*(TN + FN))
    kappa = ((intersect_water*intersect_land)-(exclusion_water*exclusion_land)+smooth) / (((intersect_water+exclusion_water)*(intersect_land+exclusion_water))+((intersect_water+exclusion_land)*(intersect_land+exclusion_land))+smooth)

    return {
        f'{prefix} IoU land': IoU_land,
        f'{prefix} IoU water': IoU_water,
        f'{prefix} IoU mean': torch.mean(torch.stack((IoU_land, IoU_water)), dim=0),
        f'{prefix} Accuracy': acc,
        f'{prefix} Balanced accuracy': balanced_acc,
        f'{prefix} Precision': prec,
        f'{prefix} Recall': rec,
        f'{prefix} F1-score': f1,
        f'{prefix} Jaccard Index': jacc,
        f'{prefix} MCC': mcc,
        f'{prefix} Cohens Kappa': kappa
    }

class All_metrics:
    def __init__(self, device, prefix="", smooth=1e-9):
        self.acc = torch.tensor([], dtype=torch.float32, device=device)
        self.balanced_acc = torch.tensor([], dtype=torch.float32, device=device)
        self.prec = torch.tensor([], dtype=torch.float32, device=device)
        self.rec = torch.tensor([], dtype=torch.float32, device=device)
        self.f1 = torch.tensor([], dtype=torch.float32, device=device)
        self.jacc = torch.tensor([], dtype=torch.float32, device=device)
        self.mcc = torch.tensor([], dtype=torch.float32, device=device)
        self.kappa = torch.tensor([], dtype=torch.float32, device=device)
        self.IoU_water = torch.tensor([], dtype=torch.float32, device=device)
        self.IoU_land = torch.tensor([], dtype=torch.float32, device=device)
        self.IoU_mean = torch.tensor([], dtype=torch.float32, device=device)
        self.smooth = smooth
        self.device = device
        self.prefix = prefix

    def calc(self, pred, real):
        # WATER = POSITIVE
        # LAND = NEGATIVE

        if len(pred.shape) == 2:
            pred = pred.reshape(1, pred.shape[0], pred.shape[1])
            real = real.reshape(1, real.shape[0], real.shape[1])
        
        pred = torch.argmax(pred, dim=1)
        real = torch.argmax(real, dim=1)

        intersect = torch.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
        exclusion = torch.where(pred!=real, pred, 2)

        intersect_land = torch.count_nonzero(intersect==LAND, dim=(1,2))          #TN
        intersect_water = torch.count_nonzero(intersect==WATER, dim=(1,2))        #TP

        union_land = torch.count_nonzero(pred==LAND, dim=(1,2)) + torch.count_nonzero(real==LAND, dim=(1,2)) - intersect_land
        union_water = torch.count_nonzero(pred==WATER, dim=(1,2)) + torch.count_nonzero(real==WATER, dim=(1,2)) - intersect_water

        exclusion_land = torch.count_nonzero(exclusion==LAND, dim=(1,2))          #FN
        exclusion_water = torch.count_nonzero(exclusion==WATER, dim=(1,2))        #FP

        IoU_land = (intersect_land + self.smooth) / (union_land + self.smooth)
        IoU_water = (intersect_water + self.smooth) / (union_water + self.smooth)

        pixels_per_img = pred.shape[1] * pred.shape[2]

        acc = (intersect_land + intersect_water) / pixels_per_img

        # balanced_acc = (TN / (TN+FP) + TP / (TP+FN) ) / 2
        balanced_acc = ((intersect_land+self.smooth)/(intersect_land+exclusion_water+self.smooth) + (intersect_water+self.smooth)/(intersect_water+exclusion_land+self.smooth))/2

        # precision = TP / (TP + FP)
        prec = (intersect_water+self.smooth) / (intersect_water + exclusion_water+self.smooth)

        # recall = TP / (TP + FN)
        rec = (intersect_water+self.smooth) / (intersect_water + exclusion_land+self.smooth)

        # F1-score = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (prec * rec) / (prec + rec)

        # jaccard = TP / (TP + FP + FN)
        jacc = (intersect_water+self.smooth) / (intersect_water + exclusion_water + exclusion_land+self.smooth)

        # MCC = (TP * TN - FP * FN) / (sqrt[(TP + FP)(TP + FN)(TN + FP)(TN + FN)])
        mcc = ((intersect_water*intersect_land)-(exclusion_water*exclusion_land)+self.smooth) / (torch.sqrt((intersect_water+exclusion_water)*(intersect_water+exclusion_land)*(intersect_land+exclusion_water)*(intersect_land+exclusion_land))+self.smooth)

        # kappa = 2 * (TP * TN - FP * FN) / ((TP + FP)*(TN + FP) + (TP + FN)*(TN + FN))
        kappa = ((intersect_water*intersect_land)-(exclusion_water*exclusion_land)+self.smooth) / (((intersect_water+exclusion_water)*(intersect_land+exclusion_water))+((intersect_water+exclusion_land)*(intersect_land+exclusion_land))+self.smooth)

        self.acc = torch.cat((self.acc, acc))
        self.balanced_acc = torch.cat((self.balanced_acc, balanced_acc))
        self.prec = torch.cat((self.prec, prec))
        self.rec = torch.cat((self.rec, rec))
        self.f1 = torch.cat((self.f1, f1))
        self.jacc = torch.cat((self.jacc, jacc))
        self.mcc = torch.cat((self.mcc, mcc))
        self.kappa = torch.cat((self.kappa, kappa))
        self.IoU_water = torch.cat((self.IoU_water, IoU_water))
        self.IoU_land = torch.cat((self.IoU_land, IoU_land))

    def clear(self):
        self.acc = torch.tensor([], dtype=torch.float32, device=self.device)
        self.balanced_acc = torch.tensor([], dtype=torch.float32, device=self.device)
        self.prec = torch.tensor([], dtype=torch.float32, device=self.device)
        self.rec = torch.tensor([], dtype=torch.float32, device=self.device)
        self.f1 = torch.tensor([], dtype=torch.float32, device=self.device)
        self.jacc = torch.tensor([], dtype=torch.float32, device=self.device)
        self.mcc = torch.tensor([], dtype=torch.float32, device=self.device)
        self.kappa = torch.tensor([], dtype=torch.float32, device=self.device)
        self.IoU_water = torch.tensor([], dtype=torch.float32, device=self.device)
        self.IoU_land = torch.tensor([], dtype=torch.float32, device=self.device)

    def get(self, mean=True):
        IoU_mean = torch.mean(torch.stack((self.IoU_land, self.IoU_water)), dim=0)
        output = {
            f'{self.prefix} IoU land': self.IoU_land,
            f'{self.prefix} IoU water': self.IoU_water,
            f'{self.prefix} IoU mean': IoU_mean,
            f'{self.prefix} Accuracy': self.acc,
            f'{self.prefix} Balanced accuracy': self.balanced_acc,
            f'{self.prefix} Precision': self.prec,
            f'{self.prefix} Recall': self.rec,
            f'{self.prefix} F1-score': self.f1,
            f'{self.prefix} Jaccard Index': self.jacc,
            f'{self.prefix} MCC': self.mcc,
            f'{self.prefix} Cohens Kappa': self.kappa
        }

        if mean:
            return {key: torch.mean(value).item() for key, value in output.items()}
        else:
            return output

class All_metrics_16:
    def __init__(self, device, prefix="", smooth=1e-9):
        self.acc = torch.tensor([], dtype=torch.float16, device=device)
        self.balanced_acc = torch.tensor([], dtype=torch.float16, device=device)
        self.prec = torch.tensor([], dtype=torch.float16, device=device)
        self.rec = torch.tensor([], dtype=torch.float16, device=device)
        self.f1 = torch.tensor([], dtype=torch.float16, device=device)
        self.jacc = torch.tensor([], dtype=torch.float16, device=device)
        self.mcc = torch.tensor([], dtype=torch.float16, device=device)
        self.kappa = torch.tensor([], dtype=torch.float16, device=device)
        self.IoU_water = torch.tensor([], dtype=torch.float16, device=device)
        self.IoU_land = torch.tensor([], dtype=torch.float16, device=device)
        self.IoU_mean = torch.tensor([], dtype=torch.float16, device=device)
        self.smooth = smooth
        self.device = device
        self.prefix = prefix

    def calc(self, pred, real):
        # WATER = POSITIVE
        # LAND = NEGATIVE

        if len(pred.shape) == 2:
            pred = pred.reshape(1, pred.shape[0], pred.shape[1])
            real = real.reshape(1, real.shape[0], real.shape[1])
        
        pred = torch.argmax(pred, dim=1)
        real = torch.argmax(real, dim=1)

        intersect = torch.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
        exclusion = torch.where(pred!=real, pred, 2)

        intersect_land = torch.count_nonzero(intersect==LAND, dim=(1,2))          #TN
        intersect_water = torch.count_nonzero(intersect==WATER, dim=(1,2))        #TP

        union_land = torch.count_nonzero(pred==LAND, dim=(1,2)) + torch.count_nonzero(real==LAND, dim=(1,2)) - intersect_land
        union_water = torch.count_nonzero(pred==WATER, dim=(1,2)) + torch.count_nonzero(real==WATER, dim=(1,2)) - intersect_water

        exclusion_land = torch.count_nonzero(exclusion==LAND, dim=(1,2))          #FN
        exclusion_water = torch.count_nonzero(exclusion==WATER, dim=(1,2))        #FP

        IoU_land = (intersect_land + self.smooth) / (union_land + self.smooth)
        IoU_water = (intersect_water + self.smooth) / (union_water + self.smooth)

        pixels_per_img = pred.shape[1] * pred.shape[2]

        acc = (intersect_land + intersect_water) / pixels_per_img

        # balanced_acc = (TN / (TN+FP) + TP / (TP+FN) ) / 2
        balanced_acc = ((intersect_land+self.smooth)/(intersect_land+exclusion_water+self.smooth) + (intersect_water+self.smooth)/(intersect_water+exclusion_land+self.smooth))/2

        # precision = TP / (TP + FP)
        prec = (intersect_water+self.smooth) / (intersect_water + exclusion_water+self.smooth)

        # recall = TP / (TP + FN)
        rec = (intersect_water+self.smooth) / (intersect_water + exclusion_land+self.smooth)

        # F1-score = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (prec * rec) / (prec + rec)

        # jaccard = TP / (TP + FP + FN)
        jacc = (intersect_water+self.smooth) / (intersect_water + exclusion_water + exclusion_land+self.smooth)

        # MCC = (TP * TN - FP * FN) / (sqrt[(TP + FP)(TP + FN)(TN + FP)(TN + FN)])
        mcc = ((intersect_water*intersect_land)-(exclusion_water*exclusion_land)+self.smooth) / (torch.sqrt((intersect_water+exclusion_water)*(intersect_water+exclusion_land)*(intersect_land+exclusion_water)*(intersect_land+exclusion_land))+self.smooth)

        # kappa = 2 * (TP * TN - FP * FN) / ((TP + FP)*(TN + FP) + (TP + FN)*(TN + FN))
        kappa = ((intersect_water*intersect_land)-(exclusion_water*exclusion_land)+self.smooth) / (((intersect_water+exclusion_water)*(intersect_land+exclusion_water))+((intersect_water+exclusion_land)*(intersect_land+exclusion_land))+self.smooth)

        self.acc = torch.cat((self.acc, acc))
        self.balanced_acc = torch.cat((self.balanced_acc, balanced_acc))
        self.prec = torch.cat((self.prec, prec))
        self.rec = torch.cat((self.rec, rec))
        self.f1 = torch.cat((self.f1, f1))
        self.jacc = torch.cat((self.jacc, jacc))
        self.mcc = torch.cat((self.mcc, mcc))
        self.kappa = torch.cat((self.kappa, kappa))
        self.IoU_water = torch.cat((self.IoU_water, IoU_water))
        self.IoU_land = torch.cat((self.IoU_land, IoU_land))

    def clear(self):
        self.acc = torch.tensor([], dtype=torch.float16, device=self.device)
        self.balanced_acc = torch.tensor([], dtype=torch.float16, device=self.device)
        self.prec = torch.tensor([], dtype=torch.float16, device=self.device)
        self.rec = torch.tensor([], dtype=torch.float16, device=self.device)
        self.f1 = torch.tensor([], dtype=torch.float16, device=self.device)
        self.jacc = torch.tensor([], dtype=torch.float16, device=self.device)
        self.mcc = torch.tensor([], dtype=torch.float16, device=self.device)
        self.kappa = torch.tensor([], dtype=torch.float16, device=self.device)
        self.IoU_water = torch.tensor([], dtype=torch.float16, device=self.device)
        self.IoU_land = torch.tensor([], dtype=torch.float16, device=self.device)

    def get(self, mean=True):
        IoU_mean = torch.mean(torch.stack((self.IoU_land, self.IoU_water)), dim=0)
        output = {
            f'{self.prefix} IoU land': self.IoU_land,
            f'{self.prefix} IoU water': self.IoU_water,
            f'{self.prefix} IoU mean': IoU_mean,
            f'{self.prefix} Accuracy': self.acc,
            f'{self.prefix} Balanced accuracy': self.balanced_acc,
            f'{self.prefix} Precision': self.prec,
            f'{self.prefix} Recall': self.rec,
            f'{self.prefix} F1-score': self.f1,
            f'{self.prefix} Jaccard Index': self.jacc,
            f'{self.prefix} MCC': self.mcc,
            f'{self.prefix} Cohens Kappa': self.kappa
        }

        if mean:
            return {key: torch.mean(value).item() for key, value in output.items()}
        else:
            return output


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