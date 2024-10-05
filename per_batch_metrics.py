import numpy as np

LAND = 0
WATER = 1

SMOOTH = 1e-9

def IoU_(pred, real):

    intersect = np.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2

    intersect_land = intersect[intersect==LAND].shape[0]
    intersect_water = intersect[intersect==WATER].shape[0]
    union_land = pred[pred==LAND].shape[0] + real[real==LAND].shape[0] - intersect_land
    union_water = pred[pred==WATER].shape[0] + real[real==WATER].shape[0] - intersect_water

    IoU_land = intersect_land / union_land if union_land != 0 else np.nan
    IoU_water = intersect_water / union_water if union_water != 0 else np.nan

    return [IoU_land, IoU_water]

def balanced_accuracy(pred, real):

    # WATER = POSITIVE
    # LAND = NEGATIVE

    intersect = np.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
    exclusion = np.where(pred!=real, pred, 2)

    intersect_land = intersect[intersect==LAND].shape[0]    #TN
    intersect_water = intersect[intersect==WATER].shape[0]  #TP

    exclusion_land = exclusion[exclusion==LAND].shape[0]    #FN
    exclusion_water = exclusion[exclusion==WATER].shape[0]  #FP

    # balanced_acc = (TN / (TN+FP) + TP / (TP+FN) ) / 2
    balanced_acc = (intersect_land/(intersect_land+exclusion_water) + intersect_water/(intersect_water+exclusion_land))/2

    return balanced_acc

def precision(pred, real):

    # WATER = POSITIVE
    # LAND = NEGATIVE

    intersect = np.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
    exclusion = np.where(pred!=real, pred, 2)

    intersect_water = intersect[intersect==WATER].shape[0]  #TP

    exclusion_water = exclusion[exclusion==WATER].shape[0]  #FP

    # precision = TP / (TP + FP)

    prec = intersect_water / (intersect_water + exclusion_water)

    return prec

def recall(pred, real):

    # WATER = POSITIVE
    # LAND = NEGATIVE

    intersect = np.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2
    exclusion = np.where(pred!=real, pred, 2)

    intersect_water = intersect[intersect==WATER].shape[0]  #TP

    exclusion_land = exclusion[exclusion==LAND].shape[0]    #FN

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

    intersect_water = intersect[intersect==WATER].shape[0]  #TP

    exclusion_land = exclusion[exclusion==LAND].shape[0]    #FN
    exclusion_water = exclusion[exclusion==WATER].shape[0]  #FP

    # jaccard = TP / (TP + FP + FN)
    jacc = intersect_water / (intersect_water + exclusion_water + exclusion_land)

    return jacc