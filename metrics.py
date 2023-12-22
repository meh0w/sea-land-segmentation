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