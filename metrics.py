import numpy as np

LAND = 1
WATER = 0

def IoU(pred, real):

    intersect = np.where(pred==real, pred, 2) #if interesction inserts the predicted element else 2

    intersect_land = intersect[intersect==LAND].shape[0]
    intersect_water = intersect[intersect==WATER].shape[0]
    union_land = pred[pred==LAND].shape[0] + real[real==LAND].shape[0] - intersect_land
    union_water = pred[pred==WATER].shape[0] + real[real==WATER].shape[0] - intersect_water

    IoU_land = intersect_land / union_land
    IoU_water = intersect_water / union_water

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