import keras
print(keras.__version__)
from keras import ops

class All_metrics(keras.metrics.Metric):

  def __init__(self, name='all_metrics', **kwargs):
    super().__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')
    self.LAND = 0
    self.WATER = 1

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = ops.cast(y_true, "bool")
    y_pred = ops.cast(y_pred, "bool")

    # values = ops.logical_and(ops.equal(y_true, True), ops.equal(y_pred, True))
    # values = ops.cast(values, self.dtype)
    # if sample_weight is not None:
    #   sample_weight = ops.cast(sample_weight, self.dtype)
    #   values = values * sample_weight
    # self.true_positives.assign_add(ops.sum(values))

    intersect = ops.where(y_pred==y_true, y_pred, 2) #if interesction inserts the predicted element else 2
    exclusion = ops.where(y_pred!=y_true, y_pred, 2)

    intersect_land = ops.count_nonzero(intersect==self.LAND, axis=(1,2))          #TN
    intersect_water = ops.count_nonzero(intersect==self.WATER, axis=(1,2))        #TP

    union_land = ops.count_nonzero(y_pred==self.LAND, axis=(1,2)) + ops.count_nonzero(y_true==self.LAND, axis=(1,2)) - intersect_land
    union_water = ops.count_nonzero(y_pred==self.WATER, axis=(1,2)) + ops.count_nonzero(y_true==self.WATER, axis=(1,2)) - intersect_water

    exclusion_land = ops.count_nonzero(exclusion==self.LAND, axis=(1,2))          #FN
    exclusion_water = ops.count_nonzero(exclusion==self.WATER, axis=(1,2))        #FP

    IoU_land = ops.divide(ops.add(intersect_land, self.smooth), ops.add((union_land, self.smooth)))
    IoU_water = ops.divide(ops.add(intersect_water, self.smooth), ops.add((union_water, self.smooth)))

    pixels_per_img = y_pred.shape[1] * y_pred.shape[2]

    acc = ops.divide(ops.add(intersect_land, intersect_water), pixels_per_img)

  def result(self):
    return self.true_positives

  def reset_states(self):
    self.true_positives.assign(0)


m = All_metrics()
m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
print(f'Intermediate result: {m.result().numpy()}')

m.update_state([1, 1, 1, 1], [0, 1, 1, 0])
print(f'Intermediate result: {m.result().numpy()}')