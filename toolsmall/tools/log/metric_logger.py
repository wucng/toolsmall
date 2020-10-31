from collections import deque, defaultdict
import numpy as np
import torch


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=10):
        self.deque = deque(maxlen=window_size)
        self.value = np.nan
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value
        self.value = value

    @property
    def median(self):
        values = np.array(self.deque)
        return np.median(values)

    @property
    def avg(self):
        values = np.array(self.deque)
        return np.mean(values)

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger:
    def __init__(self, delimiter=", "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.3f} ({:.3f})".format(name, meter.avg, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


if __name__=="__main__":
    meters = MetricLogger()

    meters.update(total_loss=12.0,**{"lbox":8.0,"lcls":4.0})
    meters.update(total_loss=15.0,**{"lbox":6.0,"lcls":9.0})
    meters.update(total_loss=13.0,**{"lbox":7.0,"lcls":6.0})

    print(meters.total_loss.median)
    print(meters.total_loss.avg)
    print(meters.total_loss.global_avg)
    print(meters.total_loss.series)
    """
    13.0
    13.333333333333334
    13.333333333333334
    [12.0, 15.0, 13.0]
    """


