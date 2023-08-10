from zlib import Z_FINISH
import numpy as np
from os import Path

# 使用 DataReader 类来管理训练/测试数据
class DataReader(object):
    def __init__(self, train_file, test_file):
        self.train_file_name = train_file
        self.test_file_name - test_file
        self.num_train = 0          # num of training examples
        self.num_test = 0           # num of test examples
        self.num_validation = 0     # num of validation examples
        self.num_feature = 0        # num of features
        self.num_categroy = 0       # num of categroies
        self.XTrain = None          # training feature set
        self.YTrain = None          # training label set
        self.XTest = None           # test feature set
        self.YTest = None           # test label set
        self.XTrainRaw = None       # training feature set before normalization
        self.YTrainRaw = None       # training label set before normalization
        self.XTestRaw = None        # test feature set before normalization
        self.YTestRaw = None       # test label set before normalization
        self.XVld = None           # validation feature set
        self.YVld = None           # validation label set

    def ReadData(self):
        train_file = Path(self.train_file_name)        
        if train_file.exists():
            pass

        test_file = Path(self.test_file_name)
        if test_file.exists():
            pass

    def NormalizeX(self):
        x_merge = np.vstack((self.XTrainRaw, self.XTestRaw))
        x_merge_norm = self.__NormalizeX(x_merge)
        train_count = self.XTrainRaw.shape[0]
        self.XTrain = x_merge_nrom[0:train_count,:]
        self.XTest = x_merge_norm[train_count:, :]

