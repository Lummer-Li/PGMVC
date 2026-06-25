import re
import scipy
import torch 
import random
import sklearn
import numpy as np
from numpy.random import randint
from torch.utils.data import Dataset
from typing import Tuple, List
from sklearn.preprocessing import OneHotEncoder

def get_mask(view_num, data_len, missing_rate):
    """
    To randomly generate missing matrix, simulating incomplete view data with complete view data.

    Parameters
    ----------
    view_num: number of view
    data_len: number of samples
    missing_rate: default(0.), super parameter to control the proportion of missing data
        the larger the value, the more missing data

    Returns
    -------
    missing_matrix: the missing matrix
    """
    mask_seed = 0
    np.random.seed(mask_seed)
    random.seed(mask_seed) 

    missing_rate = missing_rate / view_num              # 将缺失率平均分配到每个视图上，表示每个视图上的平均缺失比例
    one_rate = 1.0 - missing_rate                       # 每个视图中完整数据所占的比例
    if one_rate <= (1 / view_num):                      # 若完整率小于等于单个视图的比例 1/V，则随机选择一个视图为完整视图，并创建掩码矩阵
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:                                   # 如果完整率为 1
        matrix = np.ones((data_len, view_num)).astype(np.int64)          # 创建全1矩阵，表示所有数据完整
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        one_num = view_num * data_len * one_rate - data_len
        ratio = one_num / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int64)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int64))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int64)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int64)
        ratio = np.sum(matrix) / (view_num * data_len)
        error = abs(one_rate - ratio)
    return matrix

path = '../../our_datasets/'
def loadData(data_name):
    """
    Load multi-view dataset from .mat file with consistent data structure dynamically.
    
    Parameters:
        data_name (str): Path to .mat file containing dataset
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - features: NumPy object array of shape (1, n_views) containing view data
            - ground_truth: Flattened integer array of shape (n_samples,)
            
    Raises:
        ValueError: If dataset format is not recognized or required fields are missing
    """
    # 1. 加载数据
    data = scipy.io.loadmat(data_name) 
    
    # 2. 动态匹配所有类似 'X1', 'X2' 的键名，并按数字大小排序以保证视图顺序
    x_keys = [key for key in data.keys() if re.match(r'^X\d+$', key)]
    x_keys.sort(key=lambda k: int(k[1:])) 
    
    n_views = len(x_keys)
    if n_views == 0:
        raise ValueError(f"Required feature fields (X1, X2, etc.) missing in {data_name}")

    # 3. 动态初始化并填充 features 数组
    features = np.empty((1, n_views), dtype=object)
    for i, key in enumerate(x_keys):
        # 统一转换为 float32（原代码中绝大多数数据集也是这样处理的）
        features[0][i] = data[key].astype(np.float32)

    # 4. 提取标签 Y 并进行规范化处理
    if 'Y' not in data:
        raise ValueError(f"Ground truth field 'Y' missing in {data_name}")
        
    gnd = np.squeeze(data['Y']).astype(np.int32).flatten()
    
    return features, gnd

class MultiViewDataset(Dataset):
    def __init__(self, dataname: str, missing_rate: float):
        """
        Multi-view dataset loader with missing view handling
        
        Args:
            dataname: Name/path of the dataset (without .mat extension)
            missing_rate: Proportion of missing views (0-1)
        """
        # Load and preprocess data
        features, gnd = loadData(path + dataname + '.mat')
        self.num_views = features.shape[1]
        self.num_samples = len(gnd)
        
        # Convert features to tensors and normalize
        self.features = [
            torch.from_numpy(
                sklearn.preprocessing.MinMaxScaler().fit_transform(view)
            ).float()
            for view in features[0]
        ]
        
        # Store ground truth and indices
        self.gnd = torch.from_numpy(gnd).long()
        self.indices = torch.arange(self.num_samples)
        
        # Generate missing view mask
        self.mask = torch.from_numpy(
            get_mask(self.num_views, self.num_samples, missing_rate)
        ).float()
        
        # Validate dimensions
        self._validate_shapes()

    def _validate_shapes(self):
        """Ensure all components have consistent dimensions"""
        assert all(v.shape[0] == self.num_samples for v in self.features), \
            "Feature dimension mismatch"
        assert self.mask.shape == (self.num_samples, self.num_views), \
            "Mask shape mismatch"
        assert self.gnd.shape == (self.num_samples,), \
            "Ground truth shape mismatch"

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            tuple: Contains
                - views: List of tensors (one per view)
                - label: Ground truth label
                - idx: Sample index
                - mask: View availability mask
        """
        return (
            [view[idx] for view in self.features],  # Views
            self.gnd[idx],                          # Label
            torch.from_numpy(np.array(idx)),        # Idx
            self.indices[idx],                      # Index
            self.mask[idx]                          # Availability mask
        )

    @property
    def view_dims(self) -> list:
        """Get dimensionality of each view"""
        return [v.shape[1] for v in self.features]

def dataset_with_info(dataname: str, missing_rate: float = 0.0) -> Tuple[
    MultiViewDataset, int, int, int, List[int], np.ndarray
]:
    """
    Loads a multi-view dataset and provides comprehensive metadata
    
    Args:
        dataname: The name of Dataset file
        missing_rate: Proportion of missing views (0-1)
    
    Returns:
        Tuple containing:
        - Initialized MultiViewDataset
        - Sample count
        - Number of views
        - Cluster count
        - Feature dimensions per view
        - Ground truth labels
    
    Raises:
        ValueError: For invalid inputs or data loading failures
    """
    # Load and validate data
    try:
        features, gnd = loadData(path + dataname + '.mat')
    except Exception as e:
        raise ValueError(f"Data loading failed: {str(e)}") from e

    if features.size == 0:
        raise ValueError("Loaded features array is empty")
    if len(gnd) == 0:
        raise ValueError("No ground truth labels found")

    # Extract dataset characteristics
    num_views = features.shape[1]
    sample_count = features[0][0].shape[0]
    cluster_count = len(np.unique(gnd))
    feature_dims = [features[0][v].shape[1] for v in range(num_views)]

    # Initialize dataset
    dataset = MultiViewDataset(dataname, missing_rate=missing_rate)

    # Display dataset summary
    summary = (
        f"Dataset: {dataname}\n"
        f"Samples: {sample_count:,}\n"
        f"Views: {num_views}\n"
        f"Clusters: {cluster_count}\n"
        f"Feature Dimensions: {feature_dims}"
    )
    print(summary)

    return dataset, sample_count, num_views, cluster_count, feature_dims, gnd