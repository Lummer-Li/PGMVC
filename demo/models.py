import torch
import torch.nn as nn
import torch_clustering

class Encoder(nn.Module):
    """
    Flexible multi-layer perceptron encoder with optional batch normalization.

    Args:
        dims (list): Layer dimensions [input_dim, hidden1, ..., output_dim]
        bn (bool): Add batch norm after hidden layers
    """
    def __init__(self, dims, bn = False):
        super(Encoder, self).__init__()
        models = []
        for i in range(len(dims) - 1):
            models.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                # models.append(nn.BatchNorm1d(dims[i + 1]))
                # models.append(nn.ReLU(inplace=True))
                # models.append(nn.Dropout(0.5))
                models.append(nn.Identity())
        self.models = nn.Sequential(*models)
    def forward(self, X):
        return self.models(X)
    
class Decoder(nn.Module):
    """
    Multi-layer perceptron decoder for feature reconstruction.

    Architecture: Series of linear layers with ReLU activation on final layer.
    Typical use: Expanding latent representations to high-dimensional outputs.
    """
    def __init__(self, dims):
        super(Decoder, self).__init__()
        models = []
        for i in range(len(dims) - 1):
            models.append(nn.Linear(dims[i], dims[i + 1]))
            if i == len(dims) - 2:
                # models.append(nn.BatchNorm1d(dims[i + 1]))
                # models.append(nn.ReLU(inplace=True))
                # models.append(nn.Dropout(0.5))
                models.append(nn.Identity())
        self.models = nn.Sequential(*models)
    
    def forward(self, X):
        return self.models(X)
    
class MyNet(nn.Module):
    """
    Multi-view autoencoder network with dynamic granularity clustering.

    Key Components:
        - View-specific encoders/decoders
        - Latent space projection
        - Differentiable K-means clustering
    """
    def __init__(self, args, input_dims, view_num, class_num):
        super().__init__()
        # 架构参数初始化
        self.input_dims = input_dims  # 每个视图的输入维度
        self.view = view_num          # 视图数量
        self.class_num = class_num    # 聚类数量
        self.embedding_dim = args.embedding_dim  # 嵌入空间维度
        self.h_dims = args.hidden_dims  # 编码器隐藏层维度
        self.device = args.device     # 计算设备

        # 反转隐藏层维度用于解码器构建
        h_dims_reverse = list(reversed(args.hidden_dims))
        
        # 视图专属组件
        self.encoders = []  # 每个视图的编码器
        self.decoders = []  # 每个视图的解码器
        for v in range(self.view):
            # 编码器: Input -> Hidden Layers -> Embedding
            self.encoders.append(
                Encoder([input_dims[v]] + self.h_dims + [self.embedding_dim], bn=True).to(self.device)
            )
            # 解码器: Embedding -> Reversed Hidden -> Input
            self.decoders.append(
                Decoder([self.embedding_dim] + h_dims_reverse + [input_dims[v]]).to(self.device)
            )

        # 注册为 ModuleList 以便参数被正确追踪
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        
    def forward(self, xs, clustering=False, target=None):
        """
        Forward pass with view-specific processing.

        Args:
            xs: List of input tensors for each view
            clustering: Flag for clustering mode
            target: Optional target labels

        Returns:
            xrs: Reconstructed inputs per view
            zs: Latent embeddings per view
        """
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            xrs.append(xr)
            zs.append(z)

        return xrs, zs
        
    
    def clustering(self, features, num_clusters, max_iter=300):
        """
        使用 PyTorch K-Means 对特征进行聚类。

        Args:
            features: 特征张量 (N, D)
            num_clusters: 聚类数量
            max_iter: 最大迭代次数

        Returns:
            psedo_labels: 聚类伪标签 (N,)
        """
        kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': 0,
            'n_clusters': num_clusters,
            'verbose': False
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=max_iter, tol=1e-4, **kwargs)
        psedo_labels = clustering_model.fit_predict(features.to(dtype=torch.float64))
        
        return psedo_labels
    