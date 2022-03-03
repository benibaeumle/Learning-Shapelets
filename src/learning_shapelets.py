from collections import OrderedDict
import warnings

import numpy as np
import torch
from torch import tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class MinEuclideanDistBlock(nn.Module):
    """
    Calculates the euclidean distances of a bunch of shapelets to a data set and performs global min-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MinEuclideanDistBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    def forward(self, x):
        """
        1) Unfold the data set 2) calculate euclidean distance 3) sum over channels and 4) perform global min-pooling
        @param x: the time series data
        @type x: tensor(float) of shape (num_samples, in_channels, len_ts)
        @return: Return the euclidean for each pair of shapelet and time series instance
        @rtype: tensor(num_samples, num_shapelets)
        """
        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        # calculate euclidean distance
        x = torch.cdist(x, self.shapelets, p=2)

        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3)
        # hard min compared to soft-min from the paper
        x, _ = torch.min(x, 3)
        return x

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets.transpose(1, 0)

    def set_shapelet_weights(self, weights):
        """
        Set weights for all shapelets in this block.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (num_shapelets, in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        # transpose since internally we need shape (in_channels, num_shapelets, shapelets_size)
        weights = weights.transpose(1, 0)

        if not list(weights.shape) == list(self.shapelets.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets.shape)}"
                             f"compared to {list(weights.shape)}")

        self.shapelets = nn.Parameter(weights)
        self.shapelets.retain_grad()

    def set_weights_of_single_shapelet(self, j, weights):
        """
        Set the weights of a single shapelet.
        @param j: The index of the shapelet to set
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not list(weights.shape) == list(self.shapelets[:, j].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets[:, j].shape)}"
                             f"compared to {list(weights[j].shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.Tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        self.shapelets[:, j] = weights
        self.shapelets = nn.Parameter(self.shapelets).contiguous()
        self.shapelets.retain_grad()


class MaxCosineSimilarityBlock(nn.Module):
    """
    Calculates the cosine similarity of a bunch of shapelets to a data set and performs global max-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MaxCosineSimilarityBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels
        self.relu = nn.ReLU()

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                                dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    def forward(self, x):
        """
        1) Unfold the data set 2) calculate norm of the data and the shapelets 3) calculate pair-wise dot-product
        4) sum over channels 5) perform a ReLU to ignore the negative values and 6) perform global max-pooling
        @param x: the time series data
        @type x: tensor(float) of shape (num_samples, in_channels, len_ts)
        @return: Return the cosine similarity for each pair of shapelet and time series instance
        @rtype: tensor(num_samples, num_shapelets)
        """
        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        # normalize with l2 norm
        x = x / x.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
        shapelets_norm = self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)
        # calculate cosine similarity via dot product on already normalized ts and shapelets
        x = torch.matmul(x, shapelets_norm.transpose(1, 2))
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        n_dims = x.shape[1]
        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3) / n_dims
        # ignore negative distances
        x = self.relu(x)
        x, _ = torch.max(x, 3)
        return x

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets.transpose(1, 0)

    def set_shapelet_weights(self, weights):
        """
        Set weights for all shapelets in this block.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (num_shapelets, in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        # transpose since internally we need shape (in_channels, num_shapelets, shapelets_size)
        weights = weights.transpose(1, 0)

        if not list(weights.shape) == list(self.shapelets.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets.shape)} "
                             f"compared to {list(weights.shape)}")

        self.shapelets = nn.Parameter(weights)

    def set_weights_of_single_shapelet(self, j, weights):
        """
        Set the weights of a single shapelet.
        @param j: The index of the shapelet to set
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not list(weights.shape) == list(self.shapelets[:, j].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets[:, j].shape)} "
                             f"compared to {list(weights[j].shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.Tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        self.shapelets[:, j] = weights
        self.shapelets = nn.Parameter(self.shapelets).contiguous()


class MaxCrossCorrelationBlock(nn.Module):
    """
    Calculates the cross-correlation of a bunch of shapelets to a data set, implemented via convolution and
    performs global max-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    # TODO Why is this multiple time slower than the other two implementations?
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MaxCrossCorrelationBlock, self).__init__()
        self.shapelets = nn.Conv1d(in_channels, num_shapelets, kernel_size=shapelets_size)
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.cuda()

    def forward(self, x):
        """
        1) Apply 1D convolution 2) Apply global max-pooling
        @param x: the data set of time series
        @type x: array(float) of shape (num_samples, in_channels, len_ts)
        @return: Return the most similar values for each pair of shapelet and time series instance
        @rtype: tensor(n_samples, num_shapelets)
        """
        x = self.shapelets(x)
        x, _ = torch.max(x, 2, keepdim=True)
        return x.transpose(2, 1)

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets.weight.data

    def set_shapelet_weights(self, weights):
        """
        Set weights for all shapelets in this block.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (num_shapelets, in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()

        if not list(weights.shape) == list(self.shapelets.weight.data.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape"
                             f"{list(self.shapelets.weight.data.shape)} compared to {list(weights.shape)}")

        self.shapelets.weight.data = weights

    def set_weights_of_single_shapelet(self, j, weights):
        """
        Set the weights of a single shapelet.
        @param j: The index of the shapelet to set
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not list(weights.shape) == list(self.shapelets.weight.data[j, :].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape"
                             f"{list(self.shapelets.weight.data[j, :].shape)} compared to {list(weights.shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        self.shapelets.weight.data[j, :] = weights


class ShapeletsDistBlocks(nn.Module):
    """
    Defines a number of blocks containing a number of shapelets, whereas
    the shapelets in each block have the same size.
    Parameters
    ----------
    shapelets_size_and_len : dict(int:int)
        keys are the length of the shapelets for a block and the values the number of shapelets for the block
    in_channels : int
        the number of input channels of the dataset
    dist_measure: 'string'
        the distance measure, either of 'euclidean', 'cross-correlation', or 'cosine'
    to_cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size_and_len, in_channels=1, dist_measure='euclidean', to_cuda=True):
        super(ShapeletsDistBlocks, self).__init__()
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = OrderedDict(sorted(shapelets_size_and_len.items(), key=lambda x: x[0]))
        self.in_channels = in_channels
        self.dist_measure = dist_measure
        if dist_measure == 'euclidean':
            self.blocks = nn.ModuleList(
                [MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                       in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cross-correlation':
            self.blocks = nn.ModuleList(
                [MaxCrossCorrelationBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cosine':
            self.blocks = nn.ModuleList(
                [MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        else:
            raise ValueError("dist_measure must be either of 'euclidean', 'cross-correlation', 'cosine'")

    def forward(self, x):
        """
        Calculate the distances of each shapelet block to the time series data x and concatenate the results.
        @param x: the time series data
        @type x: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: a distance matrix containing the distances of each shapelet to the time series data
        @rtype: tensor(float) of shape
        """
        out = torch.tensor([], dtype=torch.float).cuda() if self.to_cuda else torch.tensor([], dtype=torch.float)
        for block in self.blocks:
            out = torch.cat((out, block(x)), dim=2)

        return out

    def get_blocks(self):
        """
        @return: the list of shapelet blocks
        @rtype: nn.ModuleList
        """
        return self.blocks

    def get_block(self, i):
        """
        Get a specific shapelet block. The blocks are ordered (ascending) according to the shapelet lengths.
        @param i: the index of the block to fetch
        @type i: int
        @return: return shapelet block i
        @rtype: nn.Module, either
        """
        return self.blocks[i]

    def set_shapelet_weights_of_block(self, i, weights):
        """
        Set the weights of the shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (in_channels, num_shapelets, shapelets_size)
        @return:
        @rtype: None
        """
        self.blocks[i].set_shapelet_weights(weights)

    def get_shapelets_of_block(self, i):
        """
        Return the shapelet of shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @return: the weights of the shapelet block
        @rtype: tensor(float) of shape (in_channels, num_shapelets, shapelets_size)
        """
        return self.blocks[i].get_shapelets()

    def get_shapelet(self, i, j):
        """
        Return the shapelet at index j of shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param j: the index of the shapelet in shapelet block i
        @type j: int
        @return: return the weights of the shapelet
        @rtype: tensor(float) of shape
        """
        shapelet_weights = self.blocks[i].get_shapelets()
        return shapelet_weights[j, :]

    def set_shapelet_weights_of_single_shapelet(self, i, j, weights):
        """
        Set the weights of shapelet j of shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param j: the index of the shapelet in shapelet block i
        @type j: int
        @param weights: the new weights for the shapelet
        @type weights: array-like of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        self.blocks[i].set_weights_of_single_shapelet(j, weights)

    def get_shapelets(self):
        """
        Return a matrix of all shapelets. The shapelets are ordered (ascending) according to
        the shapelet lengths and padded with NaN.
        @return: a tensor of all shapelets
        @rtype: tensor(float) with shape (in_channels, num_total_shapelets, shapelets_size_max)
        """
        max_shapelet_len = max(self.shapelets_size_and_len.keys())
        num_total_shapelets = sum(self.shapelets_size_and_len.values())
        shapelets = torch.Tensor(num_total_shapelets, self.in_channels, max_shapelet_len)
        shapelets[:] = np.nan
        start = 0
        for block in self.blocks:
            shapelets_block = block.get_shapelets()
            end = start + block.num_shapelets
            shapelets[start:end, :, :block.shapelets_size] = shapelets_block
            start += block.num_shapelets
        return shapelets

class ShapeletsDistanceLoss(nn.Module):
    """
    Calculates the cosine similarity of a bunch of shapelets to a data set and performs global max-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, dist_measure='euclidean', k=6):
        super(ShapeletsDistanceLoss, self).__init__()
        if not dist_measure == 'euclidean' and not dist_measure == 'cosine':
            raise ValueError("Parameter 'dist_measure' must be either of 'euclidean' or 'cosine'.")
        if not isinstance(k, int):
            raise ValueError("Parameter 'k' must be an integer.")
        self.dist_measure = dist_measure
        self.k = k

    def forward(self, x):
        """
        Calculate the loss as the average distance to the top k best-matching time series.
        @param x: the shapelet transform
        @type x: tensor(float) of shape (batch_size, n_shapelets)
        @return: the computed loss
        @rtype: float
        """
        y_top, y_topi = torch.topk(x.clamp(1e-8), self.k, largest=False if self.dist_measure == 'euclidean' else True,
                                   sorted=False, dim=0)
        # avoid compiler warning
        y_loss = None
        if self.dist_measure == 'euclidean':
            y_loss = torch.mean(y_top)
        elif self.dist_measure == 'cosine':
            y_loss = torch.mean(1 - y_top)
        return y_loss

class ShapeletsSimilarityLoss(nn.Module):
    """
    Calculates the cosine similarity of each block of shapelets and averages over the blocks.
    ----------
    """
    def __init__(self):
        super(ShapeletsSimilarityLoss, self).__init__()

    def cosine_distance(self, x1, x2=None, eps=1e-8):
        """
        Calculate the cosine similarity between all pairs of x1 and x2. x2 can be left zero, in case the similarity
        between solely all pairs in x1 shall be computed.
        @param x1: the first set of input vectors
        @type x1: tensor(float)
        @param x2: the second set of input vectors
        @type x2: tensor(float)
        @param eps: add small value to avoid division by zero.
        @type eps: float
        @return: a distance matrix containing the cosine similarities
        @type: tensor(float)
        """
        x2 = x1 if x2 is None else x2
        # unfold time series to emulate sliding window
        x1 = x1.unfold(2, x2.shape[2], 1).contiguous()
        x1 = x1.transpose(0, 1)
        # normalize with l2 norm
        x1 = x1 / x1.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
        x2 = x2 / x2.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)

        # calculate cosine similarity via dot product on already normalized ts and shapelets
        x1 = torch.matmul(x1, x2.transpose(1, 2))
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        # and average over dims to keep range between 0 and 1
        n_dims = x1.shape[1]
        x1 = torch.sum(x1, dim=1) / n_dims
        return x1

    def forward(self, shapelet_blocks):
        """
        Calculate the loss as the sum of the averaged cosine similarity of the shapelets in between each block.
        @param shapelet_blocks: a list of the weights (as torch parameters) of the shapelet blocks
        @type shapelet_blocks: list of torch.parameter(tensor(float))
        @return: the computed loss
        @rtype: float
        """
        losses = 0.
        for block in shapelet_blocks:
            shapelets = block[1]
            shapelets.retain_grad()
            sim = self.cosine_distance(shapelets, shapelets)
            losses += torch.mean(sim)
        return losses


class LearningShapeletsModel(nn.Module):
    """
    Implements Learning Shapelets. Just puts together the ShapeletsDistBlocks with a
    linear layer on top.
    ----------
    shapelets_size_and_len : dict(int:int)
        keys are the length of the shapelets for a block and the values the number of shapelets for the block
    in_channels : int
        the number of input channels of the dataset
    num_classes: int
        the number of classes for classification
    dist_measure: 'string'
        the distance measure, either of 'euclidean', 'cross-correlation', or 'cosine'
    to_cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='euclidean',
                 to_cuda=True):
        super(LearningShapeletsModel, self).__init__()

        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure, to_cuda=to_cuda)
        self.linear = nn.Linear(self.num_shapelets, num_classes)

        if self.to_cuda:
            self.cuda()

    def forward(self, x, optimize='acc'):
        """
        Calculate the distances of each time series to the shapelets and stack a linear layer on top.
        @param x: the time series data
        @type x: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: the logits for the class predictions of the model
        @rtype: tensor(float) of shape (num_samples, num_classes)
        """
        x = self.shapelets_blocks(x)
        if optimize == 'acc':
            x = self.linear(x)
        x = torch.squeeze(x, 1)
        return x

    def transform(self, X):
        """
        Performs the shapelet transform with the input time series data x
        @param X: the time series data
        @type X: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: the shapelet transform of x
        @rtype: tensor(float) of shape (num_samples, num_shapelets)
        """
        return self.shapelets_blocks(X)

    def get_shapelets(self):
        """
        Return a matrix of all shapelets. The shapelets are ordered (ascending) according to
        the shapelet lengths and padded with NaN.
        @return: a tensor of all shapelets
        @rtype: tensor(float) with shape (in_channels, num_total_shapelets, shapelets_size_max)
        """
        return self.shapelets_blocks.get_shapelets()

    def set_shapelet_weights(self, weights):
        """
        Set the weights of all shapelets. The shapelet weights are expected to be ordered ascending according to the
        length of the shapelets. The values in the matrix for shapelets of smaller length than the maximum
        length are just ignored.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (in_channels, num_total_shapelets, shapelets_size_max)
        @return:
        @rtype: None
        """
        start = 0
        for i, (shapelets_size, num_shapelets) in enumerate(self.shapelets_size_and_len.items()):
            end = start + num_shapelets
            self.set_shapelet_weights_of_block(i, weights[start:end, :, :shapelets_size])
            start = end

    def set_shapelet_weights_of_block(self, i, weights):
        """
        Set the weights of shapelet block i.
        @param i: The index of the shapelet block
        @type i: int
        @param weights: the weights for the shapelets of block i
        @type weights: array-like(float) of shape (in_channels, num_shapelets, shapelets_size)
        @return:
        @rtype: None
        """
        self.shapelets_blocks.set_shapelet_weights_of_block(i, weights)

    def set_weights_of_shapelet(self, i, j, weights):
        """
        Set the weights of shapelet j in shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param j: the index of the shapelet in shapelet block i
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        self.shapelets_blocks.set_shapelet_weights_of_single_shapelet(i, j, weights)


class LearningShapelets:
    """
    Wraps Learning Shapelets in a sklearn kind of fashion.
    Parameters
    ----------
    shapelets_size_and_len : dict(int:int)
        The keys are the length of the shapelets and the values the number of shapelets of
        a given length, e.g. {40: 4, 80: 4} learns 4 shapelets of length 40 and 4 shapelets of
        length 80.
    loss_func : torch.nn
        the loss function
    in_channels : int
        the number of input channels of the dataset
    num_classes : int
        the number of output classes.
    dist_measure: `euclidean`, `cross-correlation`, or `cosine`
        the distance measure to use to compute the distances between the shapelets.
      and the time series.
    verbose : bool
        monitors training loss if set to true.
    to_cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size_and_len, loss_func, in_channels=1, num_classes=2,
                 dist_measure='euclidean', verbose=0, to_cuda=True, k=0, l1=0.0, l2=0.0):

        self.model = LearningShapeletsModel(shapelets_size_and_len=shapelets_size_and_len,
                                            in_channels=in_channels, num_classes=num_classes, dist_measure=dist_measure,
                                            to_cuda=to_cuda)
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.model.cuda()

        self.shapelets_size_and_len = shapelets_size_and_len
        self.loss_func = loss_func
        self.verbose = verbose
        self.optimizer = None

        if not all([k == 0, l1 == 0.0, l2 == 0.0]) and not all([k > 0, l1 > 0.0]):
            raise ValueError("For using the regularizer, the parameters 'k' and 'l1' must be greater than zero."
                             " Otherwise 'k', 'l1', and 'l2' must all be set to zero.")
        self.k = k
        self.l1 = l1
        self.l2 = l2
        self.loss_dist = ShapeletsDistanceLoss(dist_measure=dist_measure, k=k)
        self.loss_sim_block = ShapeletsSimilarityLoss()
        # add a variable to indicate if regularization shall be used, just used to make code more readable
        self.use_regularizer = True if k > 0 and l1 > 0.0 else False

    def set_optimizer(self, optimizer):
        """
        Set an optimizer for training.
        @param optimizer: a PyTorch optimizer: https://pytorch.org/docs/stable/optim.html
        @type optimizer: torch.optim
        @return:
        @rtype: None
        """
        self.optimizer = optimizer

    def set_shapelet_weights(self, weights):
        """
        Set the weights of all shapelets. The shapelet weights are expected to be ordered ascending according to the
        length of the shapelets. The values in the matrix for shapelets of smaller length than the maximum
        length are just ignored.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (in_channels, num_total_shapelets, shapelets_size_max)
        @return:
        @rtype: None
        """
        self.model.set_shapelet_weights(weights)
        if self.optimizer is not None:
            warnings.warn("Updating the model parameters requires to reinitialize the optimizer. Please reinitialize"
                          " the optimizer via set_optimizer(optim)")

    def set_shapelet_weights_of_block(self, i, weights):
        """
        Set the weights of shapelet block i.
        @param i: The index of the shapelet block
        @type i: int
        @param weights: the weights for the shapelets of block i
        @type weights: array-like(float) of shape (in_channels, num_shapelets, shapelets_size)
        @return:
        @rtype: None
        """
        self.model.set_shapelet_weights_of_block(i, weights)
        if self.optimizer is not None:
            warnings.warn("Updating the model parameters requires to reinitialize the optimizer. Please reinitialize"
                          " the optimizer via set_optimizer(optim)")

    def update(self, x, y):
        """
        Performs one gradient update step for the batch of time series and corresponding labels y.
        @param x: the batch of time series
        @type x: array-like(float) of shape (n_batch, in_channels, len_ts)
        @param y: the labels of x
        @type y: array-like(long) of shape (n_batch)
        @return: the loss for the batch
        @rtype: float
        """
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def loss_sim(self):
        """
        Get the weights of each shapelet block and calculate the cosine distance between the
        shapelets inside each block and return the summed distances as their similarity loss.
        @return: the shapelet similarity loss for the batch
        @rtype: float
        """
        blocks = [params for params in self.model.named_parameters() if 'shapelets_blocks' in params[0]]
        loss = self.loss_sim_block(blocks)
        return loss

    def update_regularized(self, x, y):
        """
        Performs one gradient update step for the batch of time series and corresponding labels y using the
        loss L_r.
        @param x: the batch of time series
        @type x: array-like(float) of shape (n_batch, in_channels, len_ts)
        @param y: the labels of x
        @type y: array-like(long) of shape (n_batch)
        @return: the three losses cross-entropy, shapelet distance, shapelet similarity for the batch
        @rtype: Tuple of float
        """
        # get cross entropy loss and compute gradients
        y_hat = self.model(x)
        loss_ce = self.loss_func(y_hat, y)
        loss_ce.backward(retain_graph=True)

        # get shapelet distance loss and compute gradients
        dists_mat = self.model(x, 'dists')
        loss_dist = self.loss_dist(dists_mat) * self.l1
        loss_dist.backward(retain_graph=True)

        if self.l2 > 0.0:
            # get shapelet similarity loss and compute gradients
            loss_sim = self.loss_sim() * self.l2
            loss_sim.backward(retain_graph=True)

        # perform gradient upgrade step
        self.optimizer.step()
        self.optimizer.zero_grad()

        return (loss_ce.item(), loss_dist.item(), loss_sim.item()) if self.l2 > 0.0 else (
        loss_ce.item(), loss_dist.item())

    def fit(self, X, Y, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        """
        Train the model.
        @param X: the time series data set
        @type X: array-like(float) of shape (n_samples, in_channels, len_ts)
        @param Y: the labels of x
        @type Y: array-like(long) of shape (n_batch)
        @param epochs: the number of epochs to train
        @type epochs: int
        @param batch_size: the batch to train with
        @type batch_size: int
        @param shuffle: Shuffle the data at every epoch
        @type shuffle: bool
        @param drop_last: Drop the last batch if X is not divisible by the batch size
        @type drop_last: bool
        @return: a list of the training losses
        @rtype: list(float)
        """
        if self.optimizer is None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")

        # convert to pytorch tensors and data set / loader for training
        if not isinstance(X, torch.Tensor):
            X = tensor(X, dtype=torch.float).contiguous()
        if not isinstance(Y, torch.Tensor):
            Y = tensor(Y, dtype=torch.long).contiguous()
        if self.to_cuda:
            X = X.cuda()
            Y = Y.cuda()

        train_ds = TensorDataset(X, Y)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

        # set model in train mode
        self.model.train()

        losses_ce = []
        losses_dist = []
        losses_sim = []
        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)
        current_loss_ce = 0
        current_loss_dist = 0
        current_loss_sim = 0
        for _ in progress_bar:
            for j, (x, y) in enumerate(train_dl):
                # check if training should be done with regularizer
                if not self.use_regularizer:
                    current_loss_ce = self.update(x, y)
                    losses_ce.append(current_loss_ce)
                else:
                    if self.l2 > 0.0:
                        current_loss_ce, current_loss_dist, current_loss_sim = self.update_regularized(x, y)
                    else:
                        current_loss_ce, current_loss_dist = self.update_regularized(x, y)
                    losses_ce.append(current_loss_ce)
                    losses_dist.append(current_loss_dist)
                    if self.l2 > 0.0:
                        losses_sim.append(current_loss_sim)
            if not self.use_regularizer:
                progress_bar.set_description(f"Loss: {current_loss_ce}")
            else:
                if self.l1 > 0.0 and self.l2 > 0.0:
                    progress_bar.set_description(f"Loss CE: {current_loss_ce}, Loss dist: {current_loss_dist}, "
                                                 f"Loss sim: {current_loss_sim}")
                else:
                    progress_bar.set_description(f"Loss CE: {current_loss_ce}, Loss dist: {current_loss_dist}")
        return losses_ce if not self.use_regularizer else (losses_ce, losses_dist, losses_sim) if self.l2 > 0.0 else (
        losses_ce, losses_dist)

    def transform(self, X):
        """
        Performs the shapelet transform with the input time series data x
        @param X: the time series data
        @type X: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: the shapelet transform of x
        @rtype: tensor(float) of shape (num_samples, num_shapelets)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        if self.to_cuda:
            X = X.cuda()

        with torch.no_grad():
            shapelet_transform = self.model.transform(X)
        return shapelet_transform.squeeze().cpu().detach().numpy()

    def fit_transform(self, X, Y, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        """
        fit() followed by transform().
        @param X: the time series data set
        @type X: array-like(float) of shape (n_samples, in_channels, len_ts)
        @param Y: the labels of x
        @type Y: array-like(long) of shape (n_batch)
        @param epochs: the number of epochs to train
        @type epochs: int
        @param batch_size: the batch to train with
        @type batch_size: int
        @param shuffle: Shuffle the data at every epoch
        @type shuffle: bool
        @param drop_last: Drop the last batch if X is not divisible by the batch size
        @type drop_last: bool
        @return: the shapelet transform of x
        @rtype: tensor(float) of shape (num_samples, num_shapelets)
        """
        self.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return self.transform(X)

    def predict(self, X, batch_size=256):
        """
        Use the model for inference.
        @param X: the time series data
        @type X: tensor(float) of shape (n_samples, in_channels, len_ts)
        @param batch_size: the batch to predict with
        @type batch_size: int
        @return: the logits for the class predictions of the model
        @rtype: array(float) of shape (num_samples, num_classes)
        """
        X = tensor(X, dtype=torch.float32)
        if self.to_cuda:
            X = X.cuda()
        ds = TensorDataset(X)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # set model in eval mode
        self.model.eval()

        """Evaluate the given data loader on the model and return predictions"""
        result = None
        with torch.no_grad():
            for x in dl:
                y_hat = self.model(x[0])
                y_hat = y_hat.cpu().detach().numpy()
                result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
        return result

    def get_shapelets(self):
        """
        Return a matrix of all shapelets. The shapelets are ordered (ascending) according to
        the shapelet lengths and padded with NaN.
        @return: an array of all shapelets
        @rtype: numpy.array(float) with shape (in_channels, num_total_shapelets, shapelets_size_max)
        """
        return self.model.get_shapelets().clone().cpu().detach().numpy()

    def get_weights_linear_layer(self):
        """
        Returns the weights for the logistic regression layer.
        Returns
        -------
        @return: a tuple containing the weights and biases
        @rtype: tuple of numpy.array(float)
        """
        return (self.model.linear.weight.data.clone().cpu().detach().numpy(),
                self.model.linear.bias.data.clone().cpu().detach().numpy())