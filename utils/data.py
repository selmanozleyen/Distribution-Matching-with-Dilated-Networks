from torch.utils.data import Subset
import torch


class ValSubset(Subset):
    def __init__(self, inner_: Subset) -> None:
        self.inner_ = inner_

    def __getitem__(self, idx):
        temp = self.inner_.dataset.method
        self.inner_.dataset.method = 'val'
        res = self.inner_.dataset[self.inner_.indices[idx]]
        self.inner_.dataset.method = temp
        return res

    def __len__(self):
        return len(self.inner_.indices)


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    gt_discretes = torch.stack(transposed_batch[3], 0)
    return images, points, gt_discretes
