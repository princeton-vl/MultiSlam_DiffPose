import gin
import torch
from torch.utils.data import default_collate
from collections import defaultdict
from multi_slam.dpvo.lietorch import SE3

def custom_collate(items):
    for item in items:
        item['poses'] = item['poses'].data
    output = default_collate(items)
    output['poses'] = SE3(output['poses'])
    return output

@gin.configurable
class CombinedDataloader:

    def __init__(self, dataloaders, **kwargs):
        kwargs['num_workers'] = kwargs['num_workers'] // len(dataloaders)
        self.dataloader_instances = [DL(**kwargs, drop_last=True) for DL in dataloaders]
        self.combined_datalader = map(custom_collate, zip(*self.dataloader_instances))
        main_dataset_name, mode = zip(*(d.prefix.split('/') for d in self.dataloader_instances))
        self.prefix = f"{'+'.join(main_dataset_name)}/{'+'.join(mode)}"

    def reinit(self):
        for dl in self.dataloader_instances:
            dl.reinit()
        self.combined_datalader = map(custom_collate, zip(*self.dataloader_instances))

    def __next__(self):
        self.reinit()
        batches = next(self.combined_datalader)
        C, B, *_ = batches['images'].shape
        ds = torch.multinomial(torch.ones(C), B, replacement=True)
        ar = torch.arange(B)

        new_batch = defaultdict(list)
        for key in ["images", "depths", "poses", "intrinsics", "bounds"]:
            new_batch[key] = batches[key][ds, ar]

        for i, c in enumerate(ds.numpy()):
            new_batch['dataset_name'].append(batches['dataset_name'][i][c])
            new_batch['scene_id'].append(batches['scene_id'][i][c])
            new_batch['pair_id'].append(batches['pair_id'][c][i])
            new_batch['pair_names'].append([batches['pair_names'][i][j][c] for j in range(2)])

        return dict(new_batch)

    def __iter__(self):
        return self