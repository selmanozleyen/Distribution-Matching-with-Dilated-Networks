import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime
from utils.data import ValSubset
from torch.utils.data import random_split
from random import random
from models.ddm import ddm
from losses.ot_loss import OT_Loss
from utils.pytorch_utils import Save_Handle, AverageMeter
from torch.utils.tensorboard import SummaryWriter


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    st_sizes = torch.FloatTensor(transposed_batch[2])
    gt_discretes = torch.stack(transposed_batch[3], 0)
    return images, points, st_sizes, gt_discretes


class Trainer(object):
    def __init__(self, args, datargs):
        self.train_args = args
        self.datargs = datargs

    def setup(self):
        train_args = self.train_args
        datargs = self.datargs
        sub_dir = 'input-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}'.format(
            train_args['crop_size'], train_args['wot'],
            train_args['wtv'], train_args['reg'], train_args['num_of_iter_in_ot'],
            train_args['norm_cood'])

        time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        self.save_dir = os.path.join(train_args['out_path'], 'ckpts',
                                     train_args['conf_name'], train_args['dataset'], sub_dir, time_str)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        log_dir = os.path.join(train_args['out_path'], 'runs', train_args['dataset'], train_args['conf_name'],
                               time_str)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # TODO: Verify args
        self.logger = SummaryWriter(log_dir)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
        else:
            raise Exception("Gpu is not available")

        dataset_name = train_args['dataset'].lower()
        if dataset_name == 'qnrf':
            from datasets.crowd import Crowd_qnrf as Crowd
        elif dataset_name == 'nwpu':
            from datasets.crowd import Crowd_nwpu as Crowd
        elif dataset_name == 'sha' or dataset_name == 'shb':
            from datasets.crowd import Crowd_sh as Crowd
        elif dataset_name[:3] == 'ucf':
            from datasets.crowd import Crowd_ucf as Crowd
        else:
            raise NotImplementedError
        if dataset_name == 'sha' or dataset_name == 'shb':
            downsample_ratio = train_args['downsample_ratio']
            train_val = Crowd(os.path.join(datargs['data_path'],
                                           datargs["train_path"]),
                              crop_size=train_args['crop_size'],
                              downsample_ratio=downsample_ratio, method='train')
            if dataset_name == 'sha':
                train_set, val = random_split(train_val, [280, 20], generator=torch.Generator().manual_seed(42))
                val_set = ValSubset(val)
            else:
                train_set, val = random_split(train_val, [380, 20], generator=torch.Generator().manual_seed(42))
                val_set = ValSubset(val)
            self.datasets = {
                'train': train_set,
                'val': val_set
            }
        else:
            downsample_ratio = train_args['downsample_ratio']
            self.datasets = {
                'train': Crowd(os.path.join(datargs['data_path'],
                                            datargs["train_path"]),
                               crop_size=train_args['crop_size'],
                               downsample_ratio=downsample_ratio, method='train'),
                'val': Crowd(os.path.join(datargs['data_path'],
                                          datargs["val_path"]),
                             crop_size=train_args['crop_size'],
                             downsample_ratio=downsample_ratio, method='val')}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(train_args['batch_size']
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=train_args['num_workers'] * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.model = ddm(map_location=self.device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_args['lr'],
                                    weight_decay=train_args['weight_decay'], amsgrad=False)
        self.start_epoch = 0
        self.ot_loss = OT_Loss(train_args['crop_size'], downsample_ratio,
                               train_args['norm_cood'], self.device, self.logger, train_args['num_of_iter_in_ot'],
                               train_args['reg'])
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0
        if train_args['resume']:
            self.logger.add_text('log/train', 'loading pretrained model from ' + train_args['resume'], 0)
            suf = train_args['resume'].rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(train_args['resume'], self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_count = checkpoint['best_count']
                self.best_mae = checkpoint['best_mae']
                self.best_mse = checkpoint['best_mse']
                print(self.best_mae, self.best_mse, self.best_count)
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(train_args['resume'], self.device))
        else:
            self.logger.add_text('log/train', 'random initialization', 0)
        img_cnts = {'val_image_count': len(self.dataloaders['val']),
                    'train_image_count': len(self.dataloaders['train'])}
        self.logger.add_hparams({**self.train_args, **img_cnts},
                                {'best_mse': np.inf, 'best_mae': np.inf,
                                 'best_count': 0}, run_name='hparams')

    def train(self):
        """training process"""
        train_args = self.train_args
        for epoch in range(self.start_epoch, train_args['max_epoch'] + 1):
            print('log/train', '-' * 5 + 'Epoch {}/{}'.format(epoch, train_args['max_epoch']) + '-' * 5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % train_args['val_epoch'] == 0 and epoch >= train_args['val_start']:
                self.val_epoch()

    def train_eopch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        for step, (inputs, points, st_sizes, gt_discrete) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)
            wot = self.train_args['wot']
            wtv = self.train_args['wtv']
            drop = random() >= 0.5
            with torch.set_grad_enabled(True):
                outputs, outputs_normed = self.model(inputs)
                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points)
                ot_loss = ot_loss * wot
                ot_obj_value = ot_obj_value * wot
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)

                # Compute counting loss.
                count_loss = self.mae(outputs.sum(1).sum(1).sum(1),
                                      torch.from_numpy(gd_count).float().to(self.device))
                epoch_count_loss.update(count_loss.item(), N)

                # Compute TV loss.
                gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(
                    2).unsqueeze(3)
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(
                    1) * torch.from_numpy(gd_count).float().to(self.device)).mean(0) * wtv
                epoch_tv_loss.update(tv_loss.item(), N)

                loss = ot_loss + count_loss + tv_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)
                if drop:
                    self.model.dl1.train()
                    self.model.dl1a.train()
                    self.model.dl2.train()
                    self.model.dl2a.train()
                    self.model.dl3.train()
                    self.model.dl3a.train()
        mae = epoch_mae.get_avg()
        mse = np.sqrt(epoch_mse.get_avg())
        self.logger.add_scalar('loss/train', epoch_loss.get_avg(), self.epoch)
        self.logger.add_scalar('mse/train', mse, self.epoch)
        self.logger.add_scalar('mae/train', mae, self.epoch)
        self.logger.add_scalar('ot_loss/train', epoch_ot_loss.get_avg(), self.epoch)
        self.logger.add_scalar('wd/train', epoch_wd.get_avg(), self.epoch)
        self.logger.add_scalar('ot_obj_val/train', epoch_ot_obj_value.get_avg(), self.epoch)
        self.logger.add_scalar('count_loss/train', epoch_count_loss.get_avg(), self.epoch)
        self.logger.add_scalar('tv_loss/train', epoch_tv_loss.get_avg(), self.epoch)
        self.logger.add_scalar('time_cost/train', time.time() - epoch_start, self.epoch)
        print(
            'log/train',
            'Epoch {} Train, Loss: {:.2f}, OT Loss: {:.2e}, Wass Distance: {:.2f}, OT obj value: {:.2f}, '
            'Count Loss: {:.2f}, TV Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
            .format(self.epoch, epoch_loss.get_avg(), epoch_ot_loss.get_avg(), epoch_wd.get_avg(),
                    epoch_ot_obj_value.get_avg(), epoch_count_loss.get_avg(), epoch_tv_loss.get_avg(),
                    np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                    time.time() - epoch_start), self.epoch)
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, str(self.epoch)+'_ckpt.tar')
        # TODO: Reset best counts option

        torch.save({
            'epoch': self.epoch,
            'best_mae': self.best_mae,
            'best_mse': self.best_mse,
            'best_count': self.best_count,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs, _ = self.model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.logger.add_scalar('mse/val', mse, self.epoch)
        self.logger.add_scalar('mae/val', mae, self.epoch)
        self.logger.add_scalar('time_cost/val', time.time() - epoch_start, self.epoch)
        print('log/val', 'Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
              .format(self.epoch, mse, mae, time.time() - epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            filename = 'best_model_{:.2f}_{:.2f}_{}.pth'.format(self.best_mse, self.best_mae, self.best_count)
            txt = "save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse, self.best_mae, self.epoch)
            print(txt)
            self.logger.add_text('log/val',
                                 txt,
                                 self.best_count)
            best_metrics = {'best_mse': mse, 'best_mae': mae, 'best_count': self.best_count}
            for k, v in best_metrics.items():
                self.logger.add_scalar(k+'/val', v, self.epoch)
            self.logger.add_hparams({},
                                    {'best_mse': mse, 'best_mae': mae,
                                     'best_count': self.best_count}, run_name='hparams')
            torch.save(model_state_dic, os.path.join(self.save_dir, filename))
            self.best_count += 1
        elif mse < self.best_mse or mae < self.best_mae:
            filename = 'best_model_{}_{}.pth'.format(mse, mae)
            txt = "save best mse {:.2f} mae {:.2f} model epoch {}**".format(mse, mae, self.epoch)
            print(txt)

            torch.save(model_state_dic, os.path.join(self.save_dir, filename))
