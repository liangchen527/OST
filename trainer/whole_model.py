import os
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
import math
import sys
sys.path.append("..")
from utils.am_softmax import AMSoftmaxLoss
from trainer.meta_xception import MetaXception
from trainer.inner_loop_optimizers import LSLRGradientDescentLearningRule
from simswap.simswap import simswap ######Please go to the simswap project site for details
model_name = 'whole-model'

class Model(nn.Module):

    def __init__(self, opt, device, logdir=None, train=True):
        super(Model, self).__init__()
        if opt is not None:
            self.meta = opt.meta
            self.opt = opt
            self.ngpu = opt.ngpu

        self.log_embedding = False  # opt.log_embedding
        self.writer = None
        self.logdir = logdir
        self.args = opt
        self.device = device
        self.model = MetaXception(inc=3, num_output_classes=2, args=self.args, device=device, direct=True).to(device=device)
        pretrained = True
        if pretrained:
            path = './xception_meta.pth'
            prt = torch.load(path, map_location='cpu')
            self.model.load_state_dict(prt, strict=False)
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,init_learning_rate=self.args.task_learning_rate,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)
        self.inner_loop_optimizer.initialise(names_weights_dict=self.get_inner_loop_parameter_dict(params=self.model.named_parameters()))
        self.cls_criterion = AMSoftmaxLoss(gamma=0., m=0.45, s=30, t=1.)
        self.train = train
        self.simswap = simswap(device)
        self.to(device)

        if train:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.9, 0.999))
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

    def define_summary_writer(self):
        if self.logdir is not None:
            # tensor board writer
            timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            log = '{}/{}/{}'.format(self.logdir, model_name, self.meta)
            log = log + '_{}'.format(timenow)
            print('TensorBoard log dir: {}'.format(log))

            self.writer = SummaryWriter(log_dir=log)

    def get_inner_loop_parameter_dict(self, params):
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=self.device)
                else:
                    if "norm_layer" not in name:
                        param_dict[name] = param.to(device=self.device)

        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model.module.zero_grad(params=names_weights_copy)
        else:
            self.model.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(), create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
        names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}
        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            else:
                names_grads_copy[key] = names_grads_copy[key].sum(dim=0)


        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     num_step=current_step_idx)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        #print(num_devices)
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}


        return names_weights_copy

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path=None):
        if model_path !=0 and os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            print('Model found in {}'.format(model_path))

    def save_ckpt(self, dataset, epoch, iters, save_dir, best=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        mid_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)

        sub_dir = os.path.join(mid_dir, self.meta)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

        subsub_dir = os.path.join(sub_dir, dataset)
        if not os.path.exists(subsub_dir):
            os.mkdir(subsub_dir)

        if best:
            ckpt_name = "epoch_{}_iter_{}_best.pth".format(epoch, iters)
        else:
            ckpt_name = "epoch_{}_iter_{}.pth".format(epoch, iters)

        save_path = os.path.join(subsub_dir, ckpt_name)

        if self.ngpu > 1:
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)

        print("Checkpoint saved to {}".format(save_path))


    def optimize(self, img, label, aug, lab_aug, o_img, o_label):
        num_step = 1
        total_loss = []
        names_weights_copy = self.get_inner_loop_parameter_dict(self.model.named_parameters())
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}
        self.model.zero_grad()
        task_losses = []
        newaug = self.simswap(img, o_img)
        if torch.rand(1) > 0.5:
            img_sup, lab_sup = torch.cat((o_img,aug),0), torch.cat((o_label,lab_aug),0)
            img_qry, lab_qry = torch.cat((img,newaug),0), torch.cat((label,lab_aug),0)
        else:
            img_sup, lab_sup = torch.cat((o_img,newaug),0), torch.cat((o_label,lab_aug),0)
            img_qry, lab_qry = torch.cat((img,aug),0), torch.cat((label,lab_aug),0)
        for step in range(num_step):
            out, _ = self.model.forward(x=img_sup, params=names_weights_copy,training=True,
                                        backup_running_statistics=True if (step == 0) else False,
                                        num_step=step)
            support_loss = self.cls_criterion(out, lab_sup).mean()
            names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                              names_weights_copy=names_weights_copy,
                                                              use_second_order=True,
                                                              current_step_idx=step)

            out_qry,_ = self.model.forward(x=img_qry, params=names_weights_copy,
                                            training=True, backup_running_statistics=True, num_step=step)
            task_loss = self.cls_criterion(out_qry, lab_qry).mean()
            task_losses.append(task_loss)

            task_losses = torch.sum(torch.stack(task_losses))
            total_loss.append(task_losses)

        total_loss = torch.mean(torch.stack(total_loss))
        if self.train:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        self.zero_grad()

        return lab_qry, (out_qry.detach(), total_loss)

    def inference(self, img, label, aug, lab_aug, o_img, o_label):
        names_weights_copy = self.get_inner_loop_parameter_dict(self.model.named_parameters())
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {name.replace('module.', ''): value.unsqueeze(0).
            repeat([num_devices] + [1 for i in range(len(value.shape))])
                              for name, value in names_weights_copy.items()}
        newaug = self.simswap(img, o_img)
        mask = torch.rand(aug.shape[0])
        mask = torch.where(mask>0.33,torch.ones_like(mask),torch.zeros_like(mask))
        mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda(non_blocking=True)
        inp = torch.cat((mask*aug + (1-mask)*newaug, o_img), 0)
        lab = torch.cat((lab_aug, o_label), 0)
        step = 1
        for i in range(step):
            score1,_ = self.model.forward(x=inp, params=names_weights_copy, training=True,
                                          backup_running_statistics=True, num_step=0)
            support_loss = self.cls_criterion(score1, lab).mean()
            names_weights_copy = self.apply_inner_loop_update(loss=support_loss, names_weights_copy=names_weights_copy,
                                                              use_second_order=True, current_step_idx=0)
        with torch.no_grad():
            score,_ = self.model.forward(x=img, params=names_weights_copy,
                                       training=False, backup_running_statistics=False, num_step=0)

            loss_cls = self.cls_criterion(score, label).mean()
        self.model.module.restore_backup_stats()
        return score, loss_cls


    def update_tensorboard(self,  loss, step, acc=None, datas=None, name='train'):
        assert self.writer
        if loss is not None:
            loss_dic = {'Cls': loss}
            self.writer.add_scalars('{}/Loss'.format(name), tag_scalar_dict=loss_dic, global_step=step)

        if acc is not None:
            self.writer.add_scalar('{}/Acc'.format(name), acc, global_step=step)

        if datas is not None:
            self.writer.add_pr_curve(name, labels=datas[:, 1].long(), predictions=datas[:, 0], global_step=step)

    def update_tensorboard_test_accs(self, accs, step, feas=None, label=None, name='test'):
        assert self.writer
        if isinstance(accs, list):
            self.writer.add_scalars('{}/ACC'.format(name), tag_scalar_dict=accs[0], global_step=step)
            self.writer.add_scalars('{}/AUC'.format(name), tag_scalar_dict=accs[1], global_step=step)
            self.writer.add_scalars('{}/EER'.format(name), tag_scalar_dict=accs[2], global_step=step)
            self.writer.add_scalars('{}/AP'.format(name), tag_scalar_dict=accs[3], global_step=step)
        else:
            self.writer.add_scalars('{}/AUC'.format(name), tag_scalar_dict=accs, global_step=step)

        if feas is not None:
            metadata = []
            mat = None
            for key in feas:
                for i in range(feas[key].size(0)):
                    lab = 'fake' if label[key][i] == 1 else 'real'
                    metadata.append('{}_{:02d}_{}'.format(key, int(i), lab))
                if mat is None:
                    mat = feas[key]
                else:
                    mat = torch.cat((mat, feas[key]), dim=0)

            self.writer.add_embedding(mat, metadata=metadata, label_img=None, global_step=step, tag='default')


