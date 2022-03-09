# -*- coding: utf-8 -*-

import torch

from .residual_sampler import ResidualSampler

class CallbackScheduler():
    def __init__(
            self, callback=[],
            optimizer:torch.optim.Optimizer=None,
            model:torch.nn.Module=None,
            residual_sampler:ResidualSampler=None,
            residual_points:torch.Tensor=None,
            eps=1e-6, patience:int=100
        ):
        # variables used in run_callback
        self._callback = callback
        self._step = 0
        
        # variables used in step_loss
        self._eps = eps
        self._loss = 1e6
        self._max_patience = patience
        self._countdown = patience
        
        # local access to model, and sampler
        self._model = model
        self._residual_sampler = residual_sampler
        
        # public access
        self.residual_points = residual_points
        self.optimizer = optimizer
        for g in self.optimizer.param_groups:
            self.lr = g['lr']
    
    def __len__(self):
        if self._callback:
            return len(self._callback)
        return 0

    def step_loss(self, loss):
        delta_loss = abs(self._loss - loss)
        self._loss = loss
        
        if delta_loss < self._eps:
            if self._countdown == 0:
                self._countdown = self._max_patience
                self.run_callback()
            else:
                self._countdown -= 1

    def step_when(self, condition=True, verbose=False):
        if condition:
            self.run_callback(verbose=verbose)
    
    def run_callback(self, verbose=False):
        if len(self._callback) > self._step:
            if callable(self._callback[self._step]):
                if verbose:
                    print('[callback scheduler]: ', self._callback[self._step].__code__.co_name)
                self._callback[self._step](self)
        self._step += 1

    def set_lr(self, lr=1.0):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.lr = lr

    def LBFGS(self, **vargs):
        self.optimizer = torch.optim.LBFGS(
            self._model.parameters(),
            **vargs
        )
        self.lr = vargs['lr']

    def ADAM(self, **vargs):
        self.optimizer = torch.optim.Adam(
            self._model.parameters(),
            **vargs
        )
        self.lr = vargs['lr']

    @staticmethod
    def reduce_lr(ratio=0.1):
        def reduce_learning_rate(self):
            for g in self.optimizer.param_groups:
                g['lr'] *= ratio
            self.lr *= ratio
        return reduce_learning_rate

    @staticmethod
    def init_LBFGS(**vargs):
        def init_LBFGS_optimizer(self):
            self.optimizer = torch.optim.LBFGS(
                self._model.parameters(),
                **vargs
            )
            self.lr = vargs['lr']
        return init_LBFGS_optimizer

    @staticmethod
    def nothing():
        def do_nothing(self):
            pass
        return do_nothing

    @staticmethod
    def adaptive_residual_sampling(num_points=10000, expand_scale_ratio=None):
        # used in residual-based adaptive refinement (RAR)
        def residual_sampling(self):
            if not hasattr(self, '_residual_scale_offset'):
                self._residual_scale_offset = 1.0
            if expand_scale_ratio is not None:
                self._residual_scale_offset *= expand_scale_ratio
                self._residual_sampler.expand_bounds(scale_offset=self._residual_scale_offset)

            if self._residual_sampler is not None:
                self.residual_points = self._residual_sampler.append_random_totensor(source=self.residual_points, n=num_points)
        return residual_sampling

    def __str__(self):
        return "LR Scheduler (%d callbacks)" % len(self._callback)
