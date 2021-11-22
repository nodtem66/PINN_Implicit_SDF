# -*- coding: utf-8 -*-

import torch

class CallbackScheduler():
    def __init__(self, callback=[], optimizer=None, model=None, eps=1e-6, patience=100):
        self._callback = callback
        self._step = 0
        self._loss = 1e6
        self._eps = eps
        self._model = model
        self._max_patience = patience
        self._countdown = patience
        self.optimizer = optimizer
        for g in self.optimizer.param_groups:
            self.lr = g['lr']

    def step_loss(self, loss):
        delta_loss = abs(self._loss - loss)
        self._loss = loss
        
        if delta_loss < self._eps:
            if self._countdown == 0:
                self._countdown = self._max_patience
                self.run_callback()
            else:
                self._countdown -= 1

    def step_when(self, condition=True):
        if condition:
            self.run_callback()
    
    def run_callback(self):
        if len(self._callback) > self._step:
            if callable(self._callback[self._step]):
                self._callback[self._step](self)
        self._step += 1

    @staticmethod
    def reduce_lr(ratio=0.1):
        def _r(self):
            for g in self.optimizer.param_groups:
                g['lr'] *= ratio
            self.lr *= ratio
        return _r

    @staticmethod
    def init_LBFGS(**vargs):
        def _init(self):
            self.optimizer = torch.optim.LBFGS(
                self._model.parameters(),
                **vargs
            )
            self.lr = vargs['lr']
        return _init
    
    def __str__(self):
        return "LR Scheduler (%d callbacks)" % len(self._callback)