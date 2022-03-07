# -*- coding: utf-8 -*-

# Add parent directory into system path
import sys
import os
from math import ceil

from numpy import NaN

sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))

from utils import pretty_print as pretty

pretty.init_pretty_error()

from models import (
    Davies2021,
    MLP_PINN,
)
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import argparse
import torch
from torch import nn
from pathlib import Path

MODEL_TYPE_DAVIES_2021 = 1
MODEL_TYPE_MLP_PINN = 2

def MyParser():
    epilog = """
Model Type:
    1: MLP_Davies_2021
    2: MLP_PINN
    3: M2_Wang2020
    4: M2_1_Wang2020
    5: M4_Wang2020
    6: M4_1_Wang2020
    7: MLP_GPINN
    8: M2_1_GPINN
    9: MLP_PINN_RAR
    10: M4_1_GPINN
    11: M4_1_RAR
    12: M4_1_GPINN_RAR

Optimizers:
    1: Adam
    2: Adam + L-BFGS
    3: AdaBound
    4: Yogi

Examples:
    python train.py ./dataset/box_1f0_gyroid_4pi --max_epochs 2500 --model 1 --optimizer 2
"""
    parser = argparse.ArgumentParser(
        prog="train.py",
        add_help=False,
        description="Train model to predict sdf and test",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # data specific
    parser.add_argument(
        "dataset_path",
        help="STL mesh for model 1 or npz file for other model",
        default=None,
    )
    parser.add_argument(
        "-h", "--help", action="help", help="Print help and check torch devices", default=argparse.SUPPRESS
    )
    parser.add_argument("--list_device", nargs="*", help="List torch devices", action=list_device(), default=None)
    parser.add_argument(
        "-e",
        "--max_epochs",
        type=int,
        help="epochs to run training job(s) for",
        default=3000,
    )
    parser.add_argument("-m", "--model", type=int, help="Model type (see the list below)", default=1)
    parser.add_argument("--batch_size", type=int, help="Batch size (default: 10,000)", default=10000)
    parser.add_argument("--lr_step", type=int, help="number of steps per learning rate", default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--write_out_epochs", type=int, default=100)
    parser.add_argument("--log_dir", help="logging directory (default: ./runs)", type=str, default="./runs/")
    parser.add_argument("--json", help="Export JSON log", default="")
    parser.add_argument("--disable_log", help="Disable logging", action="store_true", default=False)
    parser.add_argument("--quiet", help="Disable verbose", action="store_true", default=False)
    return parser

def list_device():
    class customAction(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            setattr(args, self.dest, values)
            if torch.cuda.is_available():
                print("\nCUDA List:")
                for i in range(torch.cuda.device_count()):
                    print(f"  {i}: {torch.cuda.get_device_name(i)}")
                print()
            sys.exit(0)

    return customAction

def assert_dataset_path(path: str) -> None:
    assert path is not None, 'Path cannot be None'
    stl_file = os.path.join(path, 'raw.stl')
    train_file = os.path.join(path, 'train.npz')
    test_file = os.path.join(path, 'test.npz')

    assert os.path.exists(stl_file), f'{stl_file} didnot exist'
    assert os.path.exists(train_file), f'{train_file} didnot exist'
    assert os.path.exists(test_file), f'{test_file} didnot exist'

def main(*argv):
    parser = MyParser()
    args = parser.parse_args(*argv)

    if args.device is not None:
        device = args.device
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    
    assert_dataset_path(args.dataset_path)

    dataset_name = Path(args.dataset_path).stem
    experiment_name = str(int(time.time()))

    if not args.quiet:
        print('Dataset name: ', end='')
        pretty.pprint(dataset_name, color=pretty.BRIGHT_GREEN)
        print('Experiment name: ', end='')
        pretty.pprint(experiment_name, color=pretty.BRIGHT_BLUE)

    # initialize TensorboardX
    logdir = os.path.join(args.log_dir, f"model_{args.model}", dataset_name)
    writer_config = {
        "logdir": logdir,
        "write_to_disk": False if args.disable_log else True,
    }

    # initialize neural network model
    nn_kwargs = {"N_layers": 8, "width": 32, "activation": nn.Softplus(32), "last_activation": nn.Softplus(30)}
    if args.model == MODEL_TYPE_DAVIES_2021:
        net = Davies2021(**nn_kwargs).to(device)
    elif args.model == MODEL_TYPE_MLP_PINN:
        net = MLP_PINN(**nn_kwargs).to(device)
    else:
        raise ValueError("model must be 1-16")

    # Load train data
    from utils.dataset_generator import ImplicitDataset, TestDataset, run_batch, batch_loader
    from utils.iou import test_iou
    
    stl_file = os.path.join(args.dataset_path, 'raw.stl')
    train_file = os.path.join(args.dataset_path, 'train.npz')
    test_file = os.path.join(args.dataset_path, 'test.npz')

    train_dataset = ImplicitDataset.from_file(file=train_file, device=device)
    if not args.quiet:
        print(train_dataset)

    from utils.callback_scheduler import CallbackScheduler

    # Optimization
    torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    optimizer=torch.optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-6, amsgrad=False)
    lr_scheduler = CallbackScheduler([
        CallbackScheduler.reduce_lr(0.5),
        CallbackScheduler.nothing(),
        CallbackScheduler.reduce_lr(0.5),
        CallbackScheduler.nothing(),
        CallbackScheduler.init_LBFGS(
            lr=0.01, max_iter=20, max_eval=40, 
            tolerance_grad=1e-5, tolerance_change=1e-9,
            history_size=100,
            line_search_fn=None
        ),
        CallbackScheduler.nothing(),
    ], optimizer=optimizer, model=net, eps=1e-7, patience=300)
    
    if not args.disable_log:
        writer = SummaryWriter(**writer_config)

    # residual_points = train_dataset.points if dataset_type == DATASET_TYPE_SDF_FROM_STL else train_dataset.pde_points
    # residual_points.requires_grad_(True)
    # bc_points = train_dataset.points if dataset_type == DATASET_TYPE_SDF_FROM_STL else train_dataset.bc_points
    # bc_sdfs = train_dataset.sdfs if dataset_type == DATASET_TYPE_SDF_FROM_STL else train_dataset.bc_sdfs
    # NUM_BATCH = int(ceil(len(train_dataset.points) / args.batch_size))
    MAX_EPOCHS = int(args.lr_step * (len(lr_scheduler)+1))
    SAVE_MODEL_EVERY_EPOCH = MAX_EPOCHS // 10
    #NUM_TRAIN_SAMPLES = len(train_dataset)

    try:
        for epoch in tqdm(range(MAX_EPOCHS), disable=args.quiet):
            
            for points, sdfs in batch_loader(train_dataset.points, train_dataset.sdfs, batch_size=args.batch_size):
                
                lr_scheduler.optimizer.zero_grad()
                loss = net.loss(points, sdfs)
                loss.backward()
                lr_scheduler.optimizer.step(lambda: loss)
            
            lr_scheduler.step_when((epoch % args.lr_step) == args.lr_step - 1, verbose=False)

            if not args.disable_log:
                keys = [
                    "_loss",
                    "_loss_SDF",
                    "_loss_residual",
                    "_loss_residual_constraint",
                    "_loss_normal",
                    "_loss_cosine_similarity",
                ]

                _loss_info = {}
                for key in keys:
                    if hasattr(net, key):
                        _loss_info[key] = getattr(net, key)
                    if _loss_info[key] is NaN:
                        raise ValueError('Loss is Nan')
                
                if len(_loss_info) > 0:
                    writer.add_scalars(experiment_name + "/training_loss", _loss_info, global_step=epoch)
                writer.add_scalar(experiment_name + "/lr", lr_scheduler.lr, global_step=epoch)
                #writer.add_scalar(experiment_name + "/cuda_memory", torch.cuda.memory_allocated(device))

                if epoch > 0 and (epoch % SAVE_MODEL_EVERY_EPOCH == 0):
                    torch.save(net.state_dict(), os.path.join(logdir, experiment_name, f'{epoch}.pth'))
            # endif disable_log
        # endfor epoch
        torch.save(net.state_dict(), os.path.join(logdir, experiment_name, f'{MAX_EPOCHS}.pth'))

        # Evaluation after training
        test_dataset = TestDataset(test_file, device=device)
        test_kwarg = {'reducer': torch.mean, 'batch_size': args.batch_size}

        if not args.disable_log:
            experiment_info_dict = {
                "model": args.model,
                "lr_step": args.lr_step,
                "max_epochs": MAX_EPOCHS,
                "dataset_name": dataset_name,
                "experiment_name": experiment_name,
                #"lr": args.lr,
                #"num_layers": args.num_layers,
                #"num_hiddens": args.num_hidden_size,
                #"sampling_method": args.sampling_method,
                #"importance_weight": args.importance_weight,
                #"train_dataset": str(train_dataset),
                #"optimizer": args.optimizer,
                #"loss_lambda": str(args.loss_lambda),
                #"adaptive_residual_points": args.adaptive_residual_points,
                #"adaptive_residual_epoch": args.adaptive_residual_epoch,
            }

        writer.add_hparams(
            experiment_info_dict,
            {
                "hparam/loss_uniform_sdf": run_batch(net.test, test_dataset.uniform.points, test_dataset.uniform.sdfs, **test_kwarg).cpu().detach().numpy(),
                "hparam/loss_uniform_residual": run_batch(net.test_residual, test_dataset.uniform.points, **test_kwarg).cpu().detach().numpy(),
                "hparam/loss_uniform_norm_grads": run_batch(net.test_norm_gradient, test_dataset.uniform.points, test_dataset.uniform.norm_grads, **test_kwarg).cpu().detach().numpy(),
                "hparam/loss_random_sdf": run_batch(net.test, test_dataset.random.points, test_dataset.random.sdfs, **test_kwarg).cpu().detach().numpy(),
                "hparam/loss_random_residual": run_batch(net.test_residual, test_dataset.random.points, **test_kwarg).cpu().detach().numpy(),
                "hparam/iou": test_iou(net, test_dataset.uniform.points, test_dataset.uniform.sdfs, batch_size=args.batch_size).cpu().detach().numpy(),
            },
            name=experiment_name,
        )
        # End evaluation

    except KeyboardInterrupt as e:
        print('Bye bye')

    finally:
        if len(args.json) > 0:
            writer.export_scalars_to_json(args.json)
        if not args.disable_log:
            writer.close()
    #end try-except


if __name__ == "__main__":
    main(sys.argv[1:])
    # end main
