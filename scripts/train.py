# -*- coding: utf-8 -*-

# Add parent directory into system path
import sys
import os

sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))

from utils import optimizer
from models import (
    Davies2021,
    MLP_PINN,
    M2,
    M2_1,
    M4,
    M4_1,
    MLP_GPINN,
    MLP_GPINN_LambdaAdaptive,
    LambdaAdaptive_G_PINN,
    LambdaAdaptive,
    G_PINN,
    PINN,
    M4_1_GPINN,
    M4_1_GPINN_RAR,
    M4_1_RAR,
    MLP_PINN_RAR,
    M2_1_RAR,
    M2_RAR,
    M4_RAR,
    MLP_GPINN_RAR,
    ResidualAdaptive,
)
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import argparse
import torch
from torch import nn
import torch_optimizer
from torch.nn.init import calculate_gain


DATASET_TYPE_SDF_FROM_STL = 0
DATASET_TYPE_SDF_FROM_NPZ = 1

MODEL_TYPE_DAVIES_2021 = 1
MODEL_TYPE_MLP_PINN = 2
MODEL_TYPE_M2 = 3
MODEL_TYPE_M2_1 = 4
MODEL_TYPE_M4 = 5
MODEL_TYPE_M4_1 = 6
MODEL_TYPE_MLP_GPINN = 7
MODEL_TYPE_MLP_GPINN_ADAPTIVE = 8
MODEL_TYPE_MLP_PINN_RAR = 9
MODEL_TYPE_M4_1_GPINN = 10
MODEL_TYPE_M4_1_RAR = 11
MODEL_TYPE_M4_1_GPINN_RAR = 12
MODEL_TYPE_MLP_GPINN_RAR = 13
MODEL_TYPE_M2_RAR = 14
MODEL_TYPE_M2_1_RAR = 15
MODEL_TYPE_M4_RAR = 16


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
    python train.py ./dataset/box_1.0_1000_1e6.stl --max_epochs 2500 --model 1 --optimizer 2
"""
    parser = argparse.ArgumentParser(
        prog="train.py",
        add_help=False,
        description="Train model to predict sdf from implicit function",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # data specific
    parser.add_argument(
        "train_data",
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
    parser.add_argument("-S", "--num_hidden_size", type=int, default=32)
    parser.add_argument("-L", "--num_layers", type=int, default=8)
    parser.add_argument("--optimizer", help="Optimizer (see the list below)", type=int, default=2)
    parser.add_argument(
        "--loss_lambda",
        nargs="+",
        help="Array of lambda",
        type=float,
        default=[1.0, 1.0, 0.01],
    )
    parser.add_argument("--sampling_method", help="", type=str, default="Importance")
    parser.add_argument("--number_sampling_points", type=int, default=int(1e4))
    parser.add_argument("--number_sampling_weight", type=int, default=10)
    parser.add_argument(
        "--number_initial_uniform_points",
        help="Used in importance sampling",
        type=int,
        default=int(1e6),
    )
    parser.add_argument(
        "--importance_weight",
        help="0: uniform sampling, inf: surface sampling",
        type=int,
        default=20,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--write_out_epochs", type=int, default=100)
    parser.add_argument("--lr_step", help="step to set new learning rate", type=int, default=0)
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.001)
    parser.add_argument("--adaptive_lambda_step", type=int, default=25)
    parser.add_argument("--adaptive_residual_points", type=int, default=int(1e4))
    parser.add_argument("--adaptive_residual_epoch", type=int, default=2000)
    parser.add_argument(
        "--log_dir",
        help="logging directory (default: ./runs",
        type=str,
        default="./runs/",
    )
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


def main(*argv):
    parser = MyParser()
    args = parser.parse_args(*argv)

    if len(args.loss_lambda) < 3:
        args.loss_lambda += [0.0, 0.0, 0.0]
        args.loss_lambda = args.loss_lambda[:3]

    if args.device is not None:
        device = args.device
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    train_data_filename, train_data_ext = os.path.splitext(os.path.basename(args.train_data))
    experiment_name = str(int(time.time()))

    # initialize TensorboardX
    writer_config = {
        "logdir": os.path.join(args.log_dir, f"model_{args.model}", os.path.basename(train_data_filename)),
        "write_to_disk": False if args.disable_log else True,
    }

    # initialize neural network model
    kwargs = {"N_layers": args.num_layers, "width": args.num_hidden_size, "activation": nn.ELU()}
    if args.model == MODEL_TYPE_DAVIES_2021:
        net = Davies2021(N_layers=args.num_layers, width=args.num_hidden_size).to(device)
    elif args.model == MODEL_TYPE_MLP_PINN:
        net = MLP_PINN(loss_lambda=args.loss_lambda, **kwargs).to(device)
    elif args.model == MODEL_TYPE_M2:
        net = M2(alpha=0.9, **kwargs).to(device)
    elif args.model == MODEL_TYPE_M2_1:
        net = M2_1(alpha=0.9, **kwargs).to(device)
    elif args.model == MODEL_TYPE_M4:
        net = M4(alpha=0.9, **kwargs).to(device)
    elif args.model == MODEL_TYPE_M4_1:
        net = M4_1(alpha=0.9, **kwargs).to(device)
    elif args.model == MODEL_TYPE_MLP_GPINN:
        net = MLP_GPINN(loss_lambda=args.loss_lambda, **kwargs).to(device)
    elif args.model == MODEL_TYPE_MLP_GPINN_ADAPTIVE:
        net = MLP_GPINN_LambdaAdaptive(**kwargs).to(device)
    elif args.model == MODEL_TYPE_MLP_PINN_RAR:
        net = MLP_PINN_RAR(**kwargs).to(device)
    elif args.model == MODEL_TYPE_M4_1_GPINN:
        net = M4_1_GPINN(**kwargs).to(device)
    elif args.model == MODEL_TYPE_M4_1_RAR:
        net = M4_1_RAR(**kwargs).to(device)
    elif args.model == MODEL_TYPE_M4_1_GPINN_RAR:
        net = M4_1_GPINN_RAR(**kwargs).to(device)
    elif args.model == MODEL_TYPE_MLP_GPINN_RAR:
        net = MLP_GPINN_RAR(**kwargs).to(device)
    elif args.model == MODEL_TYPE_M2_RAR:
        net = M2_RAR(**kwargs).to(device)
    elif args.model == MODEL_TYPE_M2_1_RAR:
        net = M2_1_RAR(**kwargs).to(device)
    elif args.model == MODEL_TYPE_M4_RAR:
        net = M4_RAR(**kwargs).to(device)
    else:
        raise ValueError("model must be 1-16")

    # Load train data
    from utils import RandomMeshSDFDataset, ImplicitDataset, TestDataset, SliceDataset

    if train_data_ext == ".stl":
        dataset_type = DATASET_TYPE_SDF_FROM_STL
        output_stl = args.train_data
        train_dataset = RandomMeshSDFDataset(
            output_stl,
            sampling_method="importance",
            N=args.number_sampling_points,
            M=args.number_initial_uniform_points,
            W=args.number_sampling_weight,
            device=device,
        )
    elif train_data_ext == "" or train_data_ext is None or train_data_ext == ".npz":
        dataset_type = DATASET_TYPE_SDF_FROM_NPZ
        output_stl = os.path.join(os.path.dirname(args.train_data), train_data_filename + ".stl")
        if train_data_ext == ".npz" and '_train' in train_data_filename:
            train_data_filename = train_data_filename.replace('_train', '')
        train_dataset = ImplicitDataset.from_file(
            file=os.path.join(os.path.dirname(args.train_data), train_data_filename + "_train.npz"), device=device
        )
    else:
        raise ValueError("train_data must be either .stl or .npz")
    if not args.quiet:
        print(train_dataset)

    # Optimizer
    from utils.optimizer import CallbackScheduler

    if args.lr_step == 0:
        args.lr_step = int(args.max_epochs / 5)
    if args.optimizer == 1:
        optimizer = torch.optim.LBFGS(
            net.parameters(),
            lr=args.lr,
            max_iter=20,
            max_eval=40,
            tolerance_grad=1e-5,
            tolerance_change=1e-9,
            history_size=100,
            line_search_fn=None,
        )
        lr_scheduler = CallbackScheduler(
            [
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
            ],
            optimizer=optimizer,
            model=net,
            eps=1e-7,
            patience=int(args.lr_step * 0.5),
        )
    elif args.optimizer == 2:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-6, amsgrad=False)
        lr_scheduler = CallbackScheduler(
            [
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.init_LBFGS(
                    lr=0.5,
                    max_iter=20,
                    max_eval=40,
                    tolerance_grad=1e-5,
                    tolerance_change=1e-9,
                    history_size=100,
                    line_search_fn=None,
                ),
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
            ],
            optimizer=optimizer,
            model=net,
            eps=1e-7,
            patience=int(args.lr_step * 0.5),
        )
    elif args.optimizer == 3:
        optimizer = torch_optimizer.AdaBound(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-6, amsbound=False)
        lr_scheduler = CallbackScheduler(
            [],
            optimizer=optimizer,
            model=net,
            eps=1e-7,
            patience=int(args.lr_step * 0.5),
        )
    elif args.optimizer == 4:
        optimizer = torch_optimizer.Yogi(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-6)
        lr_scheduler = CallbackScheduler(
            [
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
            ],
            optimizer=optimizer,
            model=net,
            eps=1e-7,
            patience=int(args.lr_step * 0.5),
        )
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-6, amsgrad=False)
        lr_scheduler = CallbackScheduler(
            [
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
                CallbackScheduler.reduce_lr(0.33),
            ],
            optimizer=optimizer,
            model=net,
            eps=1e-7,
            patience=int(args.lr_step * 0.5),
        )

    if not args.disable_log:
        writer = SummaryWriter(**writer_config)

    residual_points = train_dataset.points if dataset_type == DATASET_TYPE_SDF_FROM_STL else train_dataset.pde_points
    residual_points.requires_grad_(True)
    bc_points = train_dataset.points if dataset_type == DATASET_TYPE_SDF_FROM_STL else train_dataset.bc_points
    bc_sdfs = train_dataset.sdfs if dataset_type == DATASET_TYPE_SDF_FROM_STL else train_dataset.bc_sdfs
    try:
        for epoch in tqdm(range(args.max_epochs), disable=args.quiet):
            # Training
            optimizer.zero_grad()
            if isinstance(net, G_PINN) or isinstance(net, PINN):
                y = net(residual_points)
                loss = net.loss(
                    y,
                    residual_points,
                    bc_points,
                    bc_sdfs
                )
            else:
                loss = net.loss(bc_points, bc_sdfs)
            loss.backward(retain_graph=True)

            lr_scheduler.optimizer.step(lambda: loss)
            lr_scheduler.step_when((epoch % args.lr_step) == args.lr_step - 1)
            lr_scheduler.step_loss(loss)

            if isinstance(net, LambdaAdaptive) and (
                (epoch % args.adaptive_lambda_step) == args.adaptive_lambda_step - 1
            ):
                y = net(residual_points)
                net.adaptive_lambda(
                    y,
                    residual_points,
                    bc_points,
                    bc_sdfs,
                )

            if isinstance(net, ResidualAdaptive) and (epoch == args.adaptive_residual_epoch):
                if dataset_type == DATASET_TYPE_SDF_FROM_STL:
                    extra_points = net.adjust_samples_from_residual(
                        net(residual_points), residual_points, num_samples=args.adaptive_residual_points
                    )
                    residual_points = torch.cat((residual_points, extra_points))

            if not args.disable_log:
                if args.model == 1:
                    writer.add_scalar(
                        experiment_name + "/training_loss/loss",
                        net._loss,
                        global_step=epoch,
                    )
                elif isinstance(net, G_PINN):
                    writer.add_scalars(
                        experiment_name + "/training_loss",
                        {
                            "loss": net._loss,
                            "PDE": net._loss_PDE,
                            "SDF": net._loss_SDF,
                            "gradient_PDE": net._loss_gradient_PDE,
                        },
                        global_step=epoch,
                    )
                elif isinstance(net, PINN):
                    writer.add_scalars(
                        experiment_name + "/training_loss",
                        {"loss": net._loss, "PDE": net._loss_PDE, "SDF": net._loss_SDF},
                        global_step=epoch,
                    )
                if isinstance(net, LambdaAdaptive_G_PINN):
                    writer.add_scalars(
                        experiment_name + "/loss_lambda",
                        {
                            "lambda_1": net.loss_lambda[0],
                            "lambda_2": net.loss_lambda[1],
                            "lambda_3": net.loss_lambda[2],
                        },
                        global_step=epoch,
                    )
                elif isinstance(net, LambdaAdaptive):
                    writer.add_scalars(
                        experiment_name + "/loss_lambda",
                        {
                            "lambda_1": net.loss_lambda[0],
                            "lambda_2": net.loss_lambda[1],
                        },
                        global_step=epoch,
                    )
                writer.add_scalar(experiment_name + "/lr", lr_scheduler.lr, global_step=epoch)
                writer.add_scalar(experiment_name + "/cuda_memory", torch.cuda.memory_allocated(device))
            # endif disable_log
        # endfor epoch

        if not args.disable_log:
            experiment_info_dict = {
                "lr": args.lr,
                "lr_step": args.lr_step,
                "model": args.model,
                "max_epochs": args.max_epochs,
                "num_layers": args.num_layers,
                "num_hiddens": args.num_hidden_size,
                "sampling_method": args.sampling_method,
                "importance_weight": args.importance_weight,
                "train_data": args.train_data,
                "train_dataset": str(train_dataset),
                "optimizer": args.optimizer,
                "loss_lambda": str(args.loss_lambda),
                "adaptive_residual_points": args.adaptive_residual_points,
                "adaptive_residual_epoch": args.adaptive_residual_epoch,
            }

            if dataset_type == DATASET_TYPE_SDF_FROM_STL:
                test_dataset = RandomMeshSDFDataset(output_stl, sampling_method="uniform", N=1000000, device=device)
                error_uniform = net.test(test_dataset.points, test_dataset.sdfs)
                test_dataset = RandomMeshSDFDataset(
                    output_stl,
                    sampling_method="point",
                    N=1000000,
                    ratio=0.1,
                    device=device,
                )
                error_random_point = net.test(test_dataset.points, test_dataset.sdfs)
                error_gradient = torch.finfo(torch.float32).max
            elif dataset_type == DATASET_TYPE_SDF_FROM_NPZ:
                test_dataset = TestDataset(
                    os.path.join(os.path.dirname(args.train_data), train_data_filename + "_test.npz"), device=device
                )
                error_uniform = net.test(test_dataset.points, test_dataset.sdfs)
                error_gradient = net.test_gradient(test_dataset.points, test_dataset.gradients)
                error_random_point = net.test(test_dataset.random_points, test_dataset.random_sdfs)

            writer.add_hparams(
                experiment_info_dict,
                {
                    "hparam/loss_uniform": error_uniform,
                    "hparam/loss_random_point": error_random_point,
                    "hparam/loss_gradient": error_gradient,
                },
                name=experiment_name,
            )

            from utils import SDFVisualize

            visualize = SDFVisualize(
                z_level=0,
                step=0.05,
                offset=30,
                nums=100,
                writer=writer,
                tag=experiment_name,
            )

            if os.path.exists(output_stl):
                visualize.from_nn(net, bounds_from_mesh=output_stl, device=device)
                visualize.from_mesh(output_stl)
            elif DATASET_TYPE_SDF_FROM_NPZ:
                slice_file = os.path.join(os.path.dirname(args.train_data), train_data_filename + "_slice.npz")
                if os.path.exists(slice_file):
                    visualize.from_dataset(net, slice_file)
        # endif disable_log

    except KeyboardInterrupt as e:
        print("Bye bye")

    finally:
        if len(args.json) > 0:
            writer.export_scalars_to_json(args.json)
        if not args.disable_log:
            writer.close()
    # end try-except


if __name__ == "__main__":
    main(sys.argv[1:])
    # end main
