import os
import random
import math
import tempfile
import argparse
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed
import numpy as np
from dataset.my_dataset import *
import torch.optim.lr_scheduler as lr_scheduler
from model.vit_model import Transformer
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from multi_train_utils.train_eval_utils_multi import train_one_epoch, evaluate
from warmup_scheduler import GradualWarmupScheduler
from torchinfo import summary





def main(args):
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark = True  ### Automatically finding the best algorithm
    cudnn.deterministic = True
    cudnn.benchmark = False

    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # Initialize each process environment
    init_distributed_mode(args=args)

    rank = args.rank
    local_rank = torch.distributed.get_rank()
    device = torch.device(args.device, local_rank)
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size # The learning rate is multiplied by the number of parallel Gpus

    if rank == 0:  # Print the message in the first process and instantiate tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    train_image_path = args.Train_data_path  # Data path
    valid_image_path = args.Valid_data_path

    # Instantiate the training dataset
    train_data_set = Excel_dataset(train_image_path)  # load data

    # Instantiate the testing dataset
    valid_data_set = Excel_dataset(valid_image_path)  # load data

    # Assign the training sample index to the process corresponding to each rank
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valid_data_set)

    # Make a list of sample index elements per batch_size
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               shuffle= True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(valid_data_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw)
    # instantiate the model
    model = Transformer().to(device)

    # load pre-trained weights if present
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # If no pre-trained weights exist, save the weights from the first process and load the others, keeping the initial weights the same
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # Note that it is important to specify the map_location parameter, otherwise it will cause the first GPU to use more resources
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # SyncBatchNorm adopts meaning only when training networks with BN structure
    if args.syncBN:
        # Training takes more time using SyncBatchNorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # convert to DDP model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    summary(model, (batch_size, 1, 3))

    # optimizer
    # scheduler
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.005)

    # warm_up
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warm_up_epochs,
                                                            eta_min=args.lrf * args.lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=5, total_epoch=args.warm_up_epochs,
                                       after_scheduler=scheduler_cosine)

    Train_loss = []
    T_bem = []
    Valid_loss = []
    V_bem = []
    V_bem01 = []

    for epoch in range(args.epochs):
        sum_num = train_one_epoch(model=model,
                                  optimizer=optimizer,
                                  data_loader=train_loader,
                                  device=device,
                                  epoch=epoch)
        args.lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # # # validate
        sum_num1 = evaluate(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch)

        mean_loss, t_bem = sum_num
        mean_loss_V, v_bem, v_bem01 = sum_num1
        Train_loss.append(mean_loss)
        T_bem.append(t_bem)
        Valid_loss.append(mean_loss_V)
        V_bem.append(v_bem)
        V_bem01.append(v_bem01)
        # load data in tensorboard
        tags = ["loss", "valid_loss", "bem_T", "bem_V", "bem_V01", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], mean_loss_V, epoch)
        tb_writer.add_scalar(tags[2], t_bem, epoch)
        tb_writer.add_scalar(tags[3], v_bem, epoch)
        tb_writer.add_scalar(tags[4], v_bem01, epoch)
        tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)
        # save the best model
        if epoch >= 2 and v_bem >= 0.99 and v_bem01 >= 0.97:
            weight = args.save_weights_name + str(v_bem) + "  " + str(v_bem01) + ".pth"
            torch.save(model.state_dict(), weight)


    # Delete the temporary cache file
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    tb_writer.close()
    cleanup()


if __name__ == '__main__':
    # If we want to specify the number of Gpus to use, for example, the first and fourth Gpus for training, we can use the following command:
    # Enter the following code sample into the terminal
    # CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_single_gpu.py
    # CUDA_VISIBLE_DEVICES is number the graphics card        nproc_per_node is the number of graphics cards used

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--warm_up_epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lrf', type=float, default=1e-5)
    # Enable SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)

    parser.add_argument('--Train_data-path', type=str, default='')
    parser.add_argument('--Valid_data-path', type=str, default='')

    parser.add_argument('--save_weights_name', type=str, default='', help='weights name')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')

    # Do not change this parameter, it will be assigned automatically
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # The number of processes (not threads) to start. Don't set this parameter, it will be set automatically according to nproc_per_node
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)
