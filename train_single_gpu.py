import argparse
import os
import random
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset.my_dataset import *
import torch.optim.lr_scheduler as lr_scheduler
from model.vit_model import Transformer
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate
from warmup_scheduler import GradualWarmupScheduler
from torchinfo import summary


def main(args):
    seed = 1  # Random initialization ensures that the process can be replicated
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # Check if the gpu device is available
    print(args)  # Printf hyperparameters

    if args.tensorboard:  # tensorboard was used to record the training process
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()

    if os.path.exists("./weights") is False:  # Weights storage path
        os.makedirs("./weights")

    train_image_path = args.Train_data_path  # Data path
    valid_image_path = args.Valid_data_path

    # Instantiate the training dataset
    train_data_set = Excel_dataset(train_image_path)  # load data

    # Instantiate the testing dataset
    valid_data_set = Excel_dataset(valid_image_path)  # load data

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0)

    val_loader = torch.utils.data.DataLoader(valid_data_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=0)

    # Load pre-trained weights if they exist
    model = Transformer().to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            model2_dict = model.state_dict()
            print(model2_dict.items())
            state_dict = {k: v for k, v in weights_dict.items() if k in model2_dict.keys()}
            model2_dict.update(state_dict)
            load_weights_dict = {}  # 多gpu保存的模型有module前缀
            for k, v in model2_dict.items():
                new_k = k.replace('module.', '') if 'module' in k else k
                load_weights_dict[new_k] = v
            # load_weights_dict = {k: v for k, v in weights_dict.items()
            #                      if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    summary(model, (batch_size, 1, 3))


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

    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--warm_up_epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lrf', type=float, default=1e-5)
    parser.add_argument('--tensorboard', type=bool, default=True)

    parser.add_argument('--Train_data-path', type=str, default='')
    parser.add_argument('--Valid_data-path', type=str, default='')

    parser.add_argument('--save_weights_name', type=str, default='', help='weights name')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
