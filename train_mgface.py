import argparse
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
from backbones import get_model
from dataset import get_img_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolyScheduler
from partial_fc import PartialFC, PartialFCAdamW
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def create_model(args, cfg, network, training=True):
    backbone = get_model(
        network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
    backbone = backbone.cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    if training:
        backbone.train()
    else:
        backbone.eval()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()
    return backbone


def resume_model(cfg, backbone, module_fc, opt, lr_scheduler):
    start_epoch = 0
    global_step = 0
    if cfg.resume and os.path.exists(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt")):
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint
    return start_epoch, global_step


def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity,
                project = cfg.wandb_project,
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name,
                notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")

    # add by yanyu 直接以图片的形式读取数据-----------------------------
    train_loader, num_classes, num_image = get_img_dataloader(
        cfg.rec,
        cfg.train_targets,
        args.local_rank,
        cfg.batch_size,
        cfg.seed,
        cfg.num_workers,
        cfg.max_classes,
        guide_map_size=56
    )
    if num_classes > 0:
        cfg.num_classes = num_classes
    if num_image > 0:
        cfg.num_image = num_image
    # ------------------------------------------------------------

    # 主干网络
    backbone = create_model(args, cfg, cfg.network, True)

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    l_seg = nn.CrossEntropyLoss()
    # l_fusion = nn.MSELoss()

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFCAdamW(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1
    )

    # 加载模型
    start_epoch, global_step = resume_model(cfg, backbone, module_partial_fc, opt, lr_scheduler)

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec,
        summary_writer=summary_writer, wandb_logger = wandb_logger
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, cmp_img, label_img, local_labels, diff) in enumerate(train_loader):
            global_step += 1
            a, b, c = cfg.loss_list
            gamma_w = global_step / cfg.total_step

            def get_loss(local_embeddings):
                if local_embeddings is None:
                    # 先喂[口罩图|非口罩图]--------------------------------------------------------------
                    mask_pred, local_embeddings = backbone(img)
                    cls_loss_1: torch.Tensor = module_partial_fc(local_embeddings, local_labels, opt)
                    loss_seg_1 = l_seg(torch.reshape(mask_pred, [-1, 2]), torch.reshape(label_img, [-1]))
                    loss = cls_loss_1 * a + loss_seg_1 * b
                else:
                    # 再喂[非口罩图|非口罩图]---------------------------------------------------------------------------
                    mask_pred, cmp_embeddings = backbone(cmp_img)
                    cls_loss_2: torch.Tensor = module_partial_fc(cmp_embeddings, local_labels, opt)
                    coff = torch.sum(diff)
                    loss_fusion = torch.sum(torch.mean(torch.squeeze(local_embeddings - cmp_embeddings), dim=1)) / (coff + 1e-5)
                    loss_seg_2 = l_seg(torch.reshape(mask_pred, [-1, 2]), torch.reshape(label_img, [-1]))
                    loss = a * cls_loss_2 + 0 * loss_seg_2 + c * loss_fusion * gamma_w
                return loss, local_embeddings

            local_embeddings = None
            for _ in range(2):
                loss, local_embeddings = get_loss(local_embeddings)
                local_embeddings = local_embeddings.detach()

                if cfg.fp16:
                    amp.scale(loss).backward()
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                with torch.no_grad():
                    if wandb_logger:
                        wandb_logger.log({
                            'Loss/Step Loss': loss.item(),
                            'Loss/Train Loss': loss_am.avg,
                            'Process/Step': global_step,
                            'Process/Epoch': epoch
                        })
                    loss_am.update(loss.item(), 1)
                if c == 0:
                    break
            with torch.no_grad():
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)
                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)
            opt.zero_grad()
            lr_scheduler.step()

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, f"model_epoch_{epoch}.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        from torch2onnx import convert_onnx
        convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))

        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
