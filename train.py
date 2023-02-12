import sys
sys.path.insert(1, "/HOME1/users/students/shvl/projects/spatiotemporal")
from sen12mscrts_manager import Sen12mscrtsDatasetManager
from ctgan_dataset import CTGANTorchIterableDataset
from torch.utils.data import DataLoader

import numpy as np
from torch.serialization import save
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.CTGAN import CTGAN_Generator, CTGAN_Discriminator
import torch.nn as nn
import random
import argparse
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import *
import os
import warnings
warnings.filterwarnings("ignore")

root_dir='/LOCAL2/shvl/datasets/cloud_removal/SEN12MSCRTS'
cloud_maps_dir="/LOCAL2/shvl/datasets/cloud_removal/SEN12MSCRTS_cloud_maps"


def train(opt, model_GEN, model_DIS, optimizer_G, optimizer_D, train_loader, val_loader, device, val_step, val_n_batches):

    writer = SummaryWriter(f'runs/{opt.summary_prefix}_{opt.dataset_name}')
    
    # Define loss functions
    noise = opt.label_noise
    criterionGAN = GANLoss(opt.gan_mode, device=device)
    criterionL1 = torch.nn.L1Loss()
    criterionMSE = nn.MSELoss()

    # Use GPU
    criterionGAN = criterionGAN.to(device)
    criterionL1 = criterionL1.to(device)
    criterionMSE = criterionMSE.to(device)
    model_GEN = model_GEN.to(device)
    model_DIS = model_DIS.to(device)

    """lr_scheduler"""
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=opt.n_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=opt.n_epochs, eta_min=1e-6)
    
    """training"""
    train_step = 0

    psnr_max = 0.
    ssim_max = 0.

    print('Start training!')
    for epoch in range(opt.n_epochs):
        model_GEN.train()
        model_DIS.train()

        pbar = tqdm.tqdm(total=len(train_loader), desc="Training... ", unit=" step")
        
        lr = optimizer_G.param_groups[0]['lr']
        print('\nlearning rate = %.7f' % lr)

        L1_total = []

        # initial validation
        _, _, _ = valid(opt, model_GEN, val_loader, criterionL1, writer, train_step, val_n_batches)

        for batch in train_loader:  # real_A, real_B, _

            real_B = batch["target_image"].to(device, non_blocking=True, dtype=torch.float)

            real_A = batch["inputs"]
            real_A = [a.to(device, non_blocking=True, dtype=torch.float) for a in real_A]
            real_A_combined = torch.cat((real_A[0], real_A[1], real_A[2]), 1).to(device, non_blocking=True, dtype=torch.float)
            
            M = batch["input_cloud_maps"]
            M = [cloud_map.to(device, non_blocking=True, dtype=torch.float) for cloud_map in M]

            """forward generator"""
            fake_B, cloud_mask, aux_pred = model_GEN(real_A)

            """update Discriminator"""
            set_requires_grad(model_DIS, True)
            optimizer_D.zero_grad()

            # Fake 
            fake_AB = torch.cat((real_A_combined, fake_B), 1)
            pred_fake = model_DIS(fake_AB.detach())
            loss_D_fake = criterionGAN(pred_fake, False, noise)

            # Real
            real_AB = torch.cat((real_A_combined, real_B), 1)
            pred_real = model_DIS(real_AB)
            loss_D_real = criterionGAN(pred_real, True, noise)

            # Combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizer_D.step()

            """update generator"""
            optimizer_G.zero_grad()
            set_requires_grad(model_DIS, False)

            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((real_A_combined, fake_B), 1)
            pred_fake = model_DIS(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, True, noise)

            # Second, G(A) = B
            loss_G_L1 = criterionL1(fake_B, real_B) * opt.lambda_L1
            L1_total.append(loss_G_L1.item())

            # combine loss and calculate gradients
            loss_g_clouds = 0
            for i in range(len(cloud_mask)):
                loss_g_clouds += criterionMSE(cloud_mask[i][:, 0, :, :], M[i][:, 0, :, :])

            if opt.aux_loss:
                loss_G_aux = (criterionL1(aux_pred[0], real_B) + criterionL1(aux_pred[1], real_B) + criterionL1(aux_pred[2], real_B)) * opt.lambda_aux
                loss_G = loss_G_GAN + loss_G_L1 + loss_g_clouds + loss_G_aux
            else:
                loss_G = loss_G_GAN + loss_G_L1 + loss_g_clouds
            loss_G.backward()
            optimizer_G.step()

            writer.add_scalar('training_G_GAN', loss_G_GAN, train_step)
            writer.add_scalar('training_G_L1', loss_G_L1, train_step)
            writer.add_scalar('training_D_real', loss_D_real, train_step)
            writer.add_scalar('training_D_fake', loss_D_fake, train_step)
            writer.add_scalar('training_G_clouds', loss_g_clouds, train_step)
            if opt.aux_loss:
                writer.add_scalar('training_G_AUX_L1', loss_G_aux, train_step)
            writer.flush()

            pbar.update()
            pbar.set_postfix(
                G_GAN=f"{loss_G_GAN:.4f}",
                G_L1 = f"{loss_G_L1:.4f}",
                G_L1_total=f"{np.mean(L1_total):.4f}",
                D_real=f"{loss_D_real:.4f}",
                D_fake=f"{loss_D_fake:.4f}",
                D_clouds=f"{loss_g_clouds:.4f}"
            )
            train_step += 1

            if train_step % 1000:
                torch.save(model_GEN.state_dict(),
                           os.path.join(opt.save_model_path, opt.dataset_name, f'{opt.summary_prefix}_G_step_{train_step}.pth'))
                torch.save(model_DIS.state_dict(),
                           os.path.join(opt.save_model_path, opt.dataset_name, f'{opt.summary_prefix}_D_step_{train_step}.pth'))
                print('Save model!')
            
            if train_step % val_step == 0:
                
                psnr, ssim, total_loss = valid(opt, model_GEN, val_loader, criterionL1, writer, train_step, val_n_batches)

                if psnr_max < psnr:
                    psnr_max = psnr
                    torch.save(model_GEN.state_dict(),
                               os.path.join(opt.save_model_path, opt.dataset_name, f'{opt.summary_prefix}_G_best_PSNR_{train_step}.pth'))
                    torch.save(model_DIS.state_dict(),
                               os.path.join(opt.save_model_path, opt.dataset_name, f'{opt.summary_prefix}_D_best_PSNR_{train_step}.pth'))
                    print('Save model!')
                if ssim_max < ssim:
                    ssim_max = ssim
                    torch.save(model_GEN.state_dict(),
                               os.path.join(opt.save_model_path, opt.dataset_name, f'{opt.summary_prefix}_G_best_SSIM_{train_step}.pth'))
                    torch.save(model_DIS.state_dict(),
                               os.path.join(opt.save_model_path, opt.dataset_name, f'{opt.summary_prefix}_D_best_SSIM_{train_step}.pth'))
                    print('Save model!')
                
        pbar.close()        
      
        scheduler_D.step()
        scheduler_G.step()

    print('Best PSNR: %.3f | Best SSIM: %.3f' % (psnr_max, ssim_max))


def valid(opt, model_GEN, val_loader, criterionL1, writer, train_step, val_n_batches):
    model_GEN.eval()

    psnr_list = []
    ssim_list = []
    total_loss = 0

    pbar = tqdm.tqdm(total=val_n_batches, desc="Validating...")

    with torch.no_grad():

        for n_batch, batch in enumerate(val_loader):

            real_B = batch["target_image"].to(device, non_blocking=True, dtype=torch.float)

            real_A = batch["inputs"]
            real_A = [a.to(device, non_blocking=True, dtype=torch.float) for a in real_A]

            M = batch["input_cloud_maps"]
            M = [cloud_map.to(device, non_blocking=True, dtype=torch.float) for cloud_map in M]

            fake_B, cloud_mask, _ = model_GEN(real_A)

            loss = criterionL1(fake_B, real_B)

            for image_num in range(opt.batch_size):

                output, label = fake_B[image_num], real_B[image_num]
                psnr, ssim = psnr_ssim_cal(label, output)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

            pbar.update(1)

            if n_batch >= val_n_batches - 1:
                break

    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    psnr = np.mean(psnr_list)
    ssim = np.mean(ssim_list)

    writer.add_scalar('validation_PSNR', psnr, train_step)
    writer.add_scalar('validation_SSIM', ssim, train_step)
    writer.flush()

    # pbar.set_postfix(loss_val=f"{total_loss:.4f}", psnr=f"{psnr:.3f}", ssim=f"{ssim:.3f}")

    pbar.close()
    return psnr, ssim, total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """Path"""
    parser.add_argument("--save_model_path", type=str, default='./checkpoints', help="Path to save model")                   #
    parser.add_argument("--dataset_name", type=str, default='CTGAN_Sen2_MTC', help="name of the dataset")                   #
    parser.add_argument("--summary_prefix", type=str, default='RUN_999', help="Prefix for the tensorboard writer")
    # parser.add_argument("--predict_image_path", type=str, default='./image_out', help="Path to save predicted images")
    parser.add_argument("--load_gen", type=str, default='', help="path to the model of generator")
    parser.add_argument("--load_dis", type=str, default='', help="path to the model of discriminator")

    """Parameters"""
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--val_step", type=int, default=50, help="Validate after this number of batches")                #
    parser.add_argument("--val_n_batches", type=int, default=64, help="How many batches to use for validation")
    parser.add_argument("--gan_mode", type=str, default='lsgan', help="Which gan mode(lsgan/vanilla)")
    parser.add_argument("--optimizer", type=str, default='AdamW', help="optimizer you want to use(AdamW/SGD)")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")   
    parser.add_argument("--workers", type=int, default=0, help="number of cpu threads to use during batch generation")       #
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")                                     #
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lambda_aux', type=float, default=50.0, help='weight for aux loss')
    parser.add_argument("--in_channel", type=int, default=4, help="the number of input channels")
    parser.add_argument("--out_channel", type=int, default=4, help="the number of output channels")
    parser.add_argument("--image_size", type=int, default=256, help="crop size")
    parser.add_argument("--aux_loss", action='store_true', help="whether use auxiliary loss(1/0)")
    parser.add_argument("--label_noise", action='store_true', help="whether to add noise on the label of gan training")

    """base_options"""
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id")
    parser.add_argument("--manual_seed", type=int, default=2022, help="random_seed you want")

    opt = parser.parse_args()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    print(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
    
    os.makedirs(os.path.join(opt.save_model_path, opt.dataset_name), exist_ok=True)
    fixed_seed(opt.manual_seed)

    dataset_manager = Sen12mscrtsDatasetManager(
        root_dir=root_dir,
        cloud_maps_dir=cloud_maps_dir
    )
    dataset_manager.load_dataset()

    train_data = CTGANTorchIterableDataset(dataset_manager, device=device, mode="train")
    val_data = CTGANTorchIterableDataset(dataset_manager, device=device, mode="val")

    train_loader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        collate_fn=train_data.collate_fn,
        num_workers=opt.workers,
        prefetch_factor=1,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=opt.batch_size,
        collate_fn=val_data.collate_fn,
        num_workers=opt.workers,
        prefetch_factor=1,
        pin_memory=True,
        drop_last=True
    )
    
    print('Load CTGAN model')
    GEN = CTGAN_Generator(image_size=opt.image_size)
    DIS = CTGAN_Discriminator()
    # print(GEN, DIS)

    # Use pretrained model
    if opt.load_gen and opt.load_dis:
        print('loading pre-trained model')
        GEN.load_state_dict(torch.load(opt.load_gen))
        DIS.load_state_dict(torch.load(opt.load_dis))
    
    if opt.optimizer == 'AdamW':
        optimizer_G = torch.optim.AdamW(GEN.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=5e-4)
        optimizer_D = torch.optim.AdamW(DIS.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=5e-4)
    if opt.optimizer == 'SGD':
        optimizer_G = torch.optim.SGD(GEN.parameters(), lr=opt.lr, momentum=0.9, nesterov=True)
        optimizer_D = torch.optim.SGD(DIS.parameters(), lr=opt.lr, momentum=0.9, nesterov=True)
    
    train(opt, GEN, DIS, optimizer_G, optimizer_D, train_loader, val_loader, device, opt.val_step, opt.val_n_batches)