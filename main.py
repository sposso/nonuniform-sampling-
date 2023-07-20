# pylint: disable=missing-function-docstring
import argparse
import os
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
from utils.classifier import FullClassifier, FeatureExtractorFreezeUnfreeze
from utils.train_util import initialize_data_loader
from data.split import split_dataset
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='train, resume, test arguments')
    parser.add_argument('--project_root', default= os.getcwd(), type = str)
    parser.add_argument('--data_localization',default="/home/sposso22/work/new_data/masks/complete_data/locations.csv",type=str, help= 'data paths')
    parser.add_argument('--aug', type=str, default=None, choices=[None, "BIG","MEDIUM","SMALL"])
    parser.add_argument('--batch_size', '-b',default=8, type = int, help = "mini-batch size per worker(GPU)" )
    parser.add_argument('--workers', '-w', default= 4, type = int, help ="Number of data loading workers")
    parser.add_argument('--warp', default = True, action="store_true")
    parser.add_argument('--res', '-r', default=1, type= int, help='choose wanted resolution from list')
    parser.add_argument('--sigma', '-S', default= 14, type = int, help ="Sigma value of the Gaussian Kernel")
    parser.add_argument('--exa', '-E', default= 6, type = int, help ="Saturation of heatmaps values through the Softmax Normalization")
    #parser.add_argument("--checkpoint-file",default=os.getcwd()+"/tmp/checkpoint.pth.tar",type=str,help="checkpoint file path, to load and save to")
    parser.add_argument('--epochs','-e', default = 50, type = int, help='number of total epochs to run')
    parser.add_argument('--devices', default = 2, type = int, help='number of devices (gpus) to run on')

    parser.add_argument('--model-checkpoint-folder', type=str, default="checkpoints")
    parser.add_argument('--logs-folder', type=str, default="logs")
    parser.add_argument('--experiment_name', type=str, default="classifier")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--lambd', type= float, default = 0.5)

    return parser.parse_args()

def bookkeeping(args):
    pl.seed_everything(args.seed)

    # improves speed if your accelrator (gpu, tpu, etc) has tensor cores
    # might reduce accuracy slightly but likely nothing too extreme
    #torch.set_float32_matmul_precision('high')

def main():
    args = parse_args()
    bookkeeping(args)

    res = [(288,224),(576,448),(864,672),(1152,896)][args.res]
    
    #Initilialize data
    print(args.data_localization)  
    split_dataset(args.data_localization)
    train_loader,val_loader, test_loader = initialize_data_loader(res,args.sigma,args.exa,args.batch_size,args.workers,args.project_root,args.aug,args.lambd)

    #setup entire model
    model = FullClassifier(args, res=res)
    
    wandb.login()
    wandblogger =  pl_loggers.WandbLogger(project='breastcancer', name=args.experiment_name, log_model=False)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.logs_folder, name=args.experiment_name)

    # create checkpoint callback. configured to save models with best KNN accuracy
    checkpoint_callback = ModelCheckpoint(dirpath=args.model_checkpoint_folder+args.experiment_name, save_on_train_epoch_end=True,
            save_top_k=2, monitor="accuracy", filename='{epoch}-{train_loss:.2f}-{accuracy:.2f}',
            every_n_epochs=1, mode='max')

    # backbone finetuning callback
    finetuning_callback = FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=30)

    # create pytorch lightning trainer
    # configured to use gpu and distributed data parallel (ddp)
    trainer = pl.Trainer.from_argparse_args(args, accelerator='gpu', log_every_n_steps=10, max_epochs=args.epochs,
                                            logger=[tb_logger,wandblogger], callbacks=[checkpoint_callback, finetuning_callback], 
                                            strategy='ddp', replace_sampler_ddp  =False)

    trainer.fit(
        model,
        train_loader,
        val_loader
    )


if __name__ == '__main__':
    main()
