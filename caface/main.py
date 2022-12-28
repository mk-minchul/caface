import sys
import pyrootutils
import os
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
os.chdir(root)

import torch.utils.data.distributed
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning import seed_everything
import config

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def main(args, trainer_module, data_module):

    hparams = dotdict(vars(args))

    trainer_mod = trainer_module.Trainer(**hparams)
    data_mod = data_module.DataModule(**hparams)

    if hparams.seed is not None:
        seed_everything(hparams.seed)

    # create model checkpoint callback
    monitor = 'ijbb_val/0.0001'
    mode = 'max'
    save_top_k = hparams.epochs+1 if hparams.save_all_models else 1
    checkpoint_callback = ModelCheckpoint(dirpath=hparams.output_dir, save_last=True,
                                          save_top_k=save_top_k, monitor=monitor, mode=mode)

    # create logger
    os.environ['WANDB_CONSOLE'] = 'off'
    wandb_logger = WandbLogger(save_dir=hparams.output_dir, name=os.path.basename(args.output_dir), project='caface',
           tags=None if not args.wandb_tags else args.wandb_tags.split(','),
    )
    csv_logger = CSVLogger(save_dir=hparams.output_dir, name='result')
    my_loggers = [csv_logger, wandb_logger] if hparams.tpus == 0 else [csv_logger]
    resume_from_checkpoint = hparams.resume_from_checkpoint if hparams.resume_from_checkpoint else None

    trainer = pl.Trainer(resume_from_checkpoint=resume_from_checkpoint,
                        default_root_dir=hparams.output_dir,
                        logger=my_loggers,
                        gpus=hparams.gpus,
                        tpu_cores=hparams.tpus if hparams.tpus > 0 else None,
                        max_epochs=hparams.epochs,
                        strategy=hparams.distributed_backend,
                        accelerator='gpu',
                        precision=16 if hparams.use_16bit else 32,
                        fast_dev_run=hparams.fast_dev_run,
                        callbacks=[checkpoint_callback],
                        num_sanity_val_steps=0 if hparams.batch_size > 63 else 0,
                        val_check_interval=1.0 if hparams.epochs > 4 else 0.1,
                        accumulate_grad_batches=hparams.accumulate_grad_batches,
                        gradient_clip_val = hparams.gradient_clip_value, 
                        gradient_clip_algorithm = "norm" if hparams.gradient_clip_value == 0 else 'value',
                        limit_train_batches=hparams.limit_train_batches,
                        )

    if not hparams.evaluate:
        # train / val
        print('start training')
        trainer.fit(trainer_mod, data_mod)
        print('start evaluating')
        print('evaluating from ', checkpoint_callback.best_model_path)
        trainer.test(ckpt_path='best', datamodule=data_mod)
    else:
        # eval only
        print('start evaluating')
        trainer.test(trainer_mod, datamodule=data_mod)


if __name__ == '__main__':

    args, trainer_module, data_module = config.get_args()

    print(args)

    if args.distributed_backend == 'ddp' and args.gpus > 0:
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        torch.set_num_threads(1)
        args.total_batch_size = args.batch_size
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.num_workers = min(args.num_workers, 16)
    elif args.tpus > 0:
        torch.set_num_threads(1)
        args.total_batch_size = args.batch_size
        args.batch_size = int(args.batch_size / max(1, args.tpus))
        args.num_workers = min(args.num_workers, 16)

    if args.resume_from_checkpoint:
        assert args.resume_from_checkpoint.endswith('.ckpt')
        args.output_dir = os.path.dirname(args.resume_from_checkpoint)
        print('resume from {}'.format(args.output_dir))

    main(args, trainer_module, data_module)