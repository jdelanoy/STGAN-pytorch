import argparse
import os
from datetime import datetime

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.materials_data_module import MaterialDataModule
from pl_systems.ign import IGN


def main(hparams):
    seed_everything(hparams.seed)

    # create test logger
    logger = TensorBoardLogger(
        name=hparams.experiment_name,
        version=hparams.experiment_version,
        save_dir=hparams.log_experiment_path,
    )

    # init lightining model
    model = IGN(hparams=hparams)

    # init trainer
    trainer = Trainer(
        logger=logger,
        callbacks=[LearningRateMonitor()],
        checkpoint_callback=ModelCheckpoint(monitor='val_loss', save_last=True),
        gpus=hparams.gpus,
        precision=16,
        progress_bar_refresh_rate=16,
        weights_summary='top',
        # overfit_batches=1
    )

    # Load datamodule
    humans_datamodule = MaterialDataModule(
        root=hparams.data_path,
        attrs=hparams.attrs,
        crop_size=hparams.crop_size,
        image_size=hparams.image_size,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
    )

    # 5 Start training
    trainer.fit(model, humans_datamodule)


if __name__ == '__main__':
    current_time = datetime.today().strftime('%Y-%m-%d')

    _root_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(_root_dir, 'logs/logs_1207')
    checkpoint_dir = os.path.join(log_dir, 'model_weights')
    experiment_name = 'IGN'
    experiment_version = 'Autoencoder-first_test'

    hparams = argparse.ArgumentParser(add_help=False)
    hparams.add_argument('--log-experiment-path', default=log_dir)
    hparams.add_argument('--model-save_path', default=checkpoint_dir)
    hparams.add_argument('--experiment_name', default=experiment_name)
    hparams.add_argument('--experiment_version', default=experiment_version)
    hparams = IGN.add_ckpt_args(hparams).parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
