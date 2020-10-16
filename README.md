# STGAN (CVPR 2019)

An unofficial **PyTorch**  implementation of [**STGAN: A Unified Selective Transfer Network for Arbitrary Image Attribute Editing**](https://arxiv.org/abs/1904.09709). 

## Requirements
- [Python 3.6+](https://www.python.org)
- [PyTorch 1.0+](https://pytorch.org)
- [tensorboardX 1.6+](https://github.com/lanpa/tensorboardX)
- [torchsummary](https://github.com/sksq96/pytorch-summary)
- [tqdm](https://github.com/tqdm/tqdm)
- [Pillow](https://github.com/python-pillow/Pillow)
- [easydict](https://github.com/makinacorpus/easydict)



## Preparation

Then organize the directory as:

```
├── data_root
│   └── 256px_dataset
│       ├── 000001.jpg
│       ├── 000002.jpg
│       ├── 000003.jpg
│       └── ...
│   └── attribute_dataset.txt
```

## Training

- For quickly start, you can simply use the following command to train:

  ```console
  CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config ./configs/train_stgan.yaml
  ```

- If you want to modify some hyper-parameters, please edit them in the configuration file `./configs/train_stgan.yaml` following the explanations below:
  - `exp_name`: the name of current experiment.
  - `mode`: 'train' or 'test'.
  - `cuda`: use CUDA or not.
  - `ngpu`: how many gpu cards to use. Notice: this number should be no more than the length of CUDA_VISIBLE_DEVICES list.
  - `dataset`: the name of dataset. Notice: you can extend other datasets. (material)
  - `data_root`: the root of dataset.
  - `crop_size`: the crop size of images.
  - `image_size`: the size of input images during training.
  - `g_conv_dim`: the base filter numbers of convolutional layers in G.
  - `d_conv_dim`: the base filter numbers of convolutional layers in D.
  - `d_fc_dim`: the dimmension of fully-connected layers in D.
  - `max_conv_dim`: the dimmension of fully-connected layers in D.
  - `g_layers`: the number of convolutional layers in G. Notice: same for both encoder and decoder.
  - `d_layers`: the number of convolutional layers in D.
  - `shortcut_layers`: the number of shortcut connections in G. Notice: also the number of STUs.
  - `stu_kernel_size`: the kernel size of convolutional layers in STU.
  - `use_stu`: if set to false, there will be no STU in shortcut connections.
  - `one_more_conv`: if set to true, there will be another convolutional layer between the decoder and generated image.
  - `attr_each_deconv`: if true, concatenate the attribute to each convolutional layer in decoder, otherwise, just concatenate to first layer (with the latent vector)
  - `use_attr_diff`: if true, the input to the generator is b_att-a_att (difference of attribute), otherwise it is just b_att (new attributes)
  - `attrs`: the list of all selected atrributes. Notice: please refer to `list_attr_celeba.txt` for all avaliable attributes.
  - `checkpoint`: the iteration step number of the checkpoint to be resumed. Notice: please set this to `~` if it's first time to train.
  - `batch_size`: batch size of data loader.
  - `beta1`: beta1 value of Adam optimizer.
  - `beta2`: beta2 value of Adam optimizer.
  - `g_lr`: the base learning rate of G.
  - `d_lr`: the base learning rate of D. (GAN+classifier)
  - `ld_lr`: the base learning rate of LD. (latent discriminator from FaderNet)
  - `n_critic`: number of D and LD updates per each G update.
  - `lambda_adv`: tradeoff coefficient of adversarial loss (for D and G).
  - `lambda_gp`: tradeoff coefficient of D_loss_gp (gradient penalty).
  - `lambda_d_att`: tradeoff coefficient of classification (regression) loss in D.
  - `lambda_g_att`: tradeoff coefficient of classification (regression) loss in G.
  - `lambda_g_rec`: tradeoff coefficient of reconstrution loss in G.
  - `lambda_g_latent`: tradeoff coefficient of latent disc loss in G (faderNet)
  - `lambda_ld`: tradeoff coefficient of latent disc loss in LD (faderNet)
  - `max_iters`: maximum iteration steps.
  - `lr_decay_iters`: iteration steps per learning rate decay.
  - `summary_step`: iteration steps per summary operation with tensorboardX.
  - `sample_step`: iteration steps per sampling operation.
  - `checkpoint_step`: iteration steps per checkpoint saving operation.

## Acknowledgements

This code refers to the following two projects:

[1] [TensorFlow implementation of STGAN](https://github.com/csmliu/STGAN) 

[2] [PyTorch implementation of StarGAN](https://github.com/yunjey/stargan)
