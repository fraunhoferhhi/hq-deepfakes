# Deepfake Package 1.3.0
This package, written by Arian, can be used to train faceswapping autoencoders and perform faceswaps on input videos. The user should be familiar with at least pytorch lightning, in the optimal case with lightning cli, since the code is structured according to these frameworks. The code also uses wandb for logging. Moreover, this package does not include the code needed to generate data needed for training these models; for this please refer to https://vigitlab.fe.hhi.de/git/faceex_arian.

## Setup
The package was developed on a Ubuntu 20.04 machine with an RTX 3090 and CUDA 11.7. I am sure that the package will also run older / newer systems, some of the package versions should be adjusted however. For the installation we create a conda environment and install the necessary dependencies.

```bash
conda create --name myenv python=3.7.16
conda activate myenv
```

Now we clone the repo and install some requirements.

```bash
git clone git@vigitlab.fe.hhi.de:git/deepfake_arian.git
cd deepfake_arian
```
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

Before we can finalize the installation we need to install the faceex package from here: https://vigitlab.fe.hhi.de/git/faceex_arian. Just follow the instructions given there. Note: The ibug packages are not required to run this codebase here. Finalize your installation with.
```bash
pip install deepfake_arian
```

Note: It is not mandatory to install the codebase as a package even though it is recommended for cleaner use.

## Verifying your installation
To test if everything went as planned, simply navigate into the tests directory and execute the pytest script.

```bash
cd deepfake_arian/tests/
pytest test_all.py
```

If all tests go through, your installation is complete. Note: The tests only perform dummy experiments, the models trained and videos generated in that process are not capable of performing a successfull faceswap. Tip: Have a look at test_all.py to get a better understanding of how to use this package to train a model and use it to convert videos.

## Data Preparation
To prepare the data, please refer to https://vigitlab.fe.hhi.de/git/faceex_arian. In the end, you should have two directories containing the facesets of source and target persons. On top of that, `MaskExtractor` sould be used to generate a `masks.fsa` per directory/person. Note: Save the images in your faceset in 512px resolution.

## Training
Note that this repository does not include the code to start the training, as it merely provides the necessary classes or modules. For inspiration on how to structure and setup your training have a look at the `tests` directory in this repo. To train our autoencoder we mainly use the two lightningmodules contained in `training/deepfake_arian.py`. 

### `DeepfakeModule`
The model is built in `DeepfakeModule`, this module also contains the training loop code. The module expects the following arguments:

#### `cfg`
The name of the configuration, which is used to intialize the autoencoder model inside of the module. Currently only `EB4_RDA` is supported, which is the default configuration of the autoencoder in https://dl.acm.org/doi/abs/10.1145/3577163.3595106. To tweak the configuration, have a look at `training/model.py` and `training/utils.py`.

#### `lr`
The learning rate of the adam optimizer, default `5e-5`.

#### `eps`
Epsilon parameter of the adam optimizer, default `1e-7`.

#### `l2_weight`
Weight of l2 loss in the reconstruction term of the total loss.

#### `l1_weight`
Weight of l1 loss in the reconstruction term of the total loss.

#### `ffl_weight`
Weight of focal frequency loss in the total loss.

#### `stop warping`
Should be equal to approx. 50 percent of the amount of training steps. (Warping the input faces for the first half of the training helps to model to perform swaps between the identities).

#### `image_logging_interval`
How often to log images of input and output images to wandb.

### `DeepfakeDatamodule`
The data is provided by `DeepfakeDatamodule`, which takes care of building pytorch datasets and putting them into respective loaders. It can be equipped with the following arguments.

#### `batch_size`
Batch size of training. Will be split in half and distributed over two dataloaders (one for each person).

#### `num_workers`
Num workers of training, will also be split onto two loaders.

#### `path_a`, `path_b`
Path to the directory that holds the faceset of person A.

#### `input_size`
Size in pixels of the stored images, default 512.

#### `model_img_size`
Size in pixels of model output, default 256.

#### `coverage_ratio`
How many percent of the frame center to crop before resizing, default 0.8.

#### `no_flip`
Set true if you want to disable flipping in training, default False.


## Conversion
To swap the face of a person in a video, we first need to obtain some information from the respective video. This can be done with `conversion/videoprepper.py`. It contains a class, `VideoPrepper`, that extracts landmarks as well as masks from a given video, which will be used later in the conversion process. We separate these parts of the conversion process to avoid re-computing this information everytime we want to convert the same video. Call `process_dir` if you want to process an entire directory of videos, or `process_video` to process a single video. Note that only `mp4` and `avi` are supported.

The actual conversion of the video is handled by the `Converter` class in `conversion/conversion.py`. To initatie it you have to provide the following args.

#### `model_ckpt`
Path to the ckpt of the fully trained autoencoder.

#### `model_config`
Path to the config that contains the information used to initalize the training, this will be generated by lightning, so make sure to find out where its stored. Alternatively, you can find an example config in `tests`.

#### `batch_size`
Batch size of conversion process, set according to your GPU / VRAM.

#### `pad`
How much padding (in pixels) to add to the masks before blending fake face and background (using padding leads to less artifacts). Can be either a list of four values for top, bottom, left, right or a single value for all sides. Default 30.

#### `adjust_color`
Whether to adjust the mean color in the face, based on the input face. Not recommended to use, as poisson blending does this more or less automatically.

#### `writer`
Choose one of mp4, png, mp4-crf14. First one saves the output as mp4, second one as pngs, last one as mp4 with compression rate 14. Default mp4.

#### `device`
cuda or cpu.

#### `verbose`
Set to True if you want more logging throughout the process. Default False.

### Running the `Converter`
To run the converter, simply call its `process_video` method. It requires the following args:

#### `video_path`
Path to the video that is to be converted.

#### `direction`
Wheter to swap from face A to face B or vice versa. Set according to your model and video input.

#### `prep_path`
Path to the output of the videoprepper for the given `video_path`. If not specified it will just look next to the video.

#### `out_dir`
Where to save the output. If not specified, it will be saved inside a `fakes` dir in the parent dir of the input video.