import os
import pytest
import shutil
from pathlib import Path

@pytest.fixture
def paths():
    file_path = os.path.abspath(os.path.dirname(__file__))
    images_path = os.path.join(file_path, "images")
    videos_path = os.path.join(file_path, "videos")
    config_path = os.path.join(file_path, "configs")
    outputs_path = os.path.join(file_path, "outputs")

    return images_path, videos_path, config_path, outputs_path

@pytest.fixture
def successful_path():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "successful.txt")

@pytest.fixture
def successful(successful_path):
    if os.path.exists(successful_path):
        with open(successful_path, "r") as f:
            successful = f.read().splitlines()
    else:
        successful = []
    return successful


def test_import():
    try:
        import hq_deepfakes
    except Exception as e:
        pytest.fail(f"Import of package failed! Error {e}")

def test_preparation(paths):
    import hq_deepfakes as df
    _, videos_path, _, outputs_path = paths

    outputs_path = os.path.join(outputs_path, "preparation", "faces")
    ex = df.preparation.FaceExtractor(every_nth_frame=10)
    ex.process(videos_path, outputs_path)

    mx = df.preparation.MasksExtractor()
    mx.process(data_path=outputs_path)


def test_training(paths):
    import hq_deepfakes as df
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers.logger import DummyLogger
    import yaml

    images_path, videos_path, _, outputs_path = paths

    model_cfg = {
        "cfg": "EB4_RDA",
        "lr": 5e-5,
        "eps": 1e-7,
        "l2_weight": 1,
        "l1_weight": 0,
        "ffl_weight": 0,
        "stop_warping": 5,
        "image_logging_interval": -1
    }

    data_cfg = {
        "batch_size": 2,
        "num_workers": 2,
        "path_a": os.path.join(images_path, "A"),
        "path_b": os.path.join(images_path, "B"),
        "input_size": 512,
        "model_img_size": 256,
        "coverage_ratio": 0.8,
        "no_flip": False
    }

    trainer_cfg = {
        "accelerator": "gpu", 
        "devices": -1,
        "precision": 32,
        "max_epochs": 1,
        "max_steps": 10,
        "log_every_n_steps": 1,
        "enable_progress_bar": True,
        "default_root_dir": os.path.join(outputs_path, "training")
    }
    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_last=True)
    logger = DummyLogger()
    trainer_cfg["callbacks"] = [checkpoint_callback]
    trainer_cfg["logger"] = logger

    model = df.training.DeepfakeModel(**model_cfg)
    data = df.training.DeepfakeDatamodule(**data_cfg)
    trainer = pl.Trainer(**trainer_cfg)

    trainer.fit(model, datamodule=data)

    # remove some unnecessary stuff
    # shutil.rmtree(os.path.join(outputs_path, "training", "lightning_logs"), ignore_errors=True)
    ckpt_path = Path(os.path.join(outputs_path, "training", "checkpoints"))
    for path in ckpt_path.glob("epoch*"):
        path.unlink()

def test_videoprep(paths):
    _, videos_path, _, _ = paths

    from hq_deepfakes.conversion import VideoPrepper
    vp = VideoPrepper(device="cuda", batch_size=8, verbose=True)
    vp.process_dir(videos_path)

def test_conversion(paths):
    _, videos_path, configs_path, outputs_path = paths
    
    from hq_deepfakes.conversion import Converter
    converter = Converter(
        model_ckpt=os.path.join(outputs_path, "training", "checkpoints", "last.ckpt"),
        model_config=os.path.join(configs_path, "all.yml")
    )

    converter.process_video(
        video_path=os.path.join(videos_path, "27__outside_talking_still_laughing.mp4"),
        prep_path=os.path.join(videos_path, "27__outside_talking_still_laughing_prep.fsa"),
        out_dir=os.path.join(outputs_path, "conversion"),
        direction="A"
    )
