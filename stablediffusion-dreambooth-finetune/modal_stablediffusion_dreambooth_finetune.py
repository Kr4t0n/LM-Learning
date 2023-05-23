import modal
import dataclasses

DATASET_DIR = "./dataset/dog"
REMOTE_DATASET_DIR = "/root/dataset/dog"
REMOTE_MODEL_DIR = "/root/stablediffusion-finetune-output"


stub = modal.Stub("modal-stablediffusion-dreambooth-finetune")
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "torchvision",
        "accelerate", "transformers", "datasets",
        "ftfy", "triton",
        "wandb"
    )
    .pip_install(
        "xformers"
    )
    .apt_install(
        "git"
    )
    .run_commands(
        "cd /root && git clone https://github.com/huggingface/diffusers",
        "cd /root/diffusers && pip install -e ."
    )
)
model_finetune_vol = modal.SharedVolume().persist("stablediffusion-finetune-vol")  # noqa


@dataclasses.dataclass
class SharedConfig:
    instance_name: str = "[V]"
    class_name: str = "dog"


@dataclasses.dataclass
class TrainConfig(SharedConfig):
    prefix: str = "a photo of"
    postfix: str = ""
    validation_prefix: str = "a photo of"
    validation_postfix: str = "swimming in the pool"

    model_name = "runwayml/stable-diffusion-v1-5"

    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-6
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 2000
    checkpointing_steps: int = 1000

    tracker: str = "wandb"
    validation_steps: int = 100


@dataclasses.dataclass
class InferenceConfig(SharedConfig):
    prefix: str = "a photo of"
    postfix: str = "swimming in a pool"

    num_inference_steps: int = 50
    guidance_scale: float = 7.5


@stub.function(
    image=image,
    gpu="A100",
    mounts=[
        modal.Mount.from_local_dir(DATASET_DIR, remote_path=REMOTE_DATASET_DIR)
    ],
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ],
    shared_volumes={REMOTE_MODEL_DIR: model_finetune_vol},
    timeout=86400
)
def train_stablediffusion_dog_dreambooth():

    import subprocess

    from accelerate.utils import write_basic_config

    # set up huggingface accelerate basic config
    write_basic_config(mixed_precision="fp16")

    # set up train config
    config = TrainConfig()

    # define the training prompt
    instance_phrase = f"{config.instance_name} {config.class_name}"
    prompt = f"{config.prefix} {instance_phrase} {config.postfix}".strip()
    validation_prompt = f"{config.validation_prefix} {instance_phrase} {config.validation_postfix}"  # noqa

    # run training
    subprocess.run(
        [
            "accelerate",
            "launch",
            "/root/diffusers/examples/dreambooth/train_dreambooth.py",
            f"--pretrained_model_name_or_path={config.model_name}",
            f"--instance_data_dir={REMOTE_DATASET_DIR}",
            f"--output_dir={REMOTE_MODEL_DIR}",
            f"--instance_prompt='{prompt}'",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",  # noqa
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--report_to={config.tracker}",
            f"--validation_steps={config.validation_steps}",
            f"--validation_prompt='{validation_prompt}'"
        ],
        check=True
    )
