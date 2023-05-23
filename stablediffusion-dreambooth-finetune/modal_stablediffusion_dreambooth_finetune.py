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
        "accelerate", "transformers", "datasets", "gradio",
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
    postfix: str = "swimming in the pool"

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
    validation_prompt = f"{config.validation_prefix} {instance_phrase} {config.validation_postfix}".strip()  # noqa

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


@stub.function(
    image=image,
    gpu="A100",
    shared_volumes={REMOTE_MODEL_DIR: model_finetune_vol},
    timeout=86400
)
def infer_stablediffusion_dog_dreambooth():

    import io
    import torch

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        REMOTE_MODEL_DIR,
        torch_dtype=torch.float16
    ).to("cuda")

    # set up inference config
    config = InferenceConfig()

    # define the inference prompt
    instance_phrase = f'{config.instance_name} {config.class_name}'
    prompt = f'{config.prefix} {instance_phrase} {config.postfix}'.strip()

    image = pipe(
        prompt,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale
    ).images[0]

    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

    return image_bytes


@stub.cls(
    image=image,
    shared_volumes={REMOTE_MODEL_DIR: model_finetune_vol},
    gpu="A100"
)
class Model:

    def __enter__(self):

        import torch

        from diffusers import DDIMScheduler, StableDiffusionPipeline

        ddim = DDIMScheduler.from_pretrained(
            REMOTE_MODEL_DIR,
            subfolder="scheduler"
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            REMOTE_MODEL_DIR,
            scheduler=ddim,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        pipe.enable_xformers_memory_efficient_attention()
        self.pipe = pipe

    @modal.method()
    def infer(self, input_text, config):

        image = self.pipe(
            input_text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale
        ).images[0]

        return image


@stub.function(
    image=image,
    concurrency_limit=1
)
@modal.asgi_app()
def fastapi_app():

    import gradio as gr

    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    web_app = FastAPI()

    config = InferenceConfig()
    instance_phrase = f"{config.instance_name} {config.class_name}"

    def infer(text):
        return Model().infer.call(text, config)

    title = "StableDiffusion Dreambooth Finetuned"
    description = "StableDiffusion Finetuned over Dog Dataset using Dreambooth"
    examples = [
        f"{instance_phrase}",
        f"a painting of {instance_phrase} with a pearl earring",
        f"a photo of {instance_phrase} swimming in the pool",
        f"drawing of {instance_phrase} high quality, cartoon",
    ]

    interface = gr.Interface(
        fn=infer,
        inputs="text",
        outputs=gr.Image(shape=(512, 512)),
        title=title,
        description=description,
        examples=examples,
        allow_flagging="never"
    )

    return mount_gradio_app(
        app=web_app, blocks=interface, path="/"
    )


@stub.local_entrypoint()
def main():

    image_bytes = infer_stablediffusion_dog_dreambooth.call()

    output_path = "output.png"
    with open(output_path, "wb") as f:
        f.write(image_bytes)
