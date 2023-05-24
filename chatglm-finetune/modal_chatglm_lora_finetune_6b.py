import modal
import dataclasses

DATASET_DIR = "./dataset/alpaca"
REMOTE_DATASET_DIR = "/root/dataset/alpaca"
PROJECT_DIR = "."
REMOTE_PROJECT_DIR = "/root/chatglm-finetune"
REMOTE_MODEL_CACHE = "/root/.cache"
REMOTE_MODEL_DIR = "/root/chatglm-finetune-output"

stub = modal.Stub("modal-chatglm-lora-finetune-6b")
image = modal.Image.debian_slim().pip_install(
    "torch", "sentencepiece", "wandb", "cpm_kernels",
    "bitsandbytes", "accelerate",
    "transformers==4.27.1", "datasets", "peft", "gradio"
).run_commands(
    # patch libbitsandbytes replace cpu so file with cuda one
    "cd /usr/local/lib/python3.10/site-packages/bitsandbytes/ && cp libbitsandbytes_cuda117.so libbitsandbytes_cpu.so",  # noqa
)
model_cache_vol = modal.SharedVolume().persist("chatglm-cache-vol")
model_finetune_vol = modal.SharedVolume().persist("chatglm-finetune-vol")


@dataclasses.dataclass
class TrainConfig:
    per_device_train_batch_size: int = 6
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    max_steps: int = 20000
    save_steps: int = 5000
    save_total_limit: int = 2
    remove_unused_columns: bool = False

    tracker: str = "wandb"
    logging_steps: int = 100


@dataclasses.dataclass
class InferenceConfig:
    max_length: int = 150
    do_sample: bool = False
    temperature: float = .0


@stub.function(
    image=image,
    shared_volumes={
        "/root/.cache": model_cache_vol
    },
    gpu="A10G",
    timeout=86400
)
def download_chatglm_6b():

    from transformers import AutoTokenizer, AutoModel

    model_name = "THUDM/chatglm-6b"

    AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto"
    )


@stub.function(
    image=image,
    mounts=[
        modal.Mount.from_local_dir(PROJECT_DIR, remote_path=REMOTE_PROJECT_DIR),  # noqa
        modal.Mount.from_local_dir(DATASET_DIR, remote_path=REMOTE_DATASET_DIR),  # noqa
    ],
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ],
    shared_volumes={
        REMOTE_MODEL_DIR: model_finetune_vol
    },
    gpu="A10G",
    timeout=86400
)
def train_chatglm_alpaca_lora_6b():

    import subprocess

    from accelerate.utils import write_basic_config

    # set up huggingface accelerate basic config
    write_basic_config(mixed_precision="fp16")

    # set up train config
    config = TrainConfig()

    subprocess.run(
        [
            "accelerate",
            "launch",
            f"{REMOTE_PROJECT_DIR}/chatglm_lora_finetune_6b.py",
            f"--dataset_path={REMOTE_DATASET_DIR}",
            f"--output_dir={REMOTE_MODEL_DIR}",
            f"--per_device_train_batch_size={config.per_device_train_batch_size}",  # noqa
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",  # noqa
            f"--learning_rate={config.learning_rate}",
            f"--fp16=true",
            f"--max_steps={config.max_steps}",
            f"--save_steps={config.save_steps}",
            f"--save_total_limit={config.save_total_limit}",
            f"--remove_unused_columns={config.remove_unused_columns}",
            f"--report_to={config.tracker}",
            f"--logging_steps={config.logging_steps}"
        ],
        check=True
    )


@stub.function(
    image=image,
    shared_volumes={
        REMOTE_MODEL_CACHE: model_cache_vol,
        REMOTE_MODEL_DIR: model_finetune_vol
    },
    gpu="A10G",
    timeout=86400
)
def infer_chatglm_alpaca_lora_6b():
    import torch

    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModel

    # set up inference config
    config = InferenceConfig()

    model_name = "THUDM/chatglm-6b"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto"
    ).half()
    model = PeftModel.from_pretrained(
        model,
        REMOTE_MODEL_DIR
    )

    input_texts = [
        "Instruction: you have a 6-gallon cup and a 7-gallon cup, how do you measure 4 gallons of water?\nAnswer: ",  # noqa
        "Instruction: you can solve math problem\nInput: you have a 6-gallon cup and a 7-gallon cup, how do you measure 4 gallons of water?\nAnswer: ",  # noqa
        "Instruction: solve this math problem\nInput: you have a 6-gallon cup and a 7-gallon cup, how do you measure 4 gallons of water?\nAnswer: ",  # noqa
        "Instruction: solve this math problem, use step by step thinking\nInput: you have a 6-gallon cup and a 7-gallon cup, how do you measure 4 gallons of water?\nAnswer: ",  # noqa
    ]

    with torch.no_grad():

        for input_text in input_texts:
            ids = tokenizer.encode(input_text)
            input_ids = torch.LongTensor([ids]).to(model.device)

            out = model.generate(
                input_ids=input_ids,
                max_length=config.max_length,
                do_sample=config.do_sample,
                temperature=config.temperature
            )
            out_text = tokenizer.decode(out[0])
            print(out_text)


@stub.cls(
    image=image,
    shared_volumes={
        REMOTE_MODEL_CACHE: model_cache_vol,
        REMOTE_MODEL_DIR: model_finetune_vol
    },
    gpu="A10G"
)
class Model:

    def __enter__(self):

        from peft import PeftModel
        from transformers import AutoTokenizer, AutoModel

        model_name = "THUDM/chatglm-6b"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto"
        ).half()
        self.model = PeftModel.from_pretrained(
            self.model,
            REMOTE_MODEL_DIR
        )

    @modal.method()
    def infer(self, input_text, config):

        import torch

        ids = self.tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids]).to(self.model.device)

        out = self.model.generate(
            input_ids=input_ids,
            max_length=config.max_length,
            do_sample=config.do_sample,
            temperature=config.temperature
        )
        out_text = self.tokenizer.decode(out[0])

        return out_text


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

    def infer(text):
        return Model().infer.call(text, config)

    title = "ChatGLM-6B Finetuned"
    description = "ChatGLM-6B Finetuned over Alapaca Dataset using LoRA"
    examples = [
        "Instruction: you have a 6-gallon cup and a 7-gallon cup, how do you measure 4 gallons of water?\nAnswer: ",  # noqa
        "Instruction: you can solve math problem\nInput: you have a 6-gallon cup and a 7-gallon cup, how do you measure 4 gallons of water?\nAnswer: ",  # noqa
        "Instruction: solve this math problem\nInput: you have a 6-gallon cup and a 7-gallon cup, how do you measure 4 gallons of water?\nAnswer: ",  # noqa
        "Instruction: solve this math problem, use step by step thinking\nInput: you have a 6-gallon cup and a 7-gallon cup, how do you measure 4 gallons of water?\nAnswer: ",  # noqa
    ]

    interface = gr.Interface(
        fn=infer,
        inputs="text",
        outputs="text",
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
    infer_chatglm_alpaca_lora_6b.call()
