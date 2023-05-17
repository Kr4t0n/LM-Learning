import modal

stub = modal.Stub("modal-chatglm-lora-finetune-6b")
image = modal.Image.debian_slim().pip_install(
    "torch", "sentencepiece", "wandb", "cpm_kernels",
    "bitsandbytes", "accelerate",
    "transformers==4.27.1", "datasets", "peft"
).run_commands(
    # patch libbitsandbytes replace cpu so file with cuda one
    "cd /usr/local/lib/python3.10/site-packages/bitsandbytes/ && cp libbitsandbytes_cuda117.so libbitsandbytes_cpu.so",  # noqa
)
model_cache_vol = modal.SharedVolume().persist("chatglm-cache-vol")
model_finetune_vol = modal.SharedVolume().persist("chatglm-finetune-vol")


@stub.function(
    image=image,
    shared_volumes={
        "/root/.cache": model_cache_vol
    },
    gpu="A10G",
    timeout=86400
)
def download():

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
        modal.Mount.from_local_dir(
            ".",
            remote_path="/root/chatglm-finetune"
        )
    ],
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ],
    shared_volumes={
        "/root/chatglm-finetune-output": model_finetune_vol
    },
    gpu="A10G",
    timeout=86400
)
def chatglm_alpaca_lora_6b():

    import subprocess

    subprocess.run(
        [
            "python",
            "/root/chatglm-finetune/chatglm_lora_finetune_6b.py",
            "--output_dir=/root/chatglm-finetune-output/6b_max_steps_20000",
            "--dataset_path=/root/chatglm-finetune/dataset/alpaca",
            "--per_device_train_batch_size=6",
            "--gradient_accumulation_steps=1",
            "--max_steps=20000",
            "--save_steps=1000",
            "--save_total_limit=2",
            "--learning_rate=1e-4",
            "--fp16=true",
            "--remove_unused_columns=false",
            "--report_to=wandb",
            "--logging_steps=10"
        ],
        check=True
    )


@stub.function(
    image=image,
    shared_volumes={
        "/root/.cache": model_cache_vol,
        "/root/chatglm-finetune-output": model_finetune_vol
    },
    gpu="A10G",
    timeout=86400
)
def chatglm_alpaca_lora_6b_infer():
    import torch

    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModel

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
        "/root/chatglm-finetune-output/6b_max_steps_20000"
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
                max_length=150,
                do_sample=False,
                temperature=0
            )
            out_text = tokenizer.decode(out[0])
            print(out_text)


@stub.local_entrypoint()
def main():
    chatglm_alpaca_lora_6b.call()
