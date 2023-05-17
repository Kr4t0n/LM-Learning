import os
import time

import torch
import torch.nn as nn
import datasets
import transformers

from dataclasses import dataclass, field
from huggingface_hub import upload_folder
from peft import get_peft_model, LoraConfig, TaskType


model_name = "THUDM/chatglm-6b"
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)


@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="data/alpaca")
    model_path: str = field(default="output")
    hf_repo_id: str = field(default="Kr4t0n/chatglm-alpaca-lora-6b")
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def data_collator(features):

    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)

    input_ids = []
    labels_list = []

    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):

        ids = feature["input_ids"]
        seq_len = feature["seq_len"]

        labels = (
            (
                [-100] * (seq_len - 1)
            ) + (
                ids[(seq_len - 1):]
            ) + (
                [-100] * (longest - ids_l)
            )
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)

        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labels))

    input_ids = torch.stack(input_ids)
    labels_list = torch.stack(labels_list)

    return {"input_ids": input_ids, "labels": labels_list}


class ModifiedTrainer(transformers.Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):

        return model(
            input_ids=inputs.get("input_ids"),
            labels=inputs.get("labels")
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):

        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        saved_params = {
            k: v.to("cpu")
            for k, v in self.model.named_parameters()
            if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def main():

    finetune_args, training_args = transformers.HfArgumentParser(
        (FinetuneArguments, transformers.TrainingArguments)
    ).parse_args_into_dataclasses()

    # init model
    model = transformers.AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=finetune_args.lora_alpha,
        lora_dropout=finetune_args.lora_dropout
    )
    model = get_peft_model(model, peft_config)

    # load dataset
    dataset = datasets.load_from_disk(finetune_args.dataset_path)
    print(f"Length of dataset: {len(dataset)}")

    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator
    )
    trainer.train()

    model.save_pretrained(training_args.output_dir)

    upload_folder(
        repo_id=finetune_args.hf_repo_id,
        folder_path=training_args.output_dir,
        commit_message=f"MAINT: end of training {time.time()}",
        ignore_patterns=["step_*", "epoch_*"],
        create_pr=True,
        token=os.environ["HUGGINGFACE_TOKEN"]
    )


if __name__ == '__main__':
    main()
