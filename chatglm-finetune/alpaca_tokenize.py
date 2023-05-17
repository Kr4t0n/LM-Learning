import json
import argparse
import datasets
import transformers

from tqdm.auto import tqdm

model_name = "THUDM/chatglm-6b"


def tokenize(tokenizer, example, eos_token_id, max_seq_len):

    prompt = example.get("context")
    target = example.get("target")

    prompt_ids = tokenizer.encode(
        prompt,
        max_length=max_seq_len,
        truncation=True
    )
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_len,
        truncation=True,
        add_special_tokens=False
    )

    input_ids = prompt_ids + target_ids + [eos_token_id]

    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def tokenize_jsonl(jsonl_path, max_seq_len, skip_overlen=False):

    print(max_seq_len)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    config = transformers.AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto"
    )

    with open(jsonl_path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = tokenize(
                tokenizer, example,
                eos_token_id=config.eos_token_id,
                max_seq_len=max_seq_len
            )

            if skip_overlen and len(feature["input_ids"]) > max_seq_len:
                continue

            feature["input_ids"] = feature["input_ids"][:max_seq_len]
            yield feature


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl_path",
        type=str, default="dataset/alpaca_data.jsonl"
    )
    parser.add_argument(
        "--save_path",
        type=str, default="dataset/alpaca"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int, default=384
    )
    parser.add_argument(
        "--skip_overlen",
        type=bool, default=False
    )
    args = parser.parse_args()

    dataset = datasets.Dataset.from_generator(
        lambda: tokenize_jsonl(
            args.jsonl_path,
            max_seq_len=args.max_seq_len,
            skip_overlen=args.skip_overlen
        )
    )
    dataset.save_to_disk(args.save_path)


if __name__ == '__main__':
    main()
