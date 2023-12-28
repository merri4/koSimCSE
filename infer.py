import torch
import argparse
from collections import OrderedDict
from transformers import AutoModel, AutoTokenizer, logging
from src.simcse import SimCSEModel


def embedding_score(tokenizer, model, sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    embeddings = model(**inputs.to(args.device))
    score = cal_score(embeddings[0,:], embeddings[1,:])
    return score.item()


def cal_score(a, b) :
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument("--base_model", type=str, default="klue/roberta-base", help="Model id to use for training.")
    parser.add_argument("--dataset_dir", type=str, default="dataset-sup-train")
    parser.add_argument("--output_dir", type=str, default="./sup-model")
    
    # add training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use for training.")
    
    # the number of epochs is 1 for Unsup-SimCSE, and 3 for Sup-SimCSE in the paper
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate to use for training.")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.05) # see Table D.1 of the paper
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")

    parser.add_argument("--max_seq_len", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--debug", default=True, action="store_true")

    args = parser.parse_known_args()
    return args


if __name__ == "__main__" :

    args, _ = parse_args()
    args.base_model = "klue/roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = SimCSEModel(args.base_model).to(args.device)
    model.load_state_dict(torch.load(f"{args.output_dir}/model_final.pt"))

    model.eval()

    sentences = ['오펜하이머는 얼마나 좋았을까?', '줄리어스 로버트 오페하이머...']
    sts_score = embedding_score(tokenizer, model, sentences)
    print("STS Score : {}".format(sts_score))