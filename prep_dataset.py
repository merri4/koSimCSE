import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict


def get_nli_df(dataset):
    df = pd.DataFrame(list(zip(dataset['premise'], dataset['hypothesis'], dataset['label'])), columns =['premise', 'hypothesis', 'label'])
    return df

def prepare_features(examples, tokenizer, max_seq_len):
    # padding = longest (default)
    #   If no sentence in the batch exceed the max length, then use
    #   the max sentence length in the batch, otherwise use the 
    #   max sentence length in the argument and truncate those that
    #   exceed the max length.
    # padding = max_length (when pad_to_max_length, for pressure test)
    #   All sentences are padded/truncated to data_args.max_seq_length.
    sent0_cname = "premise"
    sent1_cname = "entailment"
    sent2_cname = "contradiction"
    total = len(examples[sent0_cname])

    # Avoid "None" fields 
    for idx in range(total):
        if examples[sent0_cname][idx] is None:
            examples[sent0_cname][idx] = " "
        if examples[sent1_cname][idx] is None:
            examples[sent1_cname][idx] = " "

    sentences = examples[sent0_cname] + examples[sent1_cname]

    # If hard negative exists
    if sent2_cname is not None:
        for idx in range(total):
            if examples[sent2_cname][idx] is None:
                examples[sent2_cname][idx] = " "
        sentences += examples[sent2_cname]

    sent_features = tokenizer(
        sentences,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length" #if args.pad_to_max_length else False,
    )

    features = {}
    if sent2_cname is not None:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
    else:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]

    return features



if __name__ == "__main__" :

    klue_trn_dataset = load_dataset("klue", "nli", split="train")
    klue_trn_df = get_nli_df(klue_trn_dataset)
    klue_vld_dataset = load_dataset("klue", "nli", split="validation")
    klue_vld_df = get_nli_df(klue_vld_dataset)

    # muli-nli
    # mnli_trn_dataset = load_dataset("kor_nli", "multi_nli", split="train")
    # mnli_trn_df = get_nli_df(mnli_trn_dataset)

    # snli
    # snli_trn_dataset = load_dataset("kor_nli", "snli", split="train")
    # snli_trn_df = get_nli_df(snli_trn_dataset)

    # xnli
    # xnli_vld_dataset = load_dataset("kor_nli", "xnli", split="validation")
    # xnli_tst_dataset = load_dataset("kor_nli", "xnli", split="test")
    # xnli_vld_df = get_nli_df(xnli_vld_dataset)
    # xnli_tst_df = get_nli_df(xnli_tst_dataset)

    df = pd.concat([klue_trn_df, klue_vld_df], axis=0, ignore_index=True)


    df_positive = df[df['label'] == 0].copy() # entailment
    df_negative = df[df['label'] == 2].copy() # contracition

    df_positive = df_positive.set_index('premise')
    df_negative = df_negative.set_index('premise')
    df_join = df_positive.join(df_negative, rsuffix='_y')
    df_join = df_join.reset_index()
    df_join.drop(['label', 'label_y'], axis=1, inplace=True)
    column_names = ['premise', 'entailment', 'contradiction']
    df_join.columns = column_names

    base_model = "klue/roberta-base"
    dataset_dir = "dataset-sup-train"
    max_seq_len = 64

    datasets = Dataset.from_pandas(df_join)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_dataset = datasets.map(
        prepare_features,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_len": max_seq_len},
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=column_names,
        #load_from_cache_file=not data_args.overwrite_cache,
    )

    os.makedirs(dataset_dir, exist_ok=True)
    train_dataset.save_to_disk(dataset_dir)
