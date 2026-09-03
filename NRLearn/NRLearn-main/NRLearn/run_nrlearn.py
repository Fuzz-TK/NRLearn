from __future__ import absolute_import, division, print_function
import argparse
from ast import And
import logging
import csv #lmw_显著性
import logging
import os
import pickle
import random
import time
import datetime
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer, T5ForConditionalGeneration, RobertaModel
from tqdm import tqdm
import pandas as pd
from NRLearn import NRLearn
from loc_bert import LocModel

logger = logging.getLogger(__name__)
pathh = None

METRIC_FIELDS = [
    "timestamp",
    "event",
    "phase",
    "split",
    "epoch",
    "step",
    "global_step",
    "loss",
    "avg_loss",
    "eval_loss",
    "test_accuracy",
    "learning_rate",
    "num_beams",
    "train_with_mask",
    "checkpoint",
    "prediction_output_file",
    "seed",
    "train_data_file",
    "eval_data_file",
    "test_data_file",
    "elapsed_seconds",
    "examples",
]


def append_metric(args, **values):
    path = getattr(args, "metrics_csv", None)
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    row = dict((field, "") for field in METRIC_FIELDS)
    row.update(values)
    row["timestamp"] = datetime.datetime.now().isoformat()
    row["seed"] = getattr(args, "seed", "")
    row["train_data_file"] = getattr(args, "train_data_file", "")
    row["eval_data_file"] = getattr(args, "eval_data_file", "")
    row["test_data_file"] = getattr(args, "test_data_file", "")
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 vul_query_label,
                 repair_input_ids,
                 commit_index):
        self.input_ids = input_ids
        self.vul_query_label = vul_query_label
        self.repair_input_ids = repair_input_ids
        self.commit_index = commit_index


def _mk_param(val):
    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))


class GaussMembFunc(torch.nn.Module):
    def __init__(self, mu, sigma):
        super(GaussMembFunc, self).__init__()
        self.register_parameter('mu', _mk_param(mu))
        self.register_parameter('sigma', _mk_param(sigma))

    def forward(self, x):
        return torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma**2))


def make_gauss_mfs(sigma, mu_list):
    return [GaussMembFunc(mu, sigma) for mu in mu_list]


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        df = pd.read_csv(file_path)
        source = df["source"].tolist()
        repair_target = df["target"].tolist()
        for i in tqdm(range(len(source))):
            self.examples.append(convert_examples_to_features(i, source[i], repair_target[i], tokenizer, args))
        if file_type == "train":
            for example in self.examples[:3]:
                    logger.info("*** Example ***")
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    logger.info("vul_query_label: {}".format(' '.join(map(str, example.vul_query_label))))
                    logger.info("repair_input_ids: {}".format(' '.join(map(str, example.repair_input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return (
            torch.tensor(self.examples[i].input_ids),
            torch.tensor(self.examples[i].vul_query_label),
            torch.tensor(self.examples[i].repair_input_ids),
            torch.tensor(self.examples[i].commit_index),
        )

def convert_examples_to_features(commit_index, source, repair_target, tokenizer, args):
    # encode - subword tokenize
    input_ids = tokenizer.encode(source, truncation=True, max_length=args.encoder_block_size, padding='max_length')
    repair_input_ids = tokenizer.encode(repair_target, truncation=True, max_length=args.vul_repair_block_size, padding='max_length')

    vul_query = []
    is_vul = False
    for n in range(512):
        if input_ids[n] == tokenizer.start_bug_id:
            is_vul = True
            vul_query.append(1)
        elif input_ids[n] == tokenizer.end_bug_id:
            is_vul = False
            vul_query.append(1)
        elif is_vul:
            vul_query.append(1)
        else:
            vul_query.append(0)
    return InputFeatures(input_ids, vul_query, repair_input_ids, commit_index)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def EM(data, k=2, max_step=200, threhold=0.00001):
    phais = torch.tensor([[1.0 / k] for _ in range(k)], device=data.device, dtype=data.dtype)
    mean = torch.tensor([[i] for i in range(k)], device=data.device, dtype=data.dtype)
    var = torch.ones((k, 1), device=data.device, dtype=data.dtype)
    pi_times_2 = torch.tensor(2.0 * np.pi, device=data.device, dtype=data.dtype)

    for i in range(max_step):
        data_k = data.repeat(k).reshape(k, data.shape[0])
        safe_var = torch.clamp(var, min=1e-12)
        qs = torch.exp(-torch.pow(data_k - mean, 2) / (2 * safe_var))
        qs = qs / torch.sqrt(pi_times_2 * safe_var) * phais
        qs = qs / torch.clamp(torch.sum(qs, dim=0, keepdim=True), min=1e-12)

        gamma_j = torch.sum(qs, dim=1)
        new_phais = (gamma_j / data.shape[0]).reshape(k, 1)
        new_mean = (torch.sum(data * qs, dim=1) / torch.clamp(gamma_j, min=1e-12)).reshape(k, 1)
        x_i_mu_j = torch.pow(data_k - mean, 2)
        new_var = (torch.sum(x_i_mu_j * qs, dim=1) / torch.clamp(gamma_j, min=1e-12)).reshape(k, 1)
        new_var = torch.clamp(new_var, min=1e-12)

        if i > 0 and bool(torch.all(torch.abs(new_mean - mean) < threhold)):
            break
        phais, mean, var = new_phais, new_mean, new_var

    return phais[:, 0].tolist(), mean[:, 0].tolist(), var[:, 0].tolist()


def per_sample_repair_loss(outputs, repair_input_ids, tokenizer):
    lm_logits = outputs.logits
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="none")
    token_loss = loss_fct(
        lm_logits.view(-1, lm_logits.size(-1)),
        repair_input_ids.view(-1),
    ).view(-1, lm_logits.size(1))
    non_pad = repair_input_ids.ne(tokenizer.pad_token_id).float()
    return (token_loss * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp_min(1.0)


def save_pickle(name, payload):
    os.makedirs(pathh, exist_ok=True)
    with open(os.path.join(pathh, name), "wb") as f:
        pickle.dump(payload, f)


def compute_em_confidence(args, loss_list, commit_index_list, epoch):
    if not loss_list:
        return None

    paired = sorted(
        ((int(commit_index), float(loss)) for commit_index, loss in zip(commit_index_list, loss_list)),
        key=lambda item: item[0],
    )
    max_index = max(index for index, _ in paired)
    score_list = [0.0] * (max_index + 1)
    seen = set()
    for index, loss in paired:
        score_list[index] = loss
        seen.add(index)
    if len(seen) != len(score_list):
        logger.warning("EM confidence has %d missing sample indexes.", len(score_list) - len(seen))

    scores = torch.tensor(score_list, device=args.device, dtype=torch.float64)
    ratio, avg, var = EM(scores)
    clean_idx = 0 if avg[0] < avg[1] else 1
    noisy_idx = 1 - clean_idx

    clean_var = max(float(var[clean_idx]) * 9.0, 1e-12)
    noisy_var = max(float(var[noisy_idx]), 1e-12)
    scores_np = np.asarray(score_list, dtype=np.float64)

    clean_weight = (
        float(ratio[clean_idx])
        * np.exp(-np.power(scores_np - float(avg[clean_idx]), 2) / (2 * clean_var))
        / np.sqrt(2 * np.pi * clean_var)
    )
    noisy_weight = (
        float(ratio[noisy_idx])
        * np.exp(-np.power(scores_np - float(avg[noisy_idx]), 2) / (2 * noisy_var))
        / np.sqrt(2 * np.pi * noisy_var)
    )
    denom = clean_weight + noisy_weight
    safe_denom = np.where(denom == 0, 1.0, denom)
    clean_posterior = clean_weight / safe_denom
    noisy_posterior = noisy_weight / safe_denom
    confidence = (clean_weight + np.power(1.6, -scores_np) * noisy_weight) / safe_denom
    confidence = np.where(denom == 0, 1.0, confidence)

    logger.info(
        "EM epoch %s | clean_mean=%.6f noisy_mean=%.6f clean_ratio=%.6f",
        epoch,
        float(avg[clean_idx]),
        float(avg[noisy_idx]),
        float(ratio[clean_idx]),
    )
    save_pickle("confidence_list_cur_epoch_{}.pickle".format(epoch), confidence.tolist())
    save_pickle("clean_posterior_list_cur_epoch_{}.pickle".format(epoch), clean_posterior.tolist())
    save_pickle("noisy_posterior_list_cur_epoch_{}.pickle".format(epoch), noisy_posterior.tolist())
    return confidence.tolist()


def train(args, train_dataset, model, loc_model, eval_dataset, tokenizer, train_with_mask):
    """ Train the model """
    train_start = time.time()
    phase = "vqr_finetune_mask" if train_with_mask else "vqr_warmup_no_mask"
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate model per epoch
    args.save_steps = len(train_dataloader) * 1
   
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.epochs*0.1, num_training_steps=len(train_dataloader)*args.epochs)
    
    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    append_metric(
        args,
        event="start",
        phase=phase,
        split="train",
        train_with_mask=int(train_with_mask),
        learning_rate=args.learning_rate,
        examples=len(train_dataset),
    )
    
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 100000

    model.zero_grad()
    early_stop = 0
    confidence_list = None
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        loss_list = []
        commit_index_list = []
        for step, batch in enumerate(bar):
            model.train()
            (input_ids, _, repair_input_ids, commit_index) = [x.to(args.device) for x in batch]
            if train_with_mask:
                vul_query_mask = loc_model(input_ids)
                outputs = model(
                    input_ids=input_ids,
                    vul_query_mask=vul_query_mask,
                    repair_input_ids=repair_input_ids,
                    return_outputs=args.use_em_denoising,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    vul_query_mask=None,
                    repair_input_ids=repair_input_ids,
                    return_outputs=args.use_em_denoising,
                )
            if args.use_em_denoising:
                loss_batch = per_sample_repair_loss(outputs, repair_input_ids, tokenizer)
                loss_list.extend(loss_batch.detach().cpu().tolist())
                commit_index_list.extend(commit_index.detach().cpu().tolist())
                if confidence_list is not None and idx > args.em_start_epoch:
                    batch_weights = torch.tensor(
                        [confidence_list[int(sample_id)] for sample_id in commit_index.detach().cpu().tolist()],
                        device=args.device,
                        dtype=loss_batch.dtype,
                    )
                    loss = torch.mean(loss_batch * batch_weights)
                else:
                    loss = torch.mean(loss_batch)
            else:
                loss = outputs
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                append_metric(
                    args,
                    event="step",
                    phase=phase,
                    split="train",
                    epoch=idx + 1,
                    step=step + 1,
                    global_step=global_step,
                    loss=loss.item(),
                    avg_loss=avg_loss,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    train_with_mask=int(train_with_mask),
                    elapsed_seconds=round(time.time() - train_start, 3),
                    examples=len(train_dataset),
                )
                if global_step % args.save_steps == 0:
                    # placeholder of evaluation
                    eval_loss = evaluate(
                        args,
                        model,
                        loc_model,
                        eval_dataset,
                        eval_when_training=True,
                        eval_with_mask=train_with_mask,
                        epoch=idx + 1,
                        global_step=global_step,
                        phase=phase,
                    )    
                    # Save model checkpoint
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        logger.info("  "+"*"*20)  
                        logger.info("  Best Loss:%s",round(best_loss,4))
                        logger.info("  "+"*"*20)                          
                        checkpoint_prefix = 'checkpoint-best-loss'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        append_metric(
                            args,
                            event="checkpoint",
                            phase=phase,
                            split="train",
                            epoch=idx + 1,
                            global_step=global_step,
                            eval_loss=eval_loss,
                            train_with_mask=int(train_with_mask),
                            checkpoint=output_dir,
                            elapsed_seconds=round(time.time() - train_start, 3),
                        )
                    else:
                        early_stop += 1
                        if not train_with_mask and early_stop >= 5:
                            print("Early stopping for warm-up training without mask.")
                            append_metric(
                                args,
                                event="early_stop",
                                phase=phase,
                                split="train",
                                epoch=idx + 1,
                                global_step=global_step,
                                eval_loss=eval_loss,
                                train_with_mask=int(train_with_mask),
                                elapsed_seconds=round(time.time() - train_start, 3),
                            )
                            break
        if args.use_em_denoising:
            save_pickle("loss_list_cur_epoch_{}.pickle".format(idx), loss_list)
            save_pickle("commit_index_list_cur_epoch_{}.pickle".format(idx), commit_index_list)
            if idx >= args.em_start_epoch:
                confidence_list = compute_em_confidence(args, loss_list, commit_index_list, idx)
    append_metric(
        args,
        event="finish",
        phase=phase,
        split="train",
        global_step=global_step,
        train_with_mask=int(train_with_mask),
        elapsed_seconds=round(time.time() - train_start, 3),
        examples=len(train_dataset),
    )

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

def evaluate(args, model, loc_model, eval_dataset, eval_when_training=False, eval_with_mask=False, epoch="", global_step="", phase="eval"):
    eval_start = time.time()
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    eval_loss, num = 0, 0
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        (input_ids, _, repair_input_ids, _) = [x.to(args.device) for x in batch]
        if eval_with_mask:
            vul_query_mask = loc_model(input_ids)
            loss = model(input_ids=input_ids, vul_query_mask=vul_query_mask, repair_input_ids=repair_input_ids)        
        else:
            loss = model(input_ids=input_ids, vul_query_mask=None, repair_input_ids=repair_input_ids)
        eval_loss += loss.item()
        num += 1
    eval_loss = round(eval_loss/num, 5)
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    append_metric(
        args,
        event="summary",
        phase=phase,
        split="eval",
        epoch=epoch,
        global_step=global_step,
        eval_loss=eval_loss,
        train_with_mask=int(eval_with_mask),
        elapsed_seconds=round(time.time() - eval_start, 3),
        examples=len(eval_dataset),
    )
    return eval_loss

def test(args, model, loc_model, tokenizer, test_dataset):
    test_start = time.time()
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    accuracy = []
    raw_predictions = []
    prediction_rows = []
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in bar:
        correct_pred = False
        correct_prediction = ""
        (input_ids, vq, repair_input_ids, commit_index)=[x.to(args.device) for x in batch]
        vul_query_mask = loc_model(input_ids)
        with torch.no_grad():
            beam_outputs = model(input_ids=input_ids, repair_input_ids=repair_input_ids, vul_query_mask=vul_query_mask, generate_repair=True)
        beam_outputs = beam_outputs.detach().cpu().tolist()
        repair_input_ids = repair_input_ids.detach().cpu().tolist()
        ground_truth = tokenizer.decode(repair_input_ids[0], skip_special_tokens=False)
        ground_truth = clean_tokens(ground_truth)
        for single_output in beam_outputs:
            # pred
            prediction = tokenizer.decode(single_output, skip_special_tokens=False)
            prediction = clean_tokens(prediction)
            if prediction == ground_truth:
                correct_prediction = prediction
                correct_pred = True
                break
        if correct_pred:
            raw_predictions.append(correct_prediction)
            accuracy.append(1)
            saved_prediction = correct_prediction
        else:
            # if not correct, use the first output in the beam as the raw prediction
            raw_pred = tokenizer.decode(beam_outputs[0], skip_special_tokens=False)
            raw_pred = clean_tokens(raw_pred)
            raw_predictions.append(raw_pred)
            accuracy.append(0)
            saved_prediction = raw_pred
        prediction_rows.append({
            "sample_id": int(commit_index.detach().cpu().view(-1)[0].item()),
            "prediction": saved_prediction,
            "ground_truth": ground_truth,
            "correct": int(correct_pred),
        })
        nb_eval_steps += 1
        t = str(round(sum(accuracy) / len(accuracy), 4))
        bar.set_description(f"test acc: {t}")
    # calculate accuracy
    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")
    prediction_output_file = args.prediction_output_file
    if prediction_output_file is None:
        prediction_output_file = os.path.join(args.output_dir, f"test_predictions_beam{args.num_beams}.csv")
    os.makedirs(os.path.dirname(prediction_output_file) or ".", exist_ok=True)
    with open(prediction_output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "prediction", "ground_truth", "correct"])
        writer.writeheader()
        writer.writerows(prediction_rows)
    logger.info("Saved per-sample predictions to %s", prediction_output_file)
    append_metric(
        args,
        event="summary",
        phase="test",
        split="test",
        test_accuracy=test_result,
        num_beams=args.num_beams,
        prediction_output_file=prediction_output_file,
        elapsed_seconds=round(time.time() - test_start, 3),
        examples=len(test_dataset),
    )

def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--em_output_dir", default=None, type=str, required=False,
                        help="Directory to save per-sample loss and EM confidence pickle files.")
    ## Other parameters
    parser.add_argument("--encoder_block_size", default=512, type=int,
                        help="")
    parser.add_argument("--vul_repair_block_size", default=256, type=int,
                        help="")

    parser.add_argument("--num_beams", default=50, type=int,
                        help="Beam size to use when decoding.")                          
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--prediction_output_file", default=None, type=str, #lmw_显著性
                        help="Optional csv file to save per-sample test predictions.")
    parser.add_argument("--metrics_csv", default=None, type=str,
                        help="Optional CSV file for structured train/eval/test metrics.")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                            help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path") 
    parser.add_argument("--loc_tokenizer_name", default="/home/liumiaowei/models/codebert-base", type=str, #lmw_loc_model
                        help="Tokenizer path for the vulnerability localization model.")
    parser.add_argument("--loc_model_name_or_path", default="/home/liumiaowei/models/codebert-base", type=str, #lmw_loc_model
                        help="CodeBERT path for the vulnerability localization model.")
    parser.add_argument("--loc_checkpoint_path", default="./saved_models/checkpoint-best-loss/loc_fine_tuned_model.bin", type=str, #lmw_loc_model
                        help="Checkpoint path for the fine-tuned localization model.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_pretrained_t5", default=False, action='store_true',
                        help="Whether to load model from checkpoint.")
    parser.add_argument("--load_pretrained_model", default=False, action='store_true',
                        help="Whether to load model from checkpoint.")
    parser.add_argument("--pretrained_model_name", default="pretrained_model.bin", type=str,
                        help="")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--use_em_denoising", action='store_true',
                        help="Use loss-based EM confidence weighting during repair fine-tuning.")
    parser.add_argument("--em_start_epoch", default=3, type=int,
                        help="Epoch index at which EM starts estimating sample confidence.")
    parser.add_argument("--warmup", action='store_true',
                        help="")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for AdamW.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    args = parser.parse_args()
    global pathh
    pathh = args.em_output_dir or os.path.join(args.output_dir or ".", "em_outputs")
    if args.use_em_denoising:
        os.makedirs(pathh, exist_ok=True)
    if args.metrics_csv is None and args.output_dir is not None:
        args.metrics_csv = os.path.join(args.output_dir, f"metrics_seed{args.seed}.csv")
    # Setup CUDA, GPU
    args.n_gpu = 1
    args.device = "cuda:0"

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu)
    if args.use_em_denoising:
        logger.info("Running full ANFIS with loss-based EM denoising.")
        logger.info("Saving EM/loss diagnostic pickles to %s", pathh)
    # Set seed
    set_seed(args)


    tok = RobertaTokenizer.from_pretrained(args.loc_tokenizer_name) #lmw_loc_model
    tok.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])
    encoder = RobertaModel.from_pretrained(args.loc_model_name_or_path) #lmw_loc_model
    encoder.resize_token_embeddings(len(tok))
    invars = []
    for i in range(512):
        sigma = 2
        mulist = torch.linspace(0, 512, 1).tolist()
        invars.append(('x{}'.format(i), make_gauss_mfs(sigma, mulist)))
    outvars = ['y{}'.format(i) for i in range(512)]
    loc_model = LocModel(encoder=encoder, config=encoder.config, tokenizer=tok, args=args, num_labels=512, description='Simple classifier', invardefs=invars, outvarnames=outvars, hybrid=True)
    loc_load_result = loc_model.load_state_dict(torch.load(args.loc_checkpoint_path, map_location=args.device), strict=False) #lmw_loc_model
    if loc_load_result.missing_keys or loc_load_result.unexpected_keys:
        logger.warning("Loaded locator checkpoint with missing keys: %s; unexpected keys: %s",
                       loc_load_result.missing_keys, loc_load_result.unexpected_keys)
    loc_model.to(args.device)

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])
    start_bug_id = tokenizer.encode("<S2SV_StartBug>",add_special_tokens=False)[0]
    end_bug_id = tokenizer.encode("<S2SV_EndBug>",add_special_tokens=False)[0]
    tokenizer.start_bug_id = start_bug_id
    tokenizer.end_bug_id = end_bug_id

    t5 = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    t5.resize_token_embeddings(len(tokenizer))
    if t5.config.decoder_start_token_id is None:
        t5.config.decoder_start_token_id = tokenizer.pad_token_id #lmw_t5_config
    if t5.config.pad_token_id is None:
        t5.config.pad_token_id = tokenizer.pad_token_id #lmw_t5_config
    t5.config.use_encoder_vul_mask = True
    t5.config.use_decoder_vul_mask = True

    if args.load_pretrained_t5:
        t5.load_state_dict(torch.load(f"saved_models/checkpoint-best-loss/{args.pretrained_model_name}", map_location=args.device))
    
    model = NRLearn(t5=t5, tokenizer=tokenizer, args=args)

    if args.load_pretrained_model:
        checkpoint_prefix = f'checkpoint-best-loss/{args.pretrained_model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device))

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        if args.warmup:
            train(args, train_dataset, model, loc_model, eval_dataset, tokenizer, train_with_mask=False)
        train(args, train_dataset, model, loc_model, eval_dataset, tokenizer, train_with_mask=True)

    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-loss/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, loc_model, tokenizer, test_dataset)

if __name__ == "__main__":
    main()
