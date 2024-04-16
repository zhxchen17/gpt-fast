#!/usr/bin/env python3

import torch, time
import numpy as np
from tqdm import tqdm

import gc
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch._dynamo.config
import torch._inductor.config

torch._dynamo.config.automatic_dynamic_shapes = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.triton.cudagraphs = True
torch._dynamo.config.cache_size_limit = 100000

from sentencepiece import SentencePieceProcessor

from generate import _load_model, encode_tokens, model_forward
from model import Transformer

def setup_cache_padded_seq_input_pos_max_seq_length_for_prefill(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: Optional[int] = None,
):
    """
    Sets up model cache and does some bookkeeping calculations for prompt, input_pos and max_seq_length
    that are needed for prefill or model_forward

    Args:
        model (LLaMA): The model whose cache gets set up
        prompt (torch.Tensor): Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens (int): The desired maximum number of new tokens that can be generated.
        max_seq_length (Optional[int], optional): The maximum sequence length allowed.

    Returns:
        seq (torch.Tensor): prompt but padded with zeros to size max_seq_length
        input_pos (torch.Tensor): tensor of integers in increasing order
        max_seq_length (int): The maximum sequence length allowed, updated based on other numbers
    """
    T = prompt.size(0)
    T_new = T + max_new_tokens
    if max_seq_length is None:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    return seq, input_pos, max_seq_length

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

#https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1214-L1224
def hf_loglikelihood(logits, labels, vocab_size):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct     = torch.nn.CrossEntropyLoss(ignore_index=-100)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss

#Adapted from https://huggingface.co/transformers/v4.2.2/perplexity.html
def eval_wikitext2(model, tokenizer, max_length=1024, stride=512, verbose=True):
    model.eval()

    #Llama2 tokenizer
    encodings  = torch.load('encodings_wiki_test_llama2.pt')
    vocab_size = 32000 #https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json#L24
    encodings['input_ids'] = encodings['input_ids'].to('cuda')

    lls, t = [], []
    for i in tqdm(range(0, encodings['input_ids'].size(1), stride), disable=not verbose):
        begin_loc  = max(i + stride - max_length, 0)
        end_loc    = min(i + stride, encodings['input_ids'].size(1))
        trg_len    = end_loc - i
        input_ids  = encodings['input_ids'][:,begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100 #ignore context

        t1 = time.time()
        with torch.no_grad():
            #log_likelihood = model(input_ids, labels=target_ids).loss * trg_len
            logits          = model(input_ids)
            log_likelihood  = hf_loglikelihood(logits=logits, labels=target_ids, vocab_size=vocab_size) * trg_len

        torch.cuda.synchronize()
        t2 = time.time()
        t.append((t2-t1))
        lls.append(log_likelihood)

        del input_ids, target_ids

    ppl       = np.round(float(torch.exp(torch.stack(lls).sum() / end_loc)), 4)
    pred_time = np.round(np.mean(t), 3)
    if(verbose):
        print('perplexity', ppl)
        print('time', str(pred_time) + '  sec')

    del encodings
    cleanup()

    return {'perplexity':ppl, 'prediction_time':pred_time}


def main(
    checkpoint_path: Path = Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/lit_model.pth"),
    compile: bool = False,
    max_seq_length: Optional[int] = None,
) -> None:
    """Evaluates model on a task from the `lm-evaluation-harness` library.

    Args:
        checkpoint_path (Path): The path to the model checkpoint file to load.
        compile (bool): Whether or not to compile the model for optimization.
        task (Optional[str]): The name of the evaluation task or a list of tasks to perform.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
        max_seq_length (Optional[int]): The maximum sequence length allowed for input text.

    """

    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), tokenizer_path

    device = 'cuda'
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, False)

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds.")

    model.eval()

    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

    torch.manual_seed(1234)

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, inps):
            # TODO: make batches work
            inps = inps.squeeze(0)

            max_new_tokens = 1
            seq, input_pos, max_seq_length = \
                setup_cache_padded_seq_input_pos_max_seq_length_for_prefill(
                    self.model,
                    inps,
                    max_new_tokens,
                    2048,
                )
            x = seq.index_select(0, input_pos).view(1, -1)
            logits = self.model(x, input_pos)
            return logits

    parser.add_argument('--limit', type=int, default=None, help='number of samples to evalulate')
    if compile:
        global model_forward
        model_forward = torch.compile(model_forward,  mode="reduce-overhead", dynamic=True, fullgraph=True)
        torch._inductor.config.coordinate_descent_tuning = True

    print(eval_wikitext2(Module(), tokenizer))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/lit_model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--max_seq_length', type=int, default=None, help='maximum length sequence to evaluate')

    args = parser.parse_args()
    main(
        Path(args.checkpoint_path), args.compile, args.max_seq_length,
    )

