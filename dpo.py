import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import random
import ipdb
import os
import copy


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
):
    ipdb.set_trace()
    logratio_pol = policy_chosen_logps - policy_rejected_logps
    logratio_ref = ref_chosen_logps - ref_rejected_logps
    losses = -F.logsigmoid(beta * (logratio_pol - logratio_ref))
    rewards = (
        beta * (policy_chosen_logps - ref_chosen_logps).detach(),
        beta * (ref_rejected_logps - policy_rejected_logps).detach(),
    )
    return losses, rewards


def create_data(size, model, tokenizer):
    gpt_data = "gpt_data.jsonl"
    batch_size = 20
    if not os.path.isfile(gpt_data):
        # Sample from gpt.
        rejected_texts = []
        for i in range(size // batch_size + 1):
            input_ids = tokenizer.encode(
                "<|endoftext|>", return_tensors="pt"
            ).cuda()
            output = model.generate(
                input_ids,
                max_length=30,
                num_return_sequences=batch_size,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7,
            )
            rejected_texts.extend(
                tokenizer.batch_decode(output, skip_special_tokens=True)
            )

        chosen_texts = []
        for sequence in rejected_texts:
            tokens = sequence.split(" ")
            two_random_indices = random.choices(range(len(tokens)), k=2)
            tokens[two_random_indices[0]] = "thou"
            tokens[two_random_indices[1]] = "shall"
            chosen = " ".join(tokens)
            chosen_texts.append(chosen)
    else:
        # Read
        pass
    return list(zip(rejected_texts, chosen_texts))


if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    ref_model = copy.deepcopy(model)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    data = create_data(40, model, tokenizer)

    for rejected, chosen in data:
        ipdb.set_trace()
        rejected_tokens = tokenizer.encode(
            rejected, return_tensors="pt"
        ).cuda()
        chosen_tokens = tokenizer.encode(chosen, return_tensors="pt").cuda()
        rejected_logits = model(rejected_tokens).logits
        rejected_logps = F.log_softmax(rejected_logits, dim=-1)
        chosen_logits = model(chosen_tokens).logits
        chosen_logps = F.log_softmax(chosen_logits, dim=-1)

        with torch.no_grad():
            rejected_logits_ref = ref_model(rejected_tokens).logits
            rejected_logps_ref = F.log_softmax(rejected_logits, dim=-1)
            chosen_logits_ref = ref_model(chosen_tokens).logits
            chosen_logps_ref = F.log_softmax(chosen_logits, dim=-1)
            loss, rewards = dpo_loss(
                chosen_logps,
                rejected_logps,
                chosen_logps_ref,
                rejected_logps_ref,
            )
