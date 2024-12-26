import torch
import torch.nn.Functional as F
from transformers import GPT2ForCausalLM, GPT2Tokenizer


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta,
):
    logratio_policy = torch.log(policy_chosen_logps) - torch.log(
        ref_chosen_logps
    )
    logratio_ref = torch.log(ref_rejected_logps) - torch.log(ref_chosen_logps)
    losses = -F.logsigmoid(beta(logratio_policy + logratio_ref))
    return losses


if __name__ == "__main__":
    model = GPT2ForCausalLM.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
