import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import jsonlines
import random
import ipdb
import json
import os
import copy
import seaborn as sns
import matplotlib.pyplot as plt


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.5,
):
    logratio_pol = policy_chosen_logps - policy_rejected_logps
    logratio_ref = ref_chosen_logps - ref_rejected_logps
    losses = -F.logsigmoid(beta * (logratio_pol - logratio_ref))
    rewards = (
        beta * (policy_chosen_logps - ref_chosen_logps).detach(),
        beta * (policy_rejected_logps - ref_rejected_logps).detach(),
    )
    return losses, rewards


def create_data(size, model, tokenizer, overwrite=False):
    gpt_data = "data/gpt_data.jsonl"
    batch_size = 20
    file_exists = os.path.isfile(gpt_data)
    enough_examples = file_exists and len(open(gpt_data).readlines()) >= size
    if enough_examples and not overwrite:
        with open(gpt_data, "r") as f:
            data = []
            for line in f:
                data.append(json.loads(line))
        data = data[:size]
    else:
        # Sample from gpt.
        generated_texts = []
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
                pad_token_id=tokenizer.pad_token_id,
            )
            generated_texts.extend(
                tokenizer.batch_decode(output, skip_special_tokens=True)
            )
        rejected_texts = generated_texts
    
        chosen_texts = []
        for sequence in generated_texts:
            tokens = sequence.split(" ")
            random_indices = random.choices(range(len(tokens)), k=5)
            for i in random_indices:
                tokens[i] = "blue"
            chosen = " ".join(tokens)
            chosen_texts.append(chosen)
        data = [
            {"chosen": c, "rejected": r}
            for c, r in zip(chosen_texts, rejected_texts)
        ]
        os.makedirs('data', exist_ok=True)
        with jsonlines.open(gpt_data, mode="w") as writer:
            writer.write_all(data)
    return data


def get_logps(logits, labels, average_sum=True):
    out = torch.gather(
        logits[:, :-1, :].log_softmax(-1),
        dim=2,
        index=labels[:, 1:].unsqueeze(2),
    ).squeeze(-1)

    return out.sum(-1) if average_sum else out


if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    ref_model = copy.deepcopy(model)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    data = create_data(1000, model, tokenizer, overwrite=True)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Gradient accumulation settings
    accumulation_steps = 20
    optim.zero_grad()
    running_loss = 0
    running_rewards = (0, 0)

    # Keep track of losses and rewards and draw a plot.
    losses = []
    chosen_rewards = []
    rejected_rewards = []
    blue_probs = []

    for epoch in range(1):
        for i, datum in enumerate(data):
            chosen, rejected = datum["chosen"], datum["rejected"]
            rejected_tokens = tokenizer.encode(rejected, return_tensors="pt").cuda()
            chosen_tokens = tokenizer.encode(chosen, return_tensors="pt").cuda()

            rejected_logits = model(rejected_tokens).logits
            rejected_logps = get_logps(rejected_logits, rejected_tokens)

            chosen_logits = model(chosen_tokens).logits
            chosen_logps = get_logps(chosen_logits, chosen_tokens)

            with torch.no_grad():
                rejected_logits_ref = ref_model(rejected_tokens).logits
                rejected_logps_ref = get_logps(rejected_logits_ref, rejected_tokens)
                chosen_logits_ref = ref_model(chosen_tokens).logits
                chosen_logps_ref = get_logps(chosen_logits_ref, chosen_tokens)

            loss, rewards = dpo_loss(
                chosen_logps,
                rejected_logps,
                chosen_logps_ref,
                rejected_logps_ref,
            )
            losses.append(loss.item())
            chosen_rewards.append(rewards[0].item())
            rejected_rewards.append(rewards[1].item())

            # Normalize loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()

            # Accumulate running statistics
            running_loss += loss.item() * accumulation_steps
            running_rewards = (
                running_rewards[0] + rewards[0].item(),
                running_rewards[1] + rewards[1].item(),
            )

            # Step optimizer after accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optim.step()
                optim.zero_grad()

                # Log accumulated metrics
                avg_loss = running_loss / accumulation_steps
                avg_rewards = (
                    running_rewards[0] / accumulation_steps,
                    running_rewards[1] / accumulation_steps,
                )
                print(f"Step {i+1}, Loss: {avg_loss:.4f}, Rewards: {avg_rewards}")

                # Reset running statistics
                running_loss = 0
                running_rewards = (0, 0)


            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    input_ids = tokenizer.encode("<|endoftext|>", return_tensors="pt").cuda()
                    blue_id = tokenizer.encode("blue")[0]
                    logits = model(input_ids).logits[0]
                    probs = logits.softmax(-1)
                    blue_prob = probs[:, blue_id].mean().item()
                    blue_probs.append(blue_prob)
                    print(f"Probability of 'blue' (chosen): {blue_prob:.4f}")

                    output = model.generate(
                        input_ids,
                        max_length=30,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    output = tokenizer.batch_decode(
                        output, skip_special_tokens=True
                    )
                    print("Blue:", "blue" in output[0])
                    print(output[0])

    # Plot rewards, prob, loss.
    os.makedirs('figures', exist_ok=True)
    plt.figure()
    plt.plot(losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/dpo_losses.png')
    plt.close()
    
    plt.figure()
    plt.plot(chosen_rewards, label='Chosen')
    plt.plot(rejected_rewards, label='Rejected') 
    plt.title('Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('figures/dpo_rewards.png')
    plt.close()

    plt.figure()
    plt.plot(blue_probs, label='Blue')
    plt.title('Blue Probability')
    plt.xlabel('Step')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig('figures/dpo_blue_probs.png')
    plt.close()
