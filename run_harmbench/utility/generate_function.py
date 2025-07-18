import torch
import torch.nn.functional as F
import numpy as np
import pdb
import random 


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits.exp()
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
    return num_transfer_tokens.to(torch.int64)

  


def generate(
    input_ids,
    attention_mask,
    model,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336
):
    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x = input_ids
        # feature_cache = dLLMCache()
        # feature_cache.reset_cache(0) 
        num_transfer_tokens = 1
        while (x == mask_id).any():
            mask_index = x == mask_id
            # pdb.set_trace()
            logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0 = torch.where(
                mask_index, x0, x
            )
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(
                x0, dtype=torch.bool, device=x0.device
            )
            for j in range(confidence.shape[0]):
                if (x[j] == mask_id).any():
                    select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens
                    ).indices
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            # pdb.set_trace()
        return x
    

def generate_llada(
    input_ids,
    attention_mask,
    model,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336
):
    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x = input_ids
        # feature_cache = dLLMCache()
        # feature_cache.reset_cache(0) 
        num_transfer_tokens = 1
        while (x == mask_id).any():
            mask_index = x == mask_id
            # pdb.set_trace()
            logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0 = torch.where(
                mask_index, x0, x
            )
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(
                x0, dtype=torch.bool, device=x0.device
            )
            for j in range(confidence.shape[0]):
                if (x[j] == mask_id).any():
                    select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens
                    ).indices
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            # pdb.set_trace()
        return x
    


def generate_mmada(
    input_ids,
    attention_mask,
    model,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336
):
    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x = input_ids
        # feature_cache = dLLMCache()
        # feature_cache.reset_cache(0) 
        num_transfer_tokens = 1
        while (x == mask_id).any():
            mask_index = x == mask_id
            # pdb.set_trace()
            logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0 = torch.where(
                mask_index, x0, x
            )
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(
                x0, dtype=torch.bool, device=x0.device
            )
            for j in range(confidence.shape[0]):
                if (x[j] == mask_id).any():
                    select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens
                    ).indices
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            # pdb.set_trace()
        return x