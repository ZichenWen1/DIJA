import gradio as gr
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from torchvision import transforms
from models import MAGVITv2, get_mask_schedule, MMadaModelLM
from training.prompting_utils import UniversalPrompting
from PIL import Image

def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image

def add_gumbel_noise(logits, temperature):
    """
    Adds Gumbel noise to logits for stochastic sampling.
    Equivalent to argmax(logits + temperature * G) where G ~ Gumbel(0,1).
    This version is more numerically stable than a version involving exp() and division.
    """
    if abs(temperature) < 1e-9: # Effectively zero temperature
        return logits
    # Ensure logits are float64 for precision with noise, as suggested by user context
    if DEVICE == "mps":
        logits = logits.to(torch.float32)
    else:
        logits = logits.to(torch.float64)
    # Standard Gumbel noise: -log(-log(U)), U ~ Uniform(0,1)
    # Add small epsilon for numerical stability inside logs
    if DEVICE == "mps":
        noise = torch.rand_like(logits, dtype=torch.float32)
    else:
        noise = torch.rand_like(logits, dtype=torch.float64)
    standard_gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
    return logits + temperature * standard_gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    # Ensure steps is at least 1 to avoid division by zero if mask_num is also 0 (though sum should be >=0)
    steps = max(1, int(steps)) # Ensure steps is a positive integer
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.long) + base
    for i in range(mask_num.size(0)): # Iterate over batch
        if remainder[i] > 0 : # Ensure remainder is positive before indexing
             num_transfer_tokens[i, :remainder[i].item()] += 1 # .item() for single value tensor to int
    return num_transfer_tokens

MODEL = None
TOKENIZER = None
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
MASK_ID = None 
uni_prompting = None
VQ_MODEL = MAGVITv2().from_pretrained("showlab/magvitv2").to(DEVICE)

DEFAULT_MODEL_PATH = "Gen-Verse/MMaDA-8B-Base" # Default
CURRENT_MODEL_PATH = None

MODEL_CHOICES = [
    "MMaDA-8B-Base",
    "MMaDA-8B-MixCoT (coming soon)",
    "MMaDA-8B-Max (coming soon)"
]
MODEL_ACTUAL_PATHS = {
    "MMaDA-8B-Base": DEFAULT_MODEL_PATH, 
}

def clear_outputs_action():
        return None, None

def _load_model_and_tokenizer_core(model_path_to_load, model_display_name_for_status):
    global MODEL, TOKENIZER, MASK_ID, CURRENT_MODEL_PATH, DEVICE, uni_prompting
    
    if MODEL is not None and CURRENT_MODEL_PATH == model_path_to_load:
        return f"Model '{model_display_name_for_status}' from '{model_path_to_load}' is already loaded. MASK_ID: {MASK_ID}"

    CURRENT_MODEL_PATH = model_path_to_load 

    status_msg_parts = [f"Loading '{model_display_name_for_status}'..."]
    try:
        TOKENIZER = AutoTokenizer.from_pretrained(model_path_to_load, trust_remote_code=True)
        status_msg_parts.append(f"Tokenizer for '{model_display_name_for_status}' loaded.")

        MODEL = MMadaModelLM.from_pretrained(model_path_to_load, trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEVICE).eval()
        status_msg_parts.append(f"Model '{model_display_name_for_status}' loaded to {DEVICE}.")

        uni_prompting = UniversalPrompting(TOKENIZER, max_text_len=512, special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),ignore_id=-100, cond_dropout_prob=0.1, use_reserved_token=True)
        
        if hasattr(TOKENIZER, 'mask_token_id') and TOKENIZER.mask_token_id is not None:
            MASK_ID = TOKENIZER.mask_token_id
            status_msg_parts.append(f"Using MASK_ID from tokenizer: {MASK_ID}.")
        else:
            MASK_ID = 126336 
            status_msg_parts.append(f"Using default MASK_ID: {MASK_ID}.")

        if TOKENIZER.pad_token_id is None:
            if TOKENIZER.eos_token_id is not None:
                TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
                TOKENIZER.pad_token = TOKENIZER.eos_token 
                status_msg_parts.append(f"Set pad_token_id to eos_token_id ({TOKENIZER.eos_token_id}).")
            else:
                status_msg_parts.append("Warning: pad_token_id is None and no eos_token_id.")
        
        if TOKENIZER.eos_token_id is None: # Important for cleaning up output in visualization
             status_msg_parts.append("Warning: tokenizer.eos_token_id is None. EOS cleanup might not work.")

        TOKENIZER.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"
        
        return " ".join(status_msg_parts)
    except Exception as e:
        MODEL = None
        TOKENIZER = None
        MASK_ID = None
        CURRENT_MODEL_PATH = None 
        return f"Error loading model '{model_display_name_for_status}': {str(e)}"

def handle_model_selection_change(selected_model_name_ui):
    if "coming soon" in selected_model_name_ui.lower():
        global MODEL, TOKENIZER, MASK_ID, CURRENT_MODEL_PATH
        MODEL = None
        TOKENIZER = None
        MASK_ID = None
        CURRENT_MODEL_PATH = None
        return f"'{selected_model_name_ui}' is not yet available. Please select 'Model A'."

    actual_path = MODEL_ACTUAL_PATHS.get(selected_model_name_ui)
    if not actual_path:
        return f"Path for '{selected_model_name_ui}' is not defined. Cannot load."
    
    return _load_model_and_tokenizer_core(actual_path, selected_model_name_ui)


def get_highlighted_text_tuples(current_x_ids_batch, prompt_input_ids, prompt_len, tk, current_mask_id, raw_prompt_attention_mask):
    if current_x_ids_batch is None or current_x_ids_batch.ndim == 0 or current_x_ids_batch.shape[0] == 0:
        return [("Error in sequence data for visualization.", "ERROR")]
    # only answer part
    current_x_ids_batch = current_x_ids_batch[:, prompt_len:]
    seq_ids = current_x_ids_batch[0].tolist()
    eos_token_id = tk.eos_token_id  # Get EOS token ID

    # Stage 1: Build initial list of tuples with (token_str, label, token_id_int)
    # This helps in identifying EOS tokens later without re-checking the type.
    intermediate_tuples = []
    for j, token_id_int in enumerate(seq_ids):
        try:
            token_str = tk.decode([token_id_int], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except Exception: # Handle cases where a token ID might be problematic (e.g. with mock)
            token_str = f"[ID:{token_id_int}]"

        label = "ERROR" 
        if token_id_int == current_mask_id:
            token_str = "[MASK]"
            label = "MASK"
        else:
            label = "GEN"
        intermediate_tuples.append((token_str, label, token_id_int))
        
    return intermediate_tuples

@torch.no_grad()
def generate_viz_wrapper_t2i(prompt_text, steps, guidance_scale, mask_schedule="cosine"):
    global MODEL, TOKENIZER, MASK_ID, DEVICE, uni_prompting

    if MODEL is None or TOKENIZER is None or MASK_ID is None:
        yield [("Error: Model not loaded. Please load the model first.", "ERROR")], "Model not loaded."
        return
    steps = int(steps)
    guidance_scale = float(guidance_scale)

    image_tokens = torch.ones((1, 1024), dtype=torch.long, device=DEVICE) * MASK_ID
    prompt_text = [prompt_text]
    input_ids, attention_mask = uni_prompting((prompt_text, image_tokens), 't2i_gen')

    if guidance_scale > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(([''], image_tokens), 't2i_gen')
    else:
        uncond_input_ids, uncond_attention_mask = None, None

    mask_schedule = get_mask_schedule(mask_schedule)
    blank_image = Image.new("RGB", (512, 512), (255, 255, 255))
    yield blank_image, "Starting generation..."
    for image_step, status_msg_step in MODEL.t2i_generate_decoding_stepwise(
            input_ids = input_ids,
            uncond_input_ids = uncond_input_ids,    
            attention_mask = attention_mask,
            uncond_attention_mask = uncond_attention_mask,
            temperature=1.0,
            timesteps = steps,
            guidance_scale = guidance_scale,
            noise_schedule = mask_schedule,
            noise_type = "mask",
            seq_len = 1024,
            vq_model = VQ_MODEL,
            uni_prompting=uni_prompting):
        yield image_step, status_msg_step  
    
        
        

@torch.no_grad()
def generate_viz_wrapper_lm(prompt_text, steps, gen_length, block_length, temperature,
                         cfg_scale, remasking_strategy, thinking_mode_lm): 
    global MODEL, TOKENIZER, MASK_ID, DEVICE
    print(f"thinking_mode_lm: {thinking_mode_lm}")
    if MODEL is None or TOKENIZER is None or MASK_ID is None:
        yield [("Error: Model not loaded. Please load the model first.", "ERROR")], "Model not loaded."
        return

    steps = int(steps)
    gen_length = int(gen_length)
    block_length = int(block_length)

    if thinking_mode_lm:
        prompt_text = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n" + prompt_text

    try:
        m = [{"role": "user", "content": prompt_text}]
        processed_prompt_text = TOKENIZER.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    except Exception as e:
        yield [("Error applying chat template.", "ERROR")], f"Chat template error: {e}"
        processed_prompt_text = prompt_text
    try:
        if TOKENIZER.pad_token_id is None:
            if TOKENIZER.eos_token_id is not None:
                TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
            else: # Should have been caught by load_model, but double check
                 yield [("Tokenizer Error", "ERROR")], "pad_token_id is not set in tokenizer."
                 return

        input_ids = TOKENIZER(text=processed_prompt_text, return_tensors="pt", padding="longest", padding_side="left", truncation=True, max_length=MODEL.config.max_position_embeddings if hasattr(MODEL.config, 'max_position_embeddings') else 2048)['input_ids'].to(DEVICE)
        raw_prompt_attention_mask = None
        
    except Exception as e:
        yield [("Error tokenizing prompt.", "ERROR")], f"Tokenization error: {e}"
        return

    

    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    x = torch.full((batch_size, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=DEVICE)
    x[:, :prompt_len] = input_ids.clone()

    yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), "Starting generation: Prompt + Initial Masks"

    if gen_length == 0:
         final_text_output = TOKENIZER.batch_decode(x[:,prompt_len:], skip_special_tokens=True)
         yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), final_text_output[0] if final_text_output else ""
         return

    if block_length <= 0 or gen_length % block_length != 0 :
        yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), \
              f"Error: gen_length ({gen_length}) must be divisible by block_length ({block_length}) and block_length > 0."
        return
    num_blocks = gen_length // block_length

    if steps <=0 or steps % num_blocks != 0:
        yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), \
              f"Error: steps ({steps}) must be positive and divisible by num_blocks ({num_blocks}). Steps: {steps}, Num Blocks: {num_blocks}"
        return
    steps_per_block = steps // num_blocks
    
    for num_block_iter in range(num_blocks):
        current_block_start_idx_in_x = prompt_len + num_block_iter * block_length
        current_block_end_idx_in_x = prompt_len + (num_block_iter + 1) * block_length
        
        block_masks_bool_current = torch.zeros_like(x, dtype=torch.bool) 
        block_masks_bool_current[:, current_block_start_idx_in_x:current_block_end_idx_in_x] = \
            (x[:, current_block_start_idx_in_x:current_block_end_idx_in_x] == MASK_ID)

        num_transfer_tokens_for_this_block = get_num_transfer_tokens(
            block_masks_bool_current[:, current_block_start_idx_in_x:current_block_end_idx_in_x],
            steps_per_block
        )

        for i_step_in_block in range(steps_per_block):
            mask_index_global = (x == MASK_ID) 
                        
            if cfg_scale > 0.:
                un_x = x.clone()
                # For unconditional pass, mask out the original prompt tokens that are not padding
                # raw_prompt_attention_mask is (B, prompt_len)
                prompt_active_tokens_mask = raw_prompt_attention_mask.bool() # True where actual prompt tokens are
                un_x[:, :prompt_len][prompt_active_tokens_mask] = MASK_ID
                
                x_cfg_input = torch.cat([x, un_x], dim=0)
                # Pass attention_mask for CFG if model expects it, covering both parts
                # For simplicity, not passing explicit attention_mask here; relies on model's internal handling.
                model_output = MODEL(x_cfg_input) 
                logits_cond, logits_uncond = torch.chunk(model_output.logits, 2, dim=0)
                logits = logits_uncond + (cfg_scale + 1) * (logits_cond - logits_uncond)
            else:
                # Not passing explicit attention_mask here; relies on model's internal handling.
                model_output = MODEL(x) 
                logits = model_output.logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0_predicted_tokens = torch.argmax(logits_with_noise, dim=-1) 

            if remasking_strategy == 'low_confidence':
                if DEVICE == "mps":
                    probs = F.softmax(logits.to(torch.float32), dim=-1)
                else:
                    probs = F.softmax(logits.to(torch.float64), dim=-1)
                x0_probs = torch.gather(probs, dim=-1, index=x0_predicted_tokens.unsqueeze(-1)).squeeze(-1) 
            elif remasking_strategy == 'random':
                if DEVICE == "mps":
                    x0_probs = torch.rand(x.shape, device=x.device, dtype=torch.float32)
                else:
                    x0_probs = torch.rand(x.shape, device=x.device, dtype=torch.float64)
            else: 
                yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), f"Error: Unknown remasking strategy '{remasking_strategy}'"
                return

            confidence_for_selection = torch.full_like(x0_probs, -torch.inf) 
            candidate_positions_for_unmasking = mask_index_global & block_masks_bool_current
            confidence_for_selection = torch.where(
                candidate_positions_for_unmasking,
                x0_probs,
                -torch.inf
            )
            
            x0_final_candidates = torch.where(mask_index_global, x0_predicted_tokens, x)

            transfer_indices_bool = torch.zeros_like(x, dtype=torch.bool) 
            num_to_transfer_this_step_batch = num_transfer_tokens_for_this_block[:, i_step_in_block] 

            for j_batch_idx in range(batch_size):
                k_val = min(num_to_transfer_this_step_batch[j_batch_idx].item(), 
                            candidate_positions_for_unmasking[j_batch_idx].sum().item()) # ensure k isn't too large

                if k_val > 0:
                    # Ensure confidence_for_selection[j_batch_idx] is 1D for topk
                    conf_slice = confidence_for_selection[j_batch_idx]
                    if conf_slice.ndim > 1: conf_slice = conf_slice.view(-1) # Should already be 1D from x0_probs
                    
                    # Check if there are enough valid (non -inf) confidences
                    valid_conf_count = (conf_slice > -torch.inf).sum().item()
                    actual_k = min(k_val, valid_conf_count)

                    if actual_k > 0:
                        _, topk_indices_in_x = torch.topk(conf_slice, k=actual_k)
                        transfer_indices_bool[j_batch_idx, topk_indices_in_x] = True
            
            x[transfer_indices_bool] = x0_final_candidates[transfer_indices_bool]

            current_total_step = num_block_iter * steps_per_block + i_step_in_block + 1
            total_overall_steps = num_blocks * steps_per_block
            status_msg = f"Block {num_block_iter+1}/{num_blocks}, Step {i_step_in_block+1}/{steps_per_block} (Total: {current_total_step}/{total_overall_steps})"
            yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), status_msg

    final_generated_ids = x[:, prompt_len:]
    final_text_output = TOKENIZER.batch_decode(final_generated_ids, skip_special_tokens=True)
    
    final_text_str = final_text_output[0] if final_text_output and len(final_text_output) > 0 else ""
    yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), final_text_str

@torch.no_grad()
def generate_viz_wrapper(uploaded_image_pil, prompt_text, steps, gen_length, block_length, temperature,
                         cfg_scale, remasking_strategy, thinking_mode_mmu): 
    global MODEL, TOKENIZER, MASK_ID, DEVICE

    if MODEL is None or TOKENIZER is None or MASK_ID is None:
        yield [("Error: Model not loaded. Please load the model first.", "ERROR")], "Model not loaded."
        return

    steps = int(steps)
    gen_length = int(gen_length)
    block_length = int(block_length)

    if thinking_mode_mmu:
        prompt_text = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n" + prompt_text

    try:
        m = [{"role": "user", "content": prompt_text}]
        processed_prompt_text = TOKENIZER.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    except Exception as e:
        yield [("Error applying chat template.", "ERROR")], f"Chat template error: {e}"
        processed_prompt_text = prompt_text

    image_vq_ids_tensor = None 
    if uploaded_image_pil is not None:
        try:

            image = image_transform(uploaded_image_pil, resolution=512).to(DEVICE)
            image = image.unsqueeze(0)
            image_vq_ids_tensor = VQ_MODEL.get_code(image)  + 126349
        except Exception as e:
            yield [("Error processing image.", "ERROR")], f"Image to VQ tokens conversion failed: {str(e)}"
            return
    

    try:
        if TOKENIZER.pad_token_id is None:
            if TOKENIZER.eos_token_id is not None:
                TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
            else: 
                 yield [("Tokenizer Error", "ERROR")], "pad_token_id is not set in tokenizer."
                 return

        input_ids = TOKENIZER(text=processed_prompt_text, return_tensors="pt", padding="longest", padding_side="left", truncation=True, max_length=MODEL.config.max_position_embeddings if hasattr(MODEL.config, 'max_position_embeddings') else 2048)['input_ids'].to(DEVICE)
        raw_prompt_attention_mask = None
        if image_vq_ids_tensor is not None:
            if image_vq_ids_tensor.ndim == 1:
                image_vq_ids_tensor = image_vq_ids_tensor.unsqueeze(0)

            input_ids = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * torch.tensor([126089])).to(DEVICE),
                (torch.ones(input_ids.shape[0], 1) * torch.tensor([126084])).to(DEVICE),
                image_vq_ids_tensor,
                (torch.ones(input_ids.shape[0], 1) * torch.tensor([126085])).to(DEVICE),
                input_ids
            ], dim=1).long()
            
        else:
            input_ids = input_ids 

        
    except Exception as e:
        yield [("Error tokenizing prompt.", "ERROR")], f"Tokenization error: {e}"
        return

    

    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    x = torch.full((batch_size, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=DEVICE)
    x[:, :prompt_len] = input_ids.clone()

    yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), "Starting generation: Prompt + Initial Masks"

    if gen_length == 0:
         final_text_output = TOKENIZER.batch_decode(x[:,prompt_len:], skip_special_tokens=True)
         yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), final_text_output[0] if final_text_output else ""
         return

    if block_length <= 0 or gen_length % block_length != 0 :
        yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), \
              f"Error: gen_length ({gen_length}) must be divisible by block_length ({block_length}) and block_length > 0."
        return
    num_blocks = gen_length // block_length

    if steps <=0 or steps % num_blocks != 0:
        yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), \
              f"Error: steps ({steps}) must be positive and divisible by num_blocks ({num_blocks}). Steps: {steps}, Num Blocks: {num_blocks}"
        return
    steps_per_block = steps // num_blocks
    
    for num_block_iter in range(num_blocks):
        current_block_start_idx_in_x = prompt_len + num_block_iter * block_length
        current_block_end_idx_in_x = prompt_len + (num_block_iter + 1) * block_length
        
        block_masks_bool_current = torch.zeros_like(x, dtype=torch.bool) 
        block_masks_bool_current[:, current_block_start_idx_in_x:current_block_end_idx_in_x] = \
            (x[:, current_block_start_idx_in_x:current_block_end_idx_in_x] == MASK_ID)

        num_transfer_tokens_for_this_block = get_num_transfer_tokens(
            block_masks_bool_current[:, current_block_start_idx_in_x:current_block_end_idx_in_x],
            steps_per_block
        )

        for i_step_in_block in range(steps_per_block):
            mask_index_global = (x == MASK_ID) 
            
            if cfg_scale > 0.:
                un_x = x.clone()
                # For unconditional pass, mask out the original prompt tokens that are not padding
                # raw_prompt_attention_mask is (B, prompt_len)
                prompt_active_tokens_mask = raw_prompt_attention_mask.bool() # True where actual prompt tokens are
                un_x[:, :prompt_len][prompt_active_tokens_mask] = MASK_ID
                
                x_cfg_input = torch.cat([x, un_x], dim=0)
                # Pass attention_mask for CFG if model expects it, covering both parts
                # For simplicity, not passing explicit attention_mask here; relies on model's internal handling.
                model_output = MODEL(x_cfg_input) 
                logits_cond, logits_uncond = torch.chunk(model_output.logits, 2, dim=0)
                logits = logits_uncond + (cfg_scale + 1) * (logits_cond - logits_uncond)
            else:
                # Not passing explicit attention_mask here; relies on model's internal handling.
                model_output = MODEL(x) 
                logits = model_output.logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0_predicted_tokens = torch.argmax(logits_with_noise, dim=-1) 

            if remasking_strategy == 'low_confidence':
                if DEVICE == "mps":
                    probs = F.softmax(logits.to(torch.float32), dim=-1)
                else:
                    probs = F.softmax(logits.to(torch.float64), dim=-1)
                x0_probs = torch.gather(probs, dim=-1, index=x0_predicted_tokens.unsqueeze(-1)).squeeze(-1) 
            elif remasking_strategy == 'random':
                if DEVICE == "mps":
                    x0_probs = torch.rand(x.shape, device=x.device, dtype=torch.float32)
                else:
                    x0_probs = torch.rand(x.shape, device=x.device, dtype=torch.float64)
            else: 
                yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), f"Error: Unknown remasking strategy '{remasking_strategy}'"
                return

            confidence_for_selection = torch.full_like(x0_probs, -torch.inf) 
            candidate_positions_for_unmasking = mask_index_global & block_masks_bool_current
            confidence_for_selection = torch.where(
                candidate_positions_for_unmasking,
                x0_probs,
                -torch.inf
            )
            
            x0_final_candidates = torch.where(mask_index_global, x0_predicted_tokens, x)

            transfer_indices_bool = torch.zeros_like(x, dtype=torch.bool) 
            num_to_transfer_this_step_batch = num_transfer_tokens_for_this_block[:, i_step_in_block] 

            for j_batch_idx in range(batch_size):
                k_val = min(num_to_transfer_this_step_batch[j_batch_idx].item(), 
                            candidate_positions_for_unmasking[j_batch_idx].sum().item()) # ensure k isn't too large

                if k_val > 0:
                    # Ensure confidence_for_selection[j_batch_idx] is 1D for topk
                    conf_slice = confidence_for_selection[j_batch_idx]
                    if conf_slice.ndim > 1: conf_slice = conf_slice.view(-1) # Should already be 1D from x0_probs
                    
                    # Check if there are enough valid (non -inf) confidences
                    valid_conf_count = (conf_slice > -torch.inf).sum().item()
                    actual_k = min(k_val, valid_conf_count)

                    if actual_k > 0:
                        _, topk_indices_in_x = torch.topk(conf_slice, k=actual_k)
                        transfer_indices_bool[j_batch_idx, topk_indices_in_x] = True
            
            x[transfer_indices_bool] = x0_final_candidates[transfer_indices_bool]

            current_total_step = num_block_iter * steps_per_block + i_step_in_block + 1
            total_overall_steps = num_blocks * steps_per_block
            status_msg = f"Block {num_block_iter+1}/{num_blocks}, Step {i_step_in_block+1}/{steps_per_block} (Total: {current_total_step}/{total_overall_steps})"
            yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), status_msg

    final_generated_ids = x[:, prompt_len:]
    final_text_output = TOKENIZER.batch_decode(final_generated_ids, skip_special_tokens=True)
    
    final_text_str = final_text_output[0] if final_text_output and len(final_text_output) > 0 else ""
    yield get_highlighted_text_tuples(x, input_ids, prompt_len, TOKENIZER, MASK_ID, raw_prompt_attention_mask), final_text_str


css_styles = """
.gradio-container{font-family:'IBM Plex Sans',sans-serif;margin:auto;}
.gr-input {background:#f9f9f9 !important;border:1px solid #e0e0e0 !important;}
.gr-output{background:#f0f0f0 !important;border:1px solid #d0d0d0 !important;}

.highlighted-text span{
    padding:2px 4px;border-radius:4px;margin:1px 2px;display:inline-block;line-height:1.6;
}

footer{display:none !important}

#live-update-scrollable-box {
    max-height: 800px; /* 您可以根据需要调整这个最大高度，例如 '300px', '50vh' 等 */
    overflow-y: auto !important; /* 当内容超出 max-height 时显示垂直滚动条 */
    display: block; /* 确保元素是块级元素，以便 max-height 生效 */

}
#think_btn {
    background-color: #f3f4f6 !important;
    border: 1px solid #d0d0d0 !important;
    color: #111827 !important;
    font-size: 16px !important;
    font-weight: bold !important;
}
#think_btn:hover {
    background-color: #e0e0e0 !important;
    border: 1px solid #c0c0c0 !important;
    color: #222 !important;
}
#think_btn:active {
    background-color: #2563eb !important;
    border: 1px solid #b0b0b0 !important;
    color: white !important;
}
"""


# thinking_mode_t2i = gr.State(False)
def toggle_thinking_mode_lm(current_thinking_mode):
    # print(f"current_thinking_mode: {current_thinking_mode}")
    new_state = not current_thinking_mode
    new_label = "Thinking Mode ✅" if new_state else "Thinking Mode ❌"
    return new_state, gr.update(value=new_label)    

def toggle_thinking_mode_mmu(current_thinking_mode):
    new_state = not current_thinking_mode
    new_label = "Thinking Mode ✅" if new_state else "Thinking Mode ❌"
    return new_state, gr.update(value=new_label)


color_map_config = {
    "MASK": "lightgrey",    
    "GEN": "#DCABFA",       
}

theme = gr.themes.Ocean(
    primary_hue="fuchsia",
)
with gr.Blocks(css=css_styles, theme=theme) as demo:
# with gr.Blocks(css=css_styles, theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky)) as demo:
# with gr.Blocks() as demo:
    thinking_mode_lm = gr.State(False)
    thinking_mode_mmu = gr.State(False)
    gr.Markdown("<h1 style='text-align: center; margin-bottom: 20px;'>MMaDA: Multimodal Large Diffusion Language Models</h1>")
    gr.Markdown("MMaDA is a novel class of multimodal diffusion foundation models designed to achieve superior performance across diverse domains such as textual reasoning, multimodal understanding, and text-to-image generation")
    gr.Markdown("Github: [Gen-Verse/MMaDA](https://github.com/Gen-Verse/MMaDA)")
    gr.Markdown("Paper: [MMaDA: Multimodal Large Diffusion Language Models]()")
    gr.Markdown("### Select Model")
    with gr.Row():
        model_select_radio = gr.Radio(
            label="Select Text Generation Model",
            choices=MODEL_CHOICES,
            value=MODEL_CHOICES[0]
        )
        model_load_status_box = gr.Textbox(
            label="Model Load Status",
            interactive=False, 
            lines=3, 
            max_lines=5
        )

    gr.Markdown("## Part 1. Text Generation")
    with gr.Row():
        with gr.Column(scale=2): 
            prompt_input_box_lm = gr.Textbox(label="Enter your prompt:", lines=3, value="A rectangular prism has a length of 5 units, a width of 4 units, and a height of 3 units. What is the volume of the prism?")
            think_button_lm = gr.Button("🧠 Enable Thinking Mode", elem_id="think_btn")
            with gr.Accordion("Generation Parameters", open=True):
                with gr.Row():
                    gen_length_slider_lm = gr.Slider(minimum=8, maximum=1024, value=512, step=64, label="Generation Length", info="Number of tokens to generate.")
                    steps_slider_lm = gr.Slider(minimum=1, maximum=512, value=256, step=32, label="Total Sampling Steps", info="Must be divisible by (gen_length / block_length).")
                with gr.Row():
                    block_length_slider_lm = gr.Slider(minimum=8, maximum=1024, value=128, step=32, label="Block Length", info="gen_length must be divisible by this.")
                    remasking_dropdown_lm = gr.Dropdown(choices=['low_confidence', 'random'], value='low_confidence', label="Remasking Strategy")
                with gr.Row():
                    cfg_scale_slider_lm = gr.Slider(minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="CFG Scale", info="Classifier-Free Guidance. 0 disables it.")
                    temperature_slider_lm = gr.Slider(minimum=0.0, maximum=2.0, value=1, step=0.05, label="Temperature", info="Controls randomness via Gumbel noise. 0 is deterministic.")
                    

            with gr.Row():
                run_button_ui_lm = gr.Button("Generate Sequence", variant="primary", scale=3)
                clear_button_ui_lm = gr.Button("Clear Outputs", scale=1)

        with gr.Column(scale=3): 
            # gr.Markdown("## Live Generation Process")
            output_visualization_box_lm = gr.HighlightedText(
                label="Live Generation Process",
                show_legend=True,
                color_map=color_map_config,
                combine_adjacent=False, 
                interactive=False,
                elem_id="live-update-scrollable-box",
            )
            # gr.Markdown("## Final Generated Text")
            output_final_text_box_lm = gr.Textbox(label="Final Output", lines=8, interactive=False, show_copy_button=True)



    gr.Examples(
        examples=[
            ["A rectangular prism has a length of 5 units, a width of 4 units, and a height of 3 units. What is the volume of the prism?", 256, 512, 128, 1, 0, "low_confidence"],
            ["Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours?", 256, 512, 64, 1, 0, "low_confidence"]
        ],
        inputs=[prompt_input_box_lm, steps_slider_lm, gen_length_slider_lm, block_length_slider_lm, temperature_slider_lm, cfg_scale_slider_lm, remasking_dropdown_lm],
        outputs=[output_visualization_box_lm, output_final_text_box_lm],
        fn=generate_viz_wrapper_lm,
    )

    gr.Markdown("---")
    gr.Markdown("## Part 2. Multimodal Understanding")
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input_box_mmu = gr.Textbox(
                label="Enter your prompt:",
                lines=3,
                value="Please describe this image in detail."
            )
            think_button_mmu = gr.Button("🧠 Enable Thinking Mode", elem_id="think_btn")
            with gr.Accordion("Generation Parameters", open=True):
                with gr.Row():
                    gen_length_slider_mmu = gr.Slider(minimum=64, maximum=1024, value=512, step=64, label="Generation Length", info="Number of tokens to generate.")
                    steps_slider_mmu = gr.Slider(minimum=1, maximum=512, value=256, step=32, label="Total Sampling Steps", info="Must be divisible by (gen_length / block_length).")
                with gr.Row():
                    block_length_slider_mmu = gr.Slider(minimum=32, maximum=1024, value=128, step=32, label="Block Length", info="gen_length must be divisible by this.")
                    remasking_dropdown_mmu = gr.Dropdown(choices=['low_confidence', 'random'], value='low_confidence', label="Remasking Strategy")
                with gr.Row():
                    cfg_scale_slider_mmu = gr.Slider(minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="CFG Scale", info="Classifier-Free Guidance. 0 disables it.")
                    temperature_slider_mmu = gr.Slider(minimum=0.0, maximum=2.0, value=1, step=0.05, label="Temperature", info="Controls randomness via Gumbel noise. 0 is deterministic.")

            with gr.Row():
                image_upload_box = gr.Image(type="pil", label="Upload Image")
                
            with gr.Row():
                run_button_ui_mmu = gr.Button("Generate Description", variant="primary", scale=3)
                clear_button_ui_mmu = gr.Button("Clear Outputs", scale=1)

        with gr.Column(scale=3):
            gr.Markdown("## Live Generation Process")
            output_visualization_box_mmu = gr.HighlightedText(
                label="Token Sequence (Live Update)",
                show_legend=True,
                color_map=color_map_config,
                combine_adjacent=False,
                interactive=False,
                elem_id="live-update-scrollable-box",
            )
            gr.Markdown("## Final Generated Text")
            output_final_text_box_mmu = gr.Textbox(label="Final Output", lines=8, interactive=False, show_copy_button=True)


    gr.Examples(
        examples=[
            [
                "mmu_validation_2/sunflower.jpg",
                "Please describe this image in detail.",
                256,
                512,
                128,
                1,
                0,
                "low_confidence"
            ],
            [
                "mmu_validation_2/woman.jpg", 
                "Please describe this image in detail.",
                256,
                512,
                128,
                1,
                0,
                "low_confidence"
            ]
        ],
        inputs=[
            image_upload_box,
            prompt_input_box_mmu,
            steps_slider_mmu,
            gen_length_slider_mmu,
            block_length_slider_mmu,
            temperature_slider_mmu,
            cfg_scale_slider_mmu,
            remasking_dropdown_mmu
        ],
        outputs=[output_visualization_box_mmu, output_final_text_box_mmu],
        fn=generate_viz_wrapper,
    )

    gr.Markdown("---")
    gr.Markdown("## Part 3. Text-to-Image Generation")
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input_box_t2i = gr.Textbox(label="Enter your prompt:", lines=3, value="A sea turtle swimming near a coral reef in the ocean, with a clear blue sky and water in the background.")

            with gr.Accordion("Generation Parameters", open=True):
                with gr.Row():
                    steps_slider_t2i = gr.Slider(minimum=5, maximum=100, value=15, step=5, label="Total Sampling Steps", info="Must be divisible by (gen_length / block_length).")
                    guidance_scale_slider_t2i = gr.Slider(minimum=0.0, maximum=7.0, value=3.5, step=0.5, label="Guidance Scale", info="Classifier-Free Guidance. 0 disables it.")
            
            
            with gr.Row():
                scheduler_radio_t2i = gr.Radio(
                    choices=["cosine", "sigmoid", "linear"],
                    value="cosine", 
                    label="Scheduler",
                )

            with gr.Row():
                run_button_ui_t2i = gr.Button("Generate Image", variant="primary", scale=3)
                clear_button_ui_t2i = gr.Button("Clear Outputs", scale=1)


        with gr.Column(scale=3):
            # gr.Markdown("## Live Generation Process")
            output_image_t2i = gr.Image(label="Generated Image", interactive=False, type="pil")
            output_status_t2i = gr.Textbox(label="Generation Status", interactive=False)

    gr.Examples(
        examples=[
            ["A sea turtle swimming near a coral reef in the ocean, with a clear blue sky and water in the background.", 15, 3.5, "cosine"],
            ["A beautiful sunset over a calm ocean, with a few clouds in the sky.", 15, 3.5, "cosine"]
        ],
        inputs=[prompt_input_box_t2i, steps_slider_t2i, guidance_scale_slider_t2i, scheduler_radio_t2i],
        outputs=[output_image_t2i, output_status_t2i],
        fn=generate_viz_wrapper_t2i,
    )
    
    run_button_ui_t2i.click(
        fn=generate_viz_wrapper_t2i,
        inputs=[
            prompt_input_box_t2i,
            steps_slider_t2i,
            guidance_scale_slider_t2i,
            scheduler_radio_t2i
        ],
        outputs=[output_image_t2i, output_status_t2i]
    )

    clear_button_ui_t2i.click(
        fn=lambda: (None, ""),
        inputs=None,
        outputs=[output_image_t2i, output_status_t2i],
        queue=False
    )

    think_button_lm.click(
        fn=toggle_thinking_mode_lm,
        inputs=[thinking_mode_lm],
        outputs=[thinking_mode_lm, think_button_lm]
    )

    think_button_mmu.click(
        fn=toggle_thinking_mode_mmu,
        inputs=[thinking_mode_mmu],
        outputs=[thinking_mode_mmu, think_button_mmu]
    )
            


    def initialize_default_model():
        default_model = "MMaDA-8B-Base"
        result = handle_model_selection_change(default_model)
        return default_model, result

    demo.load(
        fn=initialize_default_model,
        inputs=None,
        outputs=[model_select_radio, model_load_status_box],
        queue=True
    )

    def clear_outputs():
        return None, None, None  # Clear image, visualization, and final text

    clear_button_ui_lm.click(
        fn=clear_outputs,
        inputs=None,
        outputs=[image_upload_box, output_visualization_box_lm, output_final_text_box_lm],
        queue=False
    )
    clear_button_ui_mmu.click(
        fn=clear_outputs,
        inputs=None,
        outputs=[image_upload_box, output_visualization_box_mmu, output_final_text_box_mmu],
        queue=False
    )

    run_button_ui_lm.click(
        fn=generate_viz_wrapper_lm,
        inputs=[
            prompt_input_box_lm,
            steps_slider_lm,
            gen_length_slider_lm,
            block_length_slider_lm,
            temperature_slider_lm,
            cfg_scale_slider_lm,
            remasking_dropdown_lm,
            thinking_mode_lm
        ],
        outputs=[output_visualization_box_lm, output_final_text_box_lm]
    )

    run_button_ui_mmu.click(
        fn=generate_viz_wrapper,
        inputs=[
            image_upload_box,
            prompt_input_box_mmu,
            steps_slider_mmu,
            gen_length_slider_mmu,
            block_length_slider_mmu,
            temperature_slider_mmu,
            cfg_scale_slider_mmu,
            remasking_dropdown_mmu,
            thinking_mode_mmu
        ],
        outputs=[output_visualization_box_mmu, output_final_text_box_mmu]
    )


if __name__ == "__main__":
    print(f"Starting Gradio App. Attempting to use device: {DEVICE}")
    demo.launch(share=True)