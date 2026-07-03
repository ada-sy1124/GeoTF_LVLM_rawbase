import torch
import torch.nn.functional as F

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_vicuna_v1
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from mgm.mm_utils import process_images_mgm
from mgm.model.builder import load_pretrained_model_mgm


def load_model(model_path, model_base, torch_dtype_name):
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[torch_dtype_name]
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model_mgm(
        model_path,
        model_base,
        model_name,
    )
    if model.dtype != torch_dtype:
        model.to(dtype=torch_dtype)
    return tokenizer, model, image_processor, context_len


def build_prompt(prompt):
    conv = conv_vicuna_v1.copy()
    qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def prepare_inputs(model, tokenizer, image_processor, image, baseline_image, prompt):
    image_tensor = process_images_mgm([image], image_processor, model.config)[0]
    baseline_tensor = process_images_mgm([baseline_image], image_processor, model.config)[0]
    device = model.device
    dtype = model.dtype
    image_tensor = image_tensor.unsqueeze(0).detach().to(device=device, dtype=dtype)
    baseline_tensor = baseline_tensor.unsqueeze(0).detach().to(device=device, dtype=dtype)

    prompt_text = build_prompt(prompt)
    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    prompt_inputs = {"input_ids": input_ids}
    return image_tensor, baseline_tensor, prompt_inputs


def generate_coordinate_ids(model, prompt_inputs, image, temperature, top_p, max_new_tokens):
    generation_kwargs = {
        "images": image,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["do_sample"] = False

    old_max_length = model.generation_config.max_length
    model.generation_config.max_length = None
    try:
        output_ids = model.generate(prompt_inputs["input_ids"], **generation_kwargs)
    finally:
        model.generation_config.max_length = old_max_length
    return output_ids


def postprocess_coordinate_answer(tokenizer, answer_ids, raw_text):
    coord_text = raw_text.strip()
    if coord_text.startswith("[") and coord_text.endswith("]"):
        coord_text = coord_text[1:-1].strip()
    coord_ids = tokenizer(
        coord_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(answer_ids.device)
    return coord_text, coord_ids


def teacher_forced_token_logps(model, prompt_inputs, answer_ids, image, temperature):
    input_ids = torch.cat((prompt_inputs["input_ids"], answer_ids), dim=1)
    inputs, _, _, _, inputs_embeds, labels = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        None,
        None,
        None,
        image,
    )
    position_ids = torch.arange(inputs_embeds.shape[1]).unsqueeze(0).to(inputs_embeds.device)
    attention_mask = torch.ones_like(position_ids).to(inputs_embeds.device)
    outputs = model(
        input_ids=inputs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = outputs["logits"]
    if not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite logits in MGM teacher-forced coordinate logprob")
    logits = logits[:, -answer_ids.shape[-1] - 1 : -1, :].float() / float(temperature)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(2, answer_ids.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(token_logps).all():
        raise FloatingPointError("non-finite token logprob in MGM teacher-forced coordinate logprob")
    return token_logps


def teacher_forced_logprob(model, prompt_inputs, answer_ids, image, temperature):
    token_logps = teacher_forced_token_logps(
        model=model,
        prompt_inputs=prompt_inputs,
        answer_ids=answer_ids,
        image=image,
        temperature=temperature,
    )
    return token_logps.sum()
