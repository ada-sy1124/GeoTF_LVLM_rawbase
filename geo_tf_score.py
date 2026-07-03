import torch

from geo_tf_geo import build_token_weights, extract_first_coordinate
from methods_helper import phi


def strip_edge_special_tokens(token_ids, tokenizer):
    ids = token_ids
    if ids.dim() == 2:
        ids = ids[0]
    ids = ids.clone()
    if ids.numel() > 0 and tokenizer.bos_token_id is not None and int(ids[0]) == int(tokenizer.bos_token_id):
        ids = ids[1:]
    if ids.numel() > 0 and tokenizer.eos_token_id is not None and int(ids[-1]) == int(tokenizer.eos_token_id):
        ids = ids[:-1]
    return ids.unsqueeze(0)


def compose_masked_image(image, baseline, up_mask):
    return phi(image, baseline, up_mask).to(dtype=image.dtype)


def extract_answer_ids(output_ids, prompt_input_ids, tokenizer):
    ids = output_ids.sequences if hasattr(output_ids, "sequences") else output_ids
    prompt_len = prompt_input_ids.shape[1]
    if ids.shape[1] >= prompt_len and torch.equal(ids[:, :prompt_len], prompt_input_ids):
        ids = ids[:, prompt_len:]
    return strip_edge_special_tokens(ids, tokenizer).to(prompt_input_ids.device)


def generate_answer_text(
    adapter,
    model,
    tokenizer,
    prompt_inputs,
    image,
    max_new_tokens,
):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output_ids = adapter.generate_coordinate_ids(
                model=model,
                prompt_inputs=prompt_inputs,
                image=image,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
            )
    finally:
        if was_training:
            model.train()
    answer_ids = extract_answer_ids(
        output_ids,
        prompt_inputs["input_ids"],
        tokenizer,
    )
    raw_answer_text = tokenizer.decode(
        answer_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    answer_text, answer_ids = adapter.postprocess_coordinate_answer(
        tokenizer,
        answer_ids,
        raw_answer_text,
    )
    return answer_text, answer_ids, output_ids


def generate_geo_tf_reference(args, adapter, model, tokenizer, prompt_inputs, image):
    answer_text, answer_ids, output_ids = generate_answer_text(
        adapter=adapter,
        model=model,
        tokenizer=tokenizer,
        prompt_inputs=prompt_inputs,
        image=image,
        max_new_tokens=args.geotf_max_new_tokens,
    )
    weight_info = build_token_weights(tokenizer, answer_ids, answer_text)
    return {
        "answer_text": answer_text,
        "answer_ids": answer_ids,
        "raw_output_ids": output_ids,
        "coord": weight_info["coordinate"],
        "coordinate_span": weight_info["coordinate_span"],
        "lat_span": weight_info["lat_span"],
        "lon_span": weight_info["lon_span"],
        "char_weights": weight_info["char_weights"],
        "char_meta": weight_info["char_meta"],
        "token_weights": weight_info["token_weights"].to(answer_ids.device),
        "token_meta": weight_info["token_meta"],
    }


def weighted_teacher_forced_nll(
    adapter,
    model,
    prompt_inputs,
    answer_ids,
    token_weights,
    image,
    logit_temperature,
):
    token_logps = adapter.teacher_forced_token_logps(
        model=model,
        prompt_inputs=prompt_inputs,
        answer_ids=answer_ids,
        image=image,
        temperature=logit_temperature,
    )
    weights = token_weights.to(device=token_logps.device, dtype=token_logps.dtype)
    return -(token_logps * weights).sum()


def geo_tf_interval_nll(
    args,
    adapter,
    model,
    prompt_inputs,
    image,
    baseline,
    up_mask,
    reference,
):
    intervals = torch.linspace(
        1.0 / args.ig_iter,
        1.0,
        args.ig_iter,
        device=up_mask.device,
        dtype=up_mask.dtype,
    )
    loss = up_mask.sum() * 0.0
    for alpha in intervals:
        local_image = compose_masked_image(image, baseline, up_mask * alpha.view(1, 1, 1, 1))
        loss = loss + weighted_teacher_forced_nll(
            adapter=adapter,
            model=model,
            prompt_inputs=prompt_inputs,
            answer_ids=reference["answer_ids"],
            token_weights=reference["token_weights"],
            image=local_image,
            logit_temperature=args.geotf_logit_temperature,
        )
    return loss / float(args.ig_iter)


def geo_tf_interval_nll_backward(
    args,
    adapter,
    model,
    prompt_inputs,
    image,
    baseline,
    make_up_mask,
    reference,
    grad_scale,
):
    intervals = torch.linspace(
        1.0 / args.ig_iter,
        1.0,
        args.ig_iter,
        device=image.device,
        dtype=torch.float32,
    )
    total_loss = torch.zeros((), device=image.device, dtype=torch.float32)
    for alpha in intervals:
        up_mask = make_up_mask()
        local_image = compose_masked_image(
            image,
            baseline,
            up_mask * alpha.to(dtype=up_mask.dtype).view(1, 1, 1, 1),
        )
        local_loss = weighted_teacher_forced_nll(
            adapter=adapter,
            model=model,
            prompt_inputs=prompt_inputs,
            answer_ids=reference["answer_ids"],
            token_weights=reference["token_weights"],
            image=local_image,
            logit_temperature=args.geotf_logit_temperature,
        )
        total_loss = total_loss + local_loss.detach()
        (local_loss * (float(grad_scale) / float(args.ig_iter))).backward()
    return total_loss / float(args.ig_iter)


def parse_generated_coordinate(text):
    return extract_first_coordinate(text).coord
