import torch

from geo_tf_geo import haversine_km
from geo_tf_score import compose_masked_image, generate_answer_text, parse_generated_coordinate
from methods_helper import upscale


def deletion_heatmap_from_preservation_mask(mask):
    return 1.0 - mask.detach()


def topk_mask(heatmap, percent):
    flat = heatmap.view(-1)
    k = max(1, int(round(flat.numel() * float(percent) / 100.0)))
    _, indices = torch.topk(flat, k=k, largest=True)
    hard = torch.zeros_like(flat)
    hard[indices] = 1.0
    return hard.view_as(heatmap)


def random_mask_like(heatmap, percent):
    flat = heatmap.view(-1)
    k = max(1, int(round(flat.numel() * float(percent) / 100.0)))
    indices = torch.randperm(flat.numel(), device=flat.device)[:k]
    hard = torch.zeros_like(flat)
    hard[indices] = 1.0
    return hard.view_as(heatmap)


def deletion_distance(
    args,
    adapter,
    model,
    tokenizer,
    prompt_inputs,
    image,
    baseline,
    reference_coord,
    deletion_mask,
):
    preservation_mask = 1.0 - deletion_mask
    up_mask = upscale(preservation_mask, image, None)
    image_deleted = compose_masked_image(image, baseline, up_mask)
    text, _, _ = generate_answer_text(
        adapter=adapter,
        model=model,
        tokenizer=tokenizer,
        prompt_inputs=prompt_inputs,
        image=image_deleted,
        max_new_tokens=args.geotf_max_new_tokens,
    )
    try:
        coord = parse_generated_coordinate(text)
    except ValueError:
        return {"text": text, "coord": None, "distance_km": None, "valid": False}
    return {
        "text": text,
        "coord": [coord.lat, coord.lon],
        "distance_km": haversine_km(coord, reference_coord),
        "valid": True,
    }


def evaluate_deletion(
    args,
    adapter,
    model,
    tokenizer,
    prompt_inputs,
    image,
    baseline,
    reference_coord,
    preservation_mask,
):
    heatmap = deletion_heatmap_from_preservation_mask(preservation_mask)
    results = {}
    with torch.no_grad():
        for raw_percent in args.geotf_eval_topk.split(","):
            percent = float(raw_percent.strip())
            ours = deletion_distance(
                args=args,
                adapter=adapter,
                model=model,
                tokenizer=tokenizer,
                prompt_inputs=prompt_inputs,
                image=image,
                baseline=baseline,
                reference_coord=reference_coord,
                deletion_mask=topk_mask(heatmap, percent),
            )
            random_results = []
            for _ in range(args.geotf_random_repeats):
                random_results.append(
                    deletion_distance(
                        args=args,
                        adapter=adapter,
                        model=model,
                        tokenizer=tokenizer,
                        prompt_inputs=prompt_inputs,
                        image=image,
                        baseline=baseline,
                        reference_coord=reference_coord,
                        deletion_mask=random_mask_like(heatmap, percent),
                    )
                )
            key = f"top{int(percent) if percent.is_integer() else percent}"
            valid_random = [item for item in random_results if item["valid"]]
            results[key] = {
                "percent": percent,
                "ours": ours,
                "random_valid_count": len(valid_random),
                "random_mean_km": (
                    sum(item["distance_km"] for item in valid_random) / len(valid_random)
                    if len(valid_random) > 0
                    else None
                ),
                "random": random_results,
            }
    return results
