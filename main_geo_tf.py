import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image

from args_geo_tf import init_geo_tf_args
from geo_tf_eval import deletion_heatmap_from_preservation_mask, evaluate_deletion
from geo_tf_score import generate_geo_tf_reference
from methods_geo_tf import geo_tf_igos_mask_optimization, geo_tf_igos_pp_mask_optimization
from model_adapters import get_model_adapter
from utils import get_initial, get_kernel_size, save_heatmaps, save_images


def load_data(args):
    rows = [json.loads(line) for line in open(args.data_path, "r", encoding="utf-8")]
    return rows


def load_image(args, row):
    image_path = row["image_path"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(args.image_folder, image_path)
    return Image.open(image_path).convert("RGB")


def prepare_image_tensors(args, adapter, model, tokenizer, image_processor, image):
    kernel_size = get_kernel_size(image.size)
    blur = cv2.GaussianBlur(
        np.asarray(image), (kernel_size, kernel_size), sigmaX=kernel_size - 1
    )
    blur = Image.fromarray(blur.astype(np.uint8))
    image_tensor, blur_tensor, prompt_inputs = adapter.prepare_inputs(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image=image,
        baseline_image=blur,
        prompt=args.geotf_prompt,
    )
    return image_tensor, blur_tensor, prompt_inputs


def run_geo_tf(args, adapter, model, tokenizer, image_processor, data):
    output_dir = os.path.join(
        args.output_dir,
        f"GeoTF_WGS84_{args.method}_{args.opt}_L1_{args.L1}_L2_{args.L2}_L3_{args.L3}_"
        f"ig_{args.ig_iter}_iter_{args.iterations}",
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    for index, row in enumerate(data):
        image = load_image(args, row)
        image_tensor, blur_tensor, prompt_inputs = prepare_image_tensors(
            args, adapter, model, tokenizer, image_processor, image
        )
        reference = generate_geo_tf_reference(
            args=args,
            adapter=adapter,
            model=model,
            tokenizer=tokenizer,
            prompt_inputs=prompt_inputs,
            image=image_tensor,
        )
        print(f"reference {index}: {reference['answer_text']}")

        pred_data = {
            "labels": reference["answer_ids"].detach(),
            "boxes": np.array([[0, 0, args.input_size, args.input_size]]),
            "pred_text": reference["answer_text"],
        }
        pred_data = get_initial(
            pred_data,
            args.diverse_k,
            args.init_posi,
            args.init_val,
            args.input_size,
            args.size,
        )

        method = geo_tf_igos_pp_mask_optimization if args.method == "iGOS++" else geo_tf_igos_mask_optimization
        mask, history = method(
            args=args,
            adapter=adapter,
            model=model,
            init_mask=pred_data["init_masks"][0],
            image=image_tensor,
            baseline=blur_tensor,
            prompt_inputs=prompt_inputs,
            reference=reference,
        )

        deletion_heatmap = deletion_heatmap_from_preservation_mask(mask)
        deletion_eval = evaluate_deletion(
            args=args,
            adapter=adapter,
            model=model,
            tokenizer=tokenizer,
            prompt_inputs=prompt_inputs,
            image=image_tensor,
            baseline=blur_tensor,
            reference_coord=reference["coord"],
            preservation_mask=mask,
        )

        sid = str(row.get("id", index)).replace(os.sep, "_")
        sample_dir = os.path.join(output_dir, f"{index}_{sid}")
        os.makedirs(sample_dir, exist_ok=True)
        mask_path = os.path.join(sample_dir, "geo_tf_mask.pt")
        heatmap_path = os.path.join(sample_dir, f"{index}_0_heatmap.jpg")

        torch.save(
            {
                "preservation_mask": mask.cpu(),
                "deletion_heatmap": deletion_heatmap.cpu(),
                "token_weights": reference["token_weights"].cpu(),
            },
            mask_path,
        )
        save_heatmaps(
            mask,
            image_tensor,
            args.size,
            index,
            0,
            sample_dir,
            args.model,
            None,
            None,
            reference["answer_ids"],
        )
        save_images(
            image_tensor,
            index,
            0,
            sample_dir,
            None,
            reference["answer_ids"],
            pred_data,
            text=args.geotf_prompt,
        )

        result = {
            "image_id": sid,
            "reference_answer": reference["answer_text"],
            "reference_coord": [reference["coord"].lat, reference["coord"].lon],
            "coordinate_span": reference["coordinate_span"],
            "lat_span": reference["lat_span"],
            "lon_span": reference["lon_span"],
            "char_meta": reference["char_meta"],
            "token_meta": reference["token_meta"],
            "heatmap_path": heatmap_path,
            "mask_tensor_path": mask_path,
            "optimization_log": history,
            "deletion_eval": deletion_eval,
        }
        with open(os.path.join(sample_dir, "geo_tf_result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"finished {index}: reference={reference['answer_text']}")


if __name__ == "__main__":
    args = init_geo_tf_args()
    torch.manual_seed(args.manual_seed)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)

    adapter = get_model_adapter(args.model)
    tokenizer, model, image_processor, _ = adapter.load_model(
        args.model_path,
        args.model_base,
        args.torch_dtype,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.gradient_checkpointing = True
    if hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing"):
        model.model.gradient_checkpointing = True
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()

    data = load_data(args)
    run_geo_tf(args, adapter, model, tokenizer, image_processor, data)
