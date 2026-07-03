import argparse


GEO_TF_PROMPT = (
    """<context>
I don't remember where the photo was taken. Could you help me find it? Estimate the latitude and longitude. 
If you are not sure about the specific location, you MUST give your best possible candidate without asking any further questions.
</context>

<formatting_rules>
CRITICAL: Do NOT use brackets, parentheses, or any other characters. Output ONLY the coordinate string, no other words.
You MUST output EXACTLY matching this format:
{latitude}, {longitude}
</formatting_rules>"""
)


def init_geo_tf_args():
    parser = argparse.ArgumentParser(description="GeoTF LVLM geolocation attribution.")

    parser.add_argument("--model", choices=["llava", "mgm"], default="llava")
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--input_size", type=int, default=336)
    parser.add_argument("--method", choices=["iGOS+", "iGOS++"], default="iGOS+")
    parser.add_argument("--opt", choices=["LS", "NAG"], default="NAG")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--ig_iter", type=int, default=10)
    parser.add_argument("--lr", type=float, default=10.0)
    parser.add_argument("--momentum", type=int, default=5)
    parser.add_argument("--L1", type=float, default=1.0)
    parser.add_argument("--L2", type=float, default=0.1)
    parser.add_argument("--L3", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--diverse_k", type=int, default=1)
    parser.add_argument("--init_posi", type=int, default=0)
    parser.add_argument("--init_val", type=float, default=0.0)
    parser.add_argument("--manual_seed", type=int, default=0)

    parser.add_argument("--geotf_prompt", type=str, default=GEO_TF_PROMPT)
    parser.add_argument("--geotf_max_new_tokens", type=int, default=64)
    parser.add_argument("--geotf_logit_temperature", type=float, default=1.0)
    parser.add_argument("--geotf_eval_topk", type=str, default="5,10,20,30")
    parser.add_argument("--geotf_random_repeats", type=int, default=10)

    return parser.parse_args()
