def get_model_adapter(model_name):
    if model_name == "llava":
        from model_adapters import llava

        return llava
    if model_name == "mgm":
        from model_adapters import mgm

        return mgm
    raise ValueError(f"unsupported model: {model_name}")
