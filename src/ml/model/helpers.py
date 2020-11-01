"""Helpful functions used by other model-related modules."""

def read_config_file(config_path: str):
    """Reads a model config file and turns it into a dictionary.
    See the README in model-config for more info on what the available config options are.
    """
    types = {
        "ensemble_size": int,
        "epochs": int,
        "input_days": int,
        "training_repeats": int,
        "loss_fn_lambda": float,
        "loss_fn_s": float,
        "training_htc_lower_bound": float,
        "training_htc_upper_bound": float,
        "channels": list,
        "validation_split": float,
        "early_stopping_variable": str,
        "early_stopping_patience": int,
        "ensemble_type": str,
        "static_features": list,
        "scale_inputs": bool,
        "scale_targets": bool,
        "single_output_only": bool,
        "building_types": list,
    }
    options = {}

    with open(config_path, "r") as f:
        for line in f:
            # Ignore any comments
            if line[0] == "#":
                continue

            k, v = [segment.strip() for segment in line.split("=")]

            if k not in types:
                raise ValueError(f"Unrecognized model configuration option: {k}")

            option_type = types[k]
            if option_type in [int, float]:
                v = option_type(v)
            elif option_type == list:
                v = v.split(",")
            elif option_type == bool:
                v = bool(int(v))

            options[k] = v

    return options
