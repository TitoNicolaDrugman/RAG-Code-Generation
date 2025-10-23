def _qcfg_to_dict(cfg):
    """
    Convert a BitsAndBytesConfig object into a serializable dictionary.
    Useful for caching metadata.
    """
    return {
        "load_in_4bit": cfg.load_in_4bit,
        "bnb_4bit_quant_type": cfg.bnb_4bit_quant_type,
        "bnb_4bit_compute_dtype": str(cfg.bnb_4bit_compute_dtype),
        "bnb_4bit_use_double_quant": cfg.bnb_4bit_use_double_quant,
    }
