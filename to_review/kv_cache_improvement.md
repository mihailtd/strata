This could be something we can improve on - like find a way to fix the downsides and get the upside of quantization

# -------------------------------------------------------------------------
    # KV CACHE PRECISION -- Action 1.1: Ban INT4 KV cache (40k token cliff).
    # BF16 is the standard production default; INT8 is accepted for long context.
    # -------------------------------------------------------------------------
    KV_CACHE_DTYPE: str = "bfloat16"

# -------------------------------------------------------------------------
