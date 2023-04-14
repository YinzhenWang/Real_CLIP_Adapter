from transformers.adapters import AdapterConfig, PrefixTuningConfig, LoRAConfig, IA3Config, MAMConfig, UniPELTConfig

configs = {
    "adapter": AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu"),
    "prefix": PrefixTuningConfig(flat=False, prefix_length=30),
    "LoRA": LoRAConfig(r=8, alpha=16),
    "IA3": IA3Config(),
    "mam": MAMConfig(),
    "unipelt": UniPELTConfig()
}