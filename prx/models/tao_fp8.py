import torch
from torchao.float8 import convert_to_float8_training, Float8LinearConfig, precompute_float8_dynamic_scale_for_fsdp


class LinearFP8:
    """Convert Linear layers to FP8LinearLayer from torchao.float8.
    This is inspired by torchtitan float8 implementation
    https://github.com/pytorch/torchtitan/blob/main/docs/float8.md

    Args:
        enable (bool): Whether to enable the conversion.
        layers_substring_filter (list[str] | None): List of substrings to identify which layers to convert.
            If None, all Linear layers will be converted.
        precompute_scale (bool): Whether to precompute the dynamic scale for FSDP. Default is True.
    """

    def __init__(
        self,
        enable: bool = False,
        layers_substring_filter: list[str] | None = None,
        precompute_scale: bool = True,
    ):
        self.enable = enable
        self.layers_substring_filter = layers_substring_filter
        self.precompute_scale = precompute_scale

    def convert_model(self, model: torch.nn.Module) -> None:
        if self.enable is False:
            return

        def module_filter_fn(mod: torch.nn.Module, fqn: str) -> bool:
            if self.layers_substring_filter is None:
                print(f"Convert {fqn} to FP8LinearLayer")
                return True

            if any([substring in fqn for substring in self.layers_substring_filter]):
                print(f"Convert {fqn} to FP8LinearLayer")
                return True

            return False

        config = Float8LinearConfig(enable_fsdp_float8_all_gather=True)

        convert_to_float8_training(model.denoiser, config=config, module_filter_fn=module_filter_fn)

    def add_optimizer_hook(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> None:
        if self.enable is False:
            return

        if self.precompute_scale is True:
            optimizer.register_step_post_hook(lambda *args, **kwargs: precompute_float8_dynamic_scale_for_fsdp(model))
