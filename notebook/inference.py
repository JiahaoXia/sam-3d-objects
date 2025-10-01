import os

# not ideal to put that here
os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
os.environ["LIDRA_SKIP_INIT"] = "true"

import sys
from typing import Union, Optional
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from hydra.utils import instantiate

import sam3d_objects  # REMARK(Pierre) : do not remove this import
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

__all__ = ["Inference"]

# concatenate "li" + "dra" to skip the automated string replacement
if "li" + "dra" not in sys.modules:
    sys.modules["li" + "dra"] = sam3d_image


class Inference:
    # public facing inference API
    # only put publicly exposed arguments here
    def __init__(self, config_file: str, compile: bool = False):

        # load inference pipeline
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"  # overwrite to disable nvdiffrast
        config.compile_model = compile
        config.workspace_dir = os.path.dirname(__file__)
        self._pipeline: InferencePipelinePointMap = instantiate(config)

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[Union[None, Image.Image, np.ndarray]] = None,
        seed: Optional[int] = None,
        return_poses: bool = False,
        pointmap=None,  # TODO(Pierre) : add pointmap type
    ) -> dict:
        # enable or disable layout model
        if return_poses:
            self._pipeline.use_layout_result = True
            self._pipeline.force_shape_in_layout = True
        else:
            self._pipeline.use_layout_result = False
            self._pipeline.force_shape_in_layout = False

        return self._pipeline.run(
            image,
            mask,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
        )
