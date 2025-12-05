import json
import sys

import numpy as np

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")
        if output_config:
            self.output_dtype = pb_utils.triton_string_to_numpy(
                output_config["data_type"]
            )
        else:
            self.output_dtype=np.float32
        pb_utils.Logger.log_error("[PASSTHROUGH] loading");

    def execute(self, requests):
        """This function is called on inference request."""

        responses = []
        for request in requests:
            input_tensors = request.inputs()
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT")
            pb_utils.Logger.log_info(f"[PASSTHROUGH] Received INPUT with shape: {in_0.shape()}")
            output = in_0.as_numpy()
            pb_utils.Logger.log_info(f"[PASSTHROUGH] Sending OUTPUT={output}")
            out_tensor = pb_utils.Tensor("OUTPUT", output.astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        pb_utils.Logger.log_info(f"[PASSTHROUGH] Processed {len(requests)} requests")
        return responses
