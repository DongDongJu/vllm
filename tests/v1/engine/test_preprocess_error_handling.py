# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import asdict
from pathlib import Path

import openai
import pytest
import torch.cuda

from tests.utils import RemoteOpenAIServer
from vllm import LLM, SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.exceptions import VLLMValidationError
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


class RequestValidationConnector(ExampleConnector):
    def validate_request(self, request: Request) -> None:
        if request.cache_salt is not None:
            raise VLLMValidationError(
                "connector rejects salted requests", parameter="cache_salt"
            )


def validation_config(path: Path, multi: bool) -> KVTransferConfig:
    child = {
        "kv_connector": "RequestValidationConnector",
        "kv_connector_module_path": __name__,
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {"shared_storage_path": str(path)},
    }
    if not multi:
        return KVTransferConfig(**child)
    return KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_consumer",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "ExampleConnector",
                    "kv_role": "kv_consumer",
                    "kv_connector_extra_config": {"shared_storage_path": str(path)},
                },
                child,
            ]
        },
    )


@pytest.mark.parametrize("multi", [False, True])
def test_connector_request_validation(tmp_path: Path, multi: bool) -> None:
    """A real connector and its MultiConnector parent reject the same input."""
    config = VllmConfig(
        cache_config=CacheConfig(block_size=16),
        device_config=DeviceConfig("cpu"),
        scheduler_config=SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            disable_hybrid_kv_cache_manager=True,
        ),
        kv_transfer_config=validation_config(tmp_path, multi),
    )
    connector = KVConnectorFactory.create_connector(
        config,
        KVConnectorRole.SCHEDULER,
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
    )
    request = Request(
        request_id="rejected",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        cache_salt="unsupported",
    )
    try:
        with pytest.raises(VLLMValidationError, match="connector rejects") as error:
            connector.validate_request(request)
        assert error.value.parameter == "cache_salt"
        assert request.cache_salt == "unsupported"
        assert list(request.all_token_ids) == [1, 2]
    finally:
        connector.shutdown()


@pytest.mark.parametrize("multi", [False, True])
def test_rejected_connector_request_keeps_engine_alive(
    tmp_path: Path, multi: bool
) -> None:
    """A rejected request leaves the engine able to serve a valid request."""
    with RemoteOpenAIServer(
        MODEL_NAME,
        [
            "--max-model-len",
            "128",
            "--max-num-seqs",
            "1",
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.1",
            "--enable-prefix-caching",
            "--kv-transfer-config",
            json.dumps(asdict(validation_config(tmp_path, multi))),
        ],
    ) as server:
        client = server.get_client()
        with pytest.raises(openai.BadRequestError, match="connector rejects"):
            client.completions.create(
                model=MODEL_NAME,
                prompt="Hello",
                max_tokens=1,
                temperature=0,
                extra_body={"cache_salt": "unsupported"},
            )
        response = client.completions.create(
            model=MODEL_NAME,
            prompt="Hello",
            max_tokens=1,
            temperature=0,
            extra_body={"ignore_eos": True},
        )
        assert response.choices[0].finish_reason in ("stop", "length")
        assert response.usage is not None and response.usage.completion_tokens == 1


def test_preprocess_error_handling(monkeypatch: pytest.MonkeyPatch):
    """Test that preprocessing errors are handled gracefully."""
    if current_platform.is_rocm() or current_platform.is_xpu():
        pytest.skip(
            "Skipped on ROCm/XPU: this test only works with 'fork', "
            "but ROCm/XPU uses 'spawn'."
        )

    assert not torch.cuda.is_initialized(), (
        "fork needs to be used for the engine "
        "core process and this isn't possible if cuda is already initialized"
    )

    # Store original method to call for non-failing requests
    original_preprocess = EngineCore.preprocess_add_request

    # Monkeypatch to make preprocess_add_request raise an exception
    # only for requests with "FAIL" in the first token
    def conditional_failing_preprocess(self, request: EngineCoreRequest):
        # Fail if the first token id is 333
        if request.prompt_token_ids and request.prompt_token_ids[0] == 333:
            raise ValueError("Simulated preprocessing error!")
        return original_preprocess(self, request)

    monkeypatch.setattr(
        EngineCore, "preprocess_add_request", conditional_failing_preprocess
    )

    llm = LLM(model=MODEL_NAME)

    # Create a failing request by crafting a request with an invalid token
    # We need to use a direct approach since LLM.generate tokenizes for us
    from vllm.inputs import TokensPrompt

    # This should raise an exception due to the preprocessing failure
    # Special token id to trigger the failure
    failing_prompt = TokensPrompt(prompt_token_ids=[333])
    outputs = llm.generate(failing_prompt, SamplingParams(max_tokens=10))  # type: ignore
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 0
    assert outputs[0].finished
    assert outputs[0].outputs[0].finish_reason == "error"

    # Verify the engine is still functional with a normal request
    outputs = llm.generate("Hello, my name is", SamplingParams(max_tokens=10))
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0
    assert outputs[0].outputs[0].finish_reason in ("stop", "length")
