"""CPU checks for exact audio packing and the native fallback contract."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import local_llm_audio as audio


@pytest.mark.parametrize("length", [1, 7, 8, 9, 99, 100, 101, 104, 199, 200, 201, 799, 800, 801, 1600, 18017])
def test_chunked_convolution_lengths_match_the_native_recording_formula(length):
    layout = audio.audio_layout([length])
    # Native helper computes 13 outputs per full 100-frame chunk, then
    # applies three stride-two convolutions to the final partial chunk.
    assert layout.output_lengths == ((length // 100) * 13 + (length % 100 + 7) // 8,)
    assert sum(layout.chunk_lengths) == length
    assert layout.cu_seqlens[-1] == layout.output_lengths[0]
    assert len(layout.valid_rows) == layout.output_lengths[0]


def test_packing_matches_row_major_boolean_selection_for_mixed_recordings():
    layout = audio.audio_layout([1, 101, 99, 800])
    rows = [row for row in range(len(layout.chunk_lengths) * layout.padded_conv_length)
            if row % layout.padded_conv_length < layout.conv_lengths[row // layout.padded_conv_length]]
    assert layout.valid_rows == tuple(rows)
    assert layout.output_lengths == (1, 14, 13, 104)
    assert layout.cu_seqlens == (0, 1, 15, 28, 132)
    assert layout.max_seqlen == 104


def test_attention_windows_follow_recording_boundaries_including_short_batches():
    assert audio.audio_layout([8, 9]).cu_seqlens == (0, 1, 3)
    layout = audio.audio_layout([801, 1600])
    assert layout.output_lengths == (105, 208)
    assert layout.cu_seqlens == (0, 104, 105, 209, 313)


@pytest.mark.parametrize("lengths", [[], [0], [-1], [1.5], [True], [1, 0]])
def test_invalid_lengths_are_rejected_before_gpu_work(lengths):
    with pytest.raises(ValueError):
        audio.audio_layout(lengths)


@pytest.mark.parametrize("window,inference", [(0, 800), (100, 0), (100, 799), (True, 800)])
def test_invalid_window_metadata_is_rejected(window, inference):
    with pytest.raises(ValueError):
        audio.audio_layout([100], window=window, inference_window=inference)


@dataclass
class TensorMetadata:
    shape: tuple
    dtype: object
    device: object
    contiguous: bool = True
    values: tuple = ()

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def tolist(self):
        assert self.device.type == "cpu", "The fast path must never read GPU lengths"
        return list(self.values)

    def to(self, dtype):
        return self


@pytest.fixture
def supported(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    class Tower:
        training = False
        num_mel_bins = 128
        n_window = 50
        n_window_infer = 800
        attn_backend = SimpleNamespace(name="FLASH_ATTN")
        conv_chunksize = 500
        dtype = torch.bfloat16
        device = torch.device("cuda:0")
    tower = Tower()
    features = TensorMetadata((128, 201), torch.float32, tower.device)
    lengths = TensorMetadata((2,), torch.int64, torch.device("cpu"), values=(100, 101))
    return torch, tower, {"input_features": features, "audio_feature_lengths": lengths}


def test_supported_pipeline_keeps_cpu_lengths_through_output_splitting(supported, monkeypatch):
    _, tower, data = supported
    result = Mock()
    encode = Mock(return_value=result)
    monkeypatch.setattr(audio, "encode_audio", encode)
    stock = Mock()
    wrapped = audio._prepare_audio(stock, type(tower))
    wrapped(SimpleNamespace(audio_tower=tower), data)
    stock.assert_not_called()
    assert encode.call_args.args[2].output_lengths == (13, 14)
    result.split.assert_called_once_with((13, 14))


@pytest.mark.parametrize("change", ["capture", "compile", "gpu_lengths", "noncontiguous", "wrong_total", "empty", "float_lengths", "training", "fp16_tower", "other_window", "other_tower", "other_backend", "missing_lengths"])
def test_unsupported_pipeline_forwards_the_original_input(supported, monkeypatch, change):
    torch, tower, data = supported
    features, lengths = data["input_features"], data["audio_feature_lengths"]
    if change == "capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    elif change == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif change == "gpu_lengths":
        lengths.device = tower.device
    elif change == "noncontiguous":
        features.contiguous = False
    elif change == "wrong_total":
        lengths.values = (100, 100)
    elif change == "empty":
        lengths.values = (100, 0)
    elif change == "float_lengths":
        lengths.dtype = torch.float32
    elif change == "training":
        tower.training = True
    elif change == "fp16_tower":
        tower.dtype = torch.float16
    elif change == "other_window":
        tower.n_window = 40
    elif change == "other_backend":
        tower.attn_backend = SimpleNamespace(name="TORCH_SDPA")
    elif change == "missing_lengths":
        data["audio_feature_lengths"] = None
    stock = Mock(return_value=object())
    encode = Mock()
    monkeypatch.setattr(audio, "encode_audio", encode)
    expected_type = str if change == "other_tower" else type(tower)
    wrapped = audio._prepare_audio(stock, expected_type)
    owner = SimpleNamespace(audio_tower=tower)
    assert wrapped(owner, data) is stock.return_value
    stock.assert_called_once_with(owner, data)
    encode.assert_not_called()


def test_first_use_logging_occurs_only_after_the_fast_path_is_selected(supported, monkeypatch):
    torch, tower, data = supported
    monkeypatch.setattr(audio, "encode_audio", Mock(return_value=Mock()))
    logger = Mock()
    wrapped = audio._prepare_audio(Mock(), type(tower), logger)
    owner = SimpleNamespace(audio_tower=tower)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    wrapped(owner, data)
    logger.info_once.assert_not_called()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    wrapped(owner, data)
    logger.info_once.assert_called_once_with("Using local Omni audio CPU layout and fixed row packing", scope="global")


def test_source_fingerprints_fail_closed_for_changed_or_missing_code(monkeypatch):
    import inspect

    assert not audio.native_contract_matches(lambda: None, lambda: None, lambda: None)
    monkeypatch.setattr(inspect, "getsource", Mock(side_effect=OSError("source unavailable")))
    assert not audio.native_contract_matches(None, None, None)


@pytest.fixture
def registration(monkeypatch):
    import sys
    from types import ModuleType

    class Tower:
        def forward(self):
            pass

    class Mixin:
        def _process_audio_input(self, audio_input):
            pass

    logger = Mock()
    def install(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    install("vllm.platforms", current_platform=SimpleNamespace(is_rocm=lambda: True))
    install("vllm.platforms.rocm", on_gfx1151=lambda: True)
    install("vllm.logger", init_logger=lambda name: logger)
    install("vllm.model_executor.models.qwen3_omni_moe_thinker",
            Qwen3OmniMoeAudioEncoder=Tower, Qwen3OmniMoeConditionalGenerationMixin=Mixin,
            _get_feat_extract_output_lengths=lambda x: x)
    monkeypatch.setattr(audio, "native_contract_matches", lambda *args: True)
    return Tower, Mixin, logger


def test_registration_is_idempotent_and_preserves_the_native_entry(registration):
    _, mixin, _ = registration
    stock = mixin._process_audio_input
    audio.register()
    wrapped = mixin._process_audio_input
    assert wrapped._local_llm_host_audio
    assert wrapped.__wrapped__ is stock
    audio.register()
    assert mixin._process_audio_input is wrapped


def test_registration_warns_and_retains_native_when_upstream_changed(registration, monkeypatch):
    _, mixin, logger = registration
    stock = mixin._process_audio_input
    monkeypatch.setattr(audio, "native_contract_matches", lambda *args: False)
    audio.register()
    assert mixin._process_audio_input is stock
    logger.warning.assert_called_once()
