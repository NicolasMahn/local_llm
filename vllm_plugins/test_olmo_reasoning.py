"""CPU-only checks for Olmo's multi-token thought delimiters and usage."""

import inspect
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

import local_llm_reasoning


BASE = "vllm.reasoning.abs_reasoning_parsers"
OLMO = "vllm.reasoning.olmo3_reasoning_parser"


@pytest.fixture
def parser_type(monkeypatch):
    class Base:
        def count_reasoning_tokens(self, token_ids):
            return 0

    class Tokenizer:
        calls = 0

        def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
            assert skip_special_tokens is False and clean_up_tokenization_spaces is False
            self.calls += 1
            pieces = {1: "<", 2: "think", 3: ">", 4: "</", 5: " </", 6: "think", 7: ">",
                      8: "a", 9: "b", 10: "42", 11: "></", 12: ">\n", 13: ">\n\n",
                      14: "</think>READY", 15: "</thi", 16: "nk>42", 17: "READY",
                      18: '>{"', 19: '{"', 20: "\n"}
            return "".join(pieces[token_id] for token_id in ids)

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return {"READY": [17], '{"': [19], "\n": [20]}[text]

    class Parser(Base):
        think_start, think_end = "<think>", "</think>"
        think_end_first_token_ids = [4, 5]
        think_end_rest_token_ids = [6, 7]

        def __init__(self):
            self.model_tokenizer = Tokenizer()
            self.native_calls = []

        def extract_reasoning(self, text, request):
            self.native_calls.append((text, request))
            match = re.fullmatch(r"(?:<think>)?(.*?)</think>(.*)", text, flags=re.DOTALL)
            return ((match[1] or None, match[2] or None) if match else (None, text))

        def extract_reasoning_streaming(self, *args):
            return args

        def is_reasoning_end(self, ids):
            return False

    class Manager:
        def __init__(self, reasoner):
            self.reasoner = reasoner
            self.tokenizer = getattr(reasoner, "model_tokenizer", None)
            self.native_calls = []

        def _get_reasoner(self, request):
            return self.reasoner

        def trim_reasoning_for_advance(self, request, new_ids):
            self.native_calls.append((request, new_ids))
            structured = request.structured_output_request
            if structured is None or structured.reasoning_end_token_index is None:
                return new_ids
            first = len(request.all_token_ids) - len(new_ids)
            trimmed = structured.reasoning_end_token_index + 1 - first
            return new_ids if trimmed <= 0 else new_ids[trimmed:]

    Parser.manager_type = Manager

    for name, attributes in ((BASE, {"ReasoningParser": Base}),
                             (OLMO, {"Olmo3ReasoningParser": Parser}),
                             ("vllm.v1.structured_output", {"StructuredOutputManager": Manager})):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return Parser


def grammar_request(ids, end):
    return SimpleNamespace(all_token_ids=ids,
        structured_output_request=SimpleNamespace(reasoning_end_token_index=end))


def test_grammar_consumes_only_merged_suffix_without_rewriting_output(parser_type):
    local_llm_reasoning.register()
    parser = parser_type()
    manager = parser_type.manager_type(parser)
    ids = [8, 4, 6, 18, 10]
    request = grammar_request(ids, 3)
    assert manager.trim_reasoning_for_advance(request, [18, 10]) == [19, 10]
    assert manager.trim_reasoning_for_advance(request, ids) == [19, 10]
    assert request.all_token_ids == ids
    # A later step must not prepend the closing token's suffix again.
    assert manager.trim_reasoning_for_advance(request, [10]) == [10]


def test_grammar_native_paths_and_other_reasoning_parsers_are_unchanged(parser_type):
    local_llm_reasoning.register()
    other = parser_type.manager_type(object())
    request = grammar_request([8, 4, 6, 18, 10], 3)
    assert other.trim_reasoning_for_advance(request, [18, 10]) == [10]
    manager = parser_type.manager_type(parser_type())
    plain = [8, 9]
    assert manager.trim_reasoning_for_advance(grammar_request(plain, None), plain) is plain
    no_structured = SimpleNamespace(all_token_ids=plain, structured_output_request=None)
    assert manager.trim_reasoning_for_advance(no_structured, plain) is plain
    for closing, expected in ((7, []), (12, [])):
        req = grammar_request([8, 4, 6, closing], 3)
        assert manager.trim_reasoning_for_advance(req, [closing]) == expected


def test_grammar_registration_is_idempotent(parser_type):
    original = parser_type.manager_type.trim_reasoning_for_advance
    local_llm_reasoning.register()
    patched = parser_type.manager_type.trim_reasoning_for_advance
    local_llm_reasoning.register()
    assert parser_type.manager_type.trim_reasoning_for_advance is patched
    assert inspect.unwrap(patched) is original


def test_future_native_grammar_suffix_is_not_duplicated(parser_type, monkeypatch):
    native = parser_type.manager_type.trim_reasoning_for_advance

    def future(self, request, ids):
        return [19] + native(self, request, ids)

    monkeypatch.setattr(parser_type.manager_type, "trim_reasoning_for_advance", future)
    local_llm_reasoning.register()
    manager = parser_type.manager_type(parser_type())
    assert manager.trim_reasoning_for_advance(grammar_request([8, 4, 6, 18, 10], 3), [18, 10]) == [19, 10]


@pytest.mark.parametrize("ids,expected", [([], 0), ([8, 9], 2), ([1, 2, 3, 8, 9], 2),
    ([8, 9, 4, 6, 7, 10], 2), ([8, 9, 5, 6, 7, 10], 3),
    ([1, 2, 3, 8, 9, 4, 6, 7, 10], 2), ([4, 6, 7], 0),
    ([1, 2, 3, 4, 6, 7], 0), ([1], 1), ([1, 2], 2), ([1, 2, 3], 0),
    ([1, 2, 11, 6, 7], 0), ([1, 2, 11, 6, 7, 10], 0),
    ([8, 4], 2), ([8, 4, 6], 3), ([8, 4, 8, 5, 6, 7, 10], 4),
    ([8, 4, 6, 7, 10, 5, 6, 7], 1), ([8, 4, 6, 12, 10], 1),
    ([8, 4, 6, 13, 10], 1), ([8, 14], 1), ([8, 15, 16], 1)])
def test_multi_token_counting_and_incomplete_delimiters(parser_type, ids, expected):
    local_llm_reasoning.register()
    parser = parser_type()
    assert parser.count_reasoning_tokens(ids) == expected
    calls = parser.model_tokenizer.calls
    assert parser.count_reasoning_tokens(tuple(ids)) == expected
    assert parser.model_tokenizer.calls == calls


def test_stream_counting_and_grammar_boundary_process_each_delta_once(parser_type, monkeypatch):
    local_llm_reasoning.register()
    parser = parser_type()
    ids = [8] * 1024 + [4, 6, 12, 10]
    original_feed = local_llm_reasoning._TokenSpans.feed
    processed = []

    def record_feed(self, new_ids):
        processed.append(len(new_ids))
        return original_feed(self, new_ids)

    monkeypatch.setattr(local_llm_reasoning._TokenSpans, "feed", record_feed)
    for end in range(1, len(ids) + 1):
        result = parser.extract_reasoning_streaming("", "", "", ids[:end - 1], ids[:end], ids[end - 1:end])
        assert result[-1] == ids[end - 1:end]
        count = parser.count_reasoning_tokens(ids[:end])
        assert count == (end if end < 1027 else 1024)
        assert parser.is_reasoning_end(ids[:end]) is (end >= 1027)
    assert sum(processed) == 2 * len(ids)
    assert parser.model_tokenizer.calls == 4
    assert parser.is_reasoning_end([8]) is False


@pytest.mark.parametrize("text,expected", [("", (None, None)), ("<think>", (None, None)),
    ("<thi", ("<thi", None)), ("Thought", ("Thought", None)),
    ("<think>Thought", ("Thought", None)), ("Thought</thi", ("Thought</thi", None))])
def test_unterminated_nonstream_remains_reasoning(parser_type, text, expected):
    local_llm_reasoning.register()
    parser = parser_type()
    assert parser.extract_reasoning(text, object()) == expected
    assert not parser.native_calls


@pytest.mark.parametrize("text", ["Thought</think>42", "<think>Thought</think>42",
    "<think>Thought</think>", "</think>", "<think></think>", "Thought\n</think>\n42"])
def test_complete_nonstream_extract_stays_native(parser_type, text):
    native = parser_type.extract_reasoning
    expected = native(parser_type(), text, None)
    local_llm_reasoning.register()
    parser, request = parser_type(), object()
    assert parser.extract_reasoning(text, request) == expected
    assert parser.native_calls == [(text, request)]


def test_registration_is_idempotent_and_keeps_future_native_counter(parser_type, monkeypatch):
    native_extract = parser_type.extract_reasoning
    native_count = parser_type.count_reasoning_tokens
    local_llm_reasoning.register()
    first_count, first_extract = parser_type.count_reasoning_tokens, parser_type.extract_reasoning
    local_llm_reasoning.register()
    assert parser_type.count_reasoning_tokens is first_count
    assert parser_type.extract_reasoning is first_extract
    assert inspect.unwrap(first_count) is native_count
    assert inspect.unwrap(first_extract) is native_extract
    native_future = lambda self, ids: 17
    monkeypatch.setattr(parser_type, "count_reasoning_tokens", native_future)
    local_llm_reasoning.register()
    assert parser_type.count_reasoning_tokens is native_future


@pytest.fixture
def real_parser(monkeypatch):
    pytest.importorskip("vllm")
    from transformers import AutoTokenizer
    from vllm.reasoning import ReasoningParserManager
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

    checkpoint = Path(os.environ.get("LOCAL_LLM_OLMO_CHECKPOINT", "/model"))
    if not checkpoint.is_dir():
        if "LOCAL_LLM_OLMO_CHECKPOINT" in os.environ:
            pytest.fail(f"Requested Olmo checkpoint is absent: {checkpoint}")
        pytest.skip("Set LOCAL_LLM_OLMO_CHECKPOINT to the local Olmo checkpoint for real-tokenizer checks")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    parser_type = ReasoningParserManager.get_reasoning_parser("olmo3")
    stock = inspect.unwrap(parser_type.extract_reasoning)
    monkeypatch.setattr(parser_type, "extract_reasoning", stock)
    monkeypatch.setattr(parser_type, "count_reasoning_tokens", ReasoningParser.count_reasoning_tokens)
    qwen_type = ReasoningParserManager.get_reasoning_parser("qwen3")
    qwen_count = qwen_type.count_reasoning_tokens
    local_llm_reasoning.register()
    assert qwen_type.count_reasoning_tokens is qwen_count
    assert ReasoningParser.count_reasoning_tokens is not parser_type.count_reasoning_tokens
    assert ReasoningParserManager.get_reasoning_parser("olmo3") is parser_type
    return tokenizer, parser_type(tokenizer), stock


@pytest.mark.parametrize("text,expected", [("", 0), ("test", 1), ("test</think>42", 1),
    ("test </think>42", 2), ("<think>test</think>42", 1),
    ("<think>test </think>42", 2), ("</think>", 0), ("<think></think>", 0),
    ("<think></think>42", 0), ("<think>", 0), ("test</think>\nREADY", 1),
    ("test</think>\n\nREADY", 1), ("test</think> READY", 1), ("test</think>\tREADY", 1),
    ("test\n</think>\nREADY", 2), ("<think>\ntest</think>\nREADY", 2),
    ("</think>\nREADY", 0), ("<think></think>\nREADY", 0)])
def test_real_manager_and_tokenizer_counts(real_parser, text, expected):
    tokenizer, parser, _ = real_parser
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert parser.count_reasoning_tokens(ids) == expected


@pytest.mark.parametrize("suffix", ["\nREADY", "\n\nREADY", " READY", "\tREADY", "", "42"])
def test_real_grammar_boundary_accepts_merged_closing_tokens(real_parser, suffix):
    tokenizer, parser, _ = real_parser
    text = "test\n</think>" + suffix
    ids = tokenizer.encode(text, add_special_tokens=False)
    for end in range(1, len(ids) + 1):
        decoded = tokenizer.decode(ids[:end], skip_special_tokens=False)
        assert parser.is_reasoning_end(ids[:end]) is ("</think>" in decoded)
    assert parser.count_reasoning_tokens(ids) == 2


def test_real_marker_can_be_generated_with_noncanonical_token_splits(real_parser):
    tokenizer, parser, _ = real_parser
    prefix = tokenizer.encode("test", add_special_tokens=False)
    marker = [token for character in "</think>"
              for token in tokenizer.encode(character, add_special_tokens=False)]
    ids = prefix + marker + tokenizer.encode("\nREADY", add_special_tokens=False)
    assert tokenizer.decode(ids) == "test</think>\nREADY"
    assert parser.count_reasoning_tokens(ids) == len(prefix)
    assert parser.is_reasoning_end(ids) is True


@pytest.mark.parametrize("separator", ["", " ", "\n", "\n\n", "\t"])
def test_real_worker_trim_preserves_merged_json_prefix_for_xgrammar(real_parser, separator):
    import xgrammar as xgr
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

    tokenizer, parser, _ = real_parser
    content = separator + '{"job":4711}'
    ids = tokenizer.encode("test</think>" + content, add_special_tokens=False)
    original = ids.copy()
    manager = StructuredOutputManager.__new__(StructuredOutputManager)
    manager.reasoner_cls, manager.tokenizer = type(parser), tokenizer
    manager.enable_in_reasoning = False
    request = SimpleNamespace(all_token_ids=ids, use_structured_output=True,
        structured_output_request=SimpleNamespace(reasoner=parser, reasoning_parser_kwargs=None,
            reasoning_ended=False, reasoning_end_token_index=None))
    assert manager.should_advance(request, new_token_ids=ids)
    end = request.structured_output_request.reasoning_end_token_index
    if separator == "":
        assert ids[end] == 89963
        assert tokenizer.decode([ids[end]]) == '>{"'
    advance = manager.trim_reasoning_for_advance(request, ids)
    grammar_content = content.lstrip() if separator in ("\n", "\n\n") else content
    assert tokenizer.decode(advance) == grammar_content
    assert request.all_token_ids == original and ids == original

    # The same boundary arriving in a one-token step supplies its suffix once;
    # later steps forward native IDs without replaying that suffix.
    boundary = grammar_request(ids[:end + 1], end)
    boundary.structured_output_request.reasoner = parser
    prefix = manager.trim_reasoning_for_advance(boundary, ids[end:end + 1])
    rest = manager.trim_reasoning_for_advance(request, ids[end + 1:])
    assert tokenizer.decode(prefix + rest) == grammar_content
    assert rest == ids[end + 1:]

    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer), max_threads=1)
    compiled = compiler.compile_json_schema(json.dumps({"type": "object", "properties": {
        "job": {"type": "integer"}}, "required": ["job"], "additionalProperties": False}))
    grammar = XgrammarGrammar(vocab_size=len(tokenizer), ctx=compiled,
                              matcher=xgr.GrammarMatcher(compiled))
    if separator in (" ", "\t"):
        # These separators are separate post-close tokens, already constrained
        # by the next-step bitmask. Keep the native trim; the grammar correctly
        # rejects a deliberately unconstrained leading-space/tab fixture.
        native = inspect.unwrap(StructuredOutputManager.trim_reasoning_for_advance)
        assert advance == native(manager, request, ids)
        assert not grammar.matcher.accept_token(advance[0])
        return
    assert grammar.accept_tokens("synthetic-olmo", advance)
    assert grammar.accept_tokens("synthetic-olmo", [tokenizer.eos_token_id])
    assert grammar.is_terminated()


def test_real_stream_counts_merge_closing_newlines_without_reprocessing_history(real_parser, monkeypatch):
    tokenizer, parser, _ = real_parser
    ids = tokenizer.encode("test\n</think>\nREADY", add_special_tokens=False)
    processed = []
    feed = local_llm_reasoning._TokenSpans.feed

    def record(self, new_ids):
        processed.append(len(new_ids))
        return feed(self, new_ids)

    monkeypatch.setattr(local_llm_reasoning._TokenSpans, "feed", record)
    previous = ""
    for end in range(1, len(ids) + 1):
        current = tokenizer.decode(ids[:end])
        parser.extract_reasoning_streaming(previous, current, current[len(previous):],
                                            ids[:end - 1], ids[:end], ids[end - 1:end])
        assert parser.count_reasoning_tokens(ids[:end]) >= 0
        previous = current
    assert sum(processed) == len(ids)
    assert parser.count_reasoning_tokens(ids) == 2


def test_real_finished_ready_diagnostic_has_useful_tokens_after_counting(real_parser):
    tokenizer, parser, _ = real_parser
    profile = Path(os.environ.get("LOCAL_LLM_OLMO_READY_PROFILE", "logs/added-models-2026-11-12/"
                                  "olmo-sampled-parser-v1-diagnostic.json"))
    if not profile.is_file():
        pytest.skip("The retained merged-newline diagnostic is not mounted")

    def rows(value):
        if isinstance(value, dict):
            if value.get("answer") == "\nREADY":
                yield value
            for item in value.values():
                yield from rows(item)
        elif isinstance(value, list):
            for item in value:
                yield from rows(item)

    row = next(rows(json.loads(profile.read_text())))
    text = row["reasoning"] + "</think>" + row["answer"]
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(ids) == text
    # The report keeps decoded fields rather than generated IDs. Re-tokenizing
    # omits the terminal token and need not reproduce generation segmentation.
    assert row["usage"]["completion_tokens"] == 479
    assert len(ids) == 478
    reason_tokens = parser.count_reasoning_tokens(ids)
    assert reason_tokens == len(tokenizer.encode(row["reasoning"], add_special_tokens=False))
    assert 0 < reason_tokens < len(ids)
    assert parser.is_reasoning_end(ids) is True
    assert parser.extract_reasoning(text, None) == (row["reasoning"], row["answer"])


@pytest.mark.parametrize("text", ["<thi", "test</", "test</think", "<think>test</thi"])
def test_real_partial_delimiters_remain_generated_thought(real_parser, text):
    tokenizer, parser, _ = real_parser
    ids = tokenizer.encode(text, add_special_tokens=False)
    opening = tokenizer.encode("<think>", add_special_tokens=False)
    expected = len(ids) - (len(opening) if text.startswith("<think>") else 0)
    assert parser.count_reasoning_tokens(ids) == expected
    expected_text = text.removeprefix("<think>")
    assert parser.extract_reasoning(text, None) == (expected_text, None)


@pytest.mark.parametrize("text", ["test</think>42", "test </think>42", "<think>test</think>42",
    "<think>test</think>", "</think>", "<think></think>"])
def test_real_complete_extraction_is_unchanged(real_parser, text):
    _, parser, stock = real_parser
    assert parser.extract_reasoning(text, None) == stock(parser, text, None)


def test_real_unterminated_profile_cannot_become_nonstream_final(real_parser):
    tokenizer, parser, _ = real_parser
    profile = Path(os.environ.get("LOCAL_LLM_OLMO_PROFILE", "logs/added-models-2026-11-12/"
                                  "olmo-3.1-32b-think-native-profile.json"))
    if not profile.is_file():
        pytest.skip("The retained real truncated profile is not mounted")
    row = next(row for row in json.loads(profile.read_text())["rows"]
               if row["usage"]["completion_tokens"] == 1024)
    assert row["answer"] == "" and row["reasoning"] and row["finish_reason"] == "length"
    ids = tokenizer.encode(row["reasoning"], add_special_tokens=False)
    assert ids and parser.count_reasoning_tokens(ids) == len(ids)
    assert parser.extract_reasoning(row["reasoning"], None) == (row["reasoning"], None)
    assert parser.count_reasoning_tokens([]) == 0
