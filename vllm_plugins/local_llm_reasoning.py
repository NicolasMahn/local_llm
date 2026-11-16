"""Keep Olmo's reasoning boundaries correct across ordinary BPE tokens."""

from bisect import bisect_left, bisect_right
from functools import wraps


class _TokenSpans:
    """Incrementally locate a string delimiter without re-decoding the history.

    ASCII think markers remain exact when individual byte-level BPE tokens are
    decoded. Non-ASCII byte fragments can decode to replacement characters, but
    cannot create or remove an ASCII marker. A token crossing a thought boundary
    counts as reasoning if it contains any thought characters.
    """

    def __init__(self, parser):
        self.parser = parser
        self.seen = 0
        self.initial = ""
        self.tail = ""
        self.characters = 0
        self.starts = []
        self.ends = []
        self.closing = None
        self.last_ids = ()

    def feed(self, token_ids):
        parser = self.parser
        texts = getattr(parser, "_local_llm_token_texts", None)
        if texts is None:
            texts = parser._local_llm_token_texts = {}
        for token_id in token_ids:
            self.seen += 1
            if self.closing is not None:
                continue
            if token_id not in texts:
                texts[token_id] = parser.model_tokenizer.decode(
                    [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            text = texts[token_id]
            self.starts.append(self.characters)
            self.initial = (self.initial + text)[:len(parser.think_start)]
            combined = self.tail + text
            found = combined.find(parser.think_end)
            if found >= 0:
                self.closing = self.characters - len(self.tail) + found
            self.characters += len(text)
            self.ends.append(self.characters)
            self.tail = combined[-(len(parser.think_end) - 1):]
        self.last_ids = (self.last_ids + tuple(token_ids))[-8:]

    def count(self):
        opening = len(self.parser.think_start) if self.initial == self.parser.think_start else 0
        end = self.closing if self.closing is not None else self.characters
        if end <= opening:
            return 0
        first_token = bisect_right(self.ends, opening)
        final_token = bisect_left(self.starts, end)
        return final_token - first_token


def _scan(parser, token_ids):
    spans = _TokenSpans(parser)
    spans.feed(token_ids)
    return spans


def _count_reasoning_tokens(parser, token_ids):
    streamed = getattr(parser, "_local_llm_stream_spans", None)
    if streamed is not None and streamed.seen == len(token_ids):
        return streamed.count()
    return _scan(parser, token_ids).count()


def _register_grammar_suffix(olmo_type):
    from vllm.v1.structured_output import StructuredOutputManager

    stock = StructuredOutputManager.trim_reasoning_for_advance
    if getattr(stock, "_local_llm_olmo_reasoning", False):
        return

    @wraps(stock)
    def trim_reasoning_for_advance(self, request, new_token_ids):
        content_ids = stock(self, request, new_token_ids)
        structured = request.structured_output_request
        end = structured.reasoning_end_token_index if structured is not None else None
        first = len(request.all_token_ids) - len(new_token_ids)
        if end is None or not first <= end < len(request.all_token_ids):
            return content_ids
        if content_ids != new_token_ids[end + 1 - first:]:
            return content_ids  # A newer native trim already owns this boundary.
        reasoner = self._get_reasoner(request)
        if not isinstance(reasoner, olmo_type):
            return content_ids
        # A BPE token can finish </think> and begin the JSON object together
        # (89963 decodes to '>{"'). Native trimming discards that entire token.
        # Advance only the grammar through its content suffix; generated IDs
        # and the user-visible response must retain the original tokenization.
        boundary = request.all_token_ids[max(0, end - len(reasoner.think_end) + 1):end + 1]
        text = self.tokenizer.decode(boundary, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        closing = text.find(reasoner.think_end)
        if closing < 0:
            return content_ids
        suffix = text[closing + len(reasoner.think_end):]
        # Native trimming ignores whitespace merged into the closing token.
        # Keep that behavior: xgrammar's JSON root expects the object itself.
        if not suffix or suffix.isspace():
            return content_ids
        return self.tokenizer.encode(suffix, add_special_tokens=False) + content_ids

    trim_reasoning_for_advance._local_llm_olmo_reasoning = True
    StructuredOutputManager.trim_reasoning_for_advance = trim_reasoning_for_advance


def register():
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
    from vllm.reasoning.olmo3_reasoning_parser import Olmo3ReasoningParser

    _register_grammar_suffix(Olmo3ReasoningParser)

    stock_count = Olmo3ReasoningParser.count_reasoning_tokens
    # A future native implementation owns its counting semantics. The current
    # parser inherits a base stub whose zero means "not implemented".
    if stock_count is ReasoningParser.count_reasoning_tokens:
        @wraps(stock_count)
        def count_reasoning_tokens(self, token_ids):
            return _count_reasoning_tokens(self, token_ids)

        count_reasoning_tokens._local_llm_olmo_reasoning = True
        Olmo3ReasoningParser.count_reasoning_tokens = count_reasoning_tokens

    stock_stream = Olmo3ReasoningParser.extract_reasoning_streaming
    if not getattr(stock_stream, "_local_llm_olmo_reasoning", False):
        @wraps(stock_stream)
        def extract_reasoning_streaming(self, previous_text, current_text, delta_text,
                                        previous_token_ids, current_token_ids, delta_token_ids):
            result = stock_stream(self, previous_text, current_text, delta_text,
                                  previous_token_ids, current_token_ids, delta_token_ids)
            spans = getattr(self, "_local_llm_stream_spans", None)
            if spans is None or spans.seen != len(previous_token_ids):
                spans = self._local_llm_stream_spans = _scan(self, previous_token_ids)
            # The API calls count_reasoning_tokens after every parse_delta.
            # Feeding only the supplied delta makes total counting work linear
            # rather than re-decoding an ever-growing history on every token.
            spans.feed(delta_token_ids)
            return result

        extract_reasoning_streaming._local_llm_olmo_reasoning = True
        Olmo3ReasoningParser.extract_reasoning_streaming = extract_reasoning_streaming

    stock_end = Olmo3ReasoningParser.is_reasoning_end
    if not getattr(stock_end, "_local_llm_olmo_reasoning", False):
        @wraps(stock_end)
        def is_reasoning_end(self, input_ids):
            spans = getattr(self, "_local_llm_end_spans", None)
            # Structured decoding extends one sequence. Rewinds or changed
            # suffixes start a new scan; no full-prefix decoding is repeated.
            if (spans is None or len(input_ids) < spans.seen
                    or tuple(input_ids[max(0, spans.seen - 8):spans.seen]) != spans.last_ids):
                spans = self._local_llm_end_spans = _TokenSpans(self)
            spans.feed(input_ids[spans.seen:])
            return spans.closing is not None

        is_reasoning_end._local_llm_olmo_reasoning = True
        Olmo3ReasoningParser.is_reasoning_end = is_reasoning_end

    stock_extract = Olmo3ReasoningParser.extract_reasoning
    if getattr(stock_extract, "_local_llm_olmo_reasoning", False):
        return

    @wraps(stock_extract)
    def extract_reasoning(self, model_output, request):
        if self.think_end in model_output:
            return stock_extract(self, model_output, request)
        # Streaming starts in REASONING even when <think> was supplied by the
        # prompt. A generation cap before </think> must not expose unfinished
        # thought as the final answer only because the request did not stream.
        if model_output.startswith(self.think_start):
            model_output = model_output[len(self.think_start):]
        return model_output or None, None

    extract_reasoning._local_llm_olmo_reasoning = True
    Olmo3ReasoningParser.extract_reasoning = extract_reasoning
