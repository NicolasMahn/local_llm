"""Forced tool calls for Gemma 4 and the Mistral models (a vLLM plugin).

Clients force a call with tool_choice "required" or a named tool: MCP
agents that must act, libraries that read structured output from a tool
call. vLLM enforces that with a grammar for most models, but has none for
Gemma 4 (its arguments quote strings with <|"|>, not JSON quotes), so it
quietly treats forcing as "auto" and Gemma may answer in plain text.

The grammar here forces only what matters: Gemma's own call opener with an
allowed tool name. The arguments stay in Gemma's format, which vLLM's
parser reads as usual, so a thinking phase and streaming work unchanged.

Mistral Small 3.2 and Ministral 3 run on the standard tokenizer here (see
ministral-3-3b in model_catalog.py). vLLM then forces a bare JSON array,
which only its pre-v11 Mistral parser reads; these tokenizers are newer,
so the forced call came back as text. The grammar here forces their own
[TOOL_CALLS]NAME[ARGS]{...} instead, with the tool's JSON schema.
"""

import json


def register():
    # Deliberately unguarded, as in local_llm_attention: a vLLM update that
    # moves these must stop the model, not silently drop forced calls again.
    from openai.types.responses import ToolChoiceFunction
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionNamedToolChoiceParam
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser

    stock = Gemma4EngineToolParser.adjust_request
    if getattr(stock, "_local_llm", False):
        return

    def forced_names(request):
        choice = request.tool_choice
        if isinstance(choice, ChatCompletionNamedToolChoiceParam):
            return [choice.function.name]
        if isinstance(choice, ToolChoiceFunction):
            return [choice.name]
        if choice == "required":
            # Chat requests nest the name under "function"; Responses requests do not,
            # and may hold built-in tools without a name.
            functions = (getattr(tool, "function", tool) for tool in request.tools)
            return [function.name for function in functions if getattr(function, "name", None)]
        return []

    def adjust_request(self, request):
        request = stock(self, request)
        names = forced_names(request) if request.tools else []
        if names and request.structured_outputs is None:
            one_call = request.parallel_tool_calls is False
            request.structured_outputs = StructuredOutputsParams(structural_tag=grammar(names, one_call))
        return request

    adjust_request._local_llm = True
    Gemma4EngineToolParser.adjust_request = adjust_request

    from vllm.parser.mistral import MistralParser
    from vllm.utils.mistral import is_mistral_tokenizer

    mistral_stock = MistralParser.adjust_request

    def mistral_adjust_request(self, request):
        own_constraint = getattr(request, "structured_outputs", None) is not None
        request = mistral_stock(self, request)
        if (own_constraint or self._is_pre_v11 or is_mistral_tokenizer(self.model_tokenizer)
                or not getattr(request, "tools", None)):
            return request
        names = set(forced_names(request))
        if names:
            schemas = {tool.function.name: tool.function.parameters or {"type": "object"}
                       for tool in request.tools if tool.function.name in names}
            request.structured_outputs = StructuredOutputsParams(
                structural_tag=mistral_grammar(schemas, request.parallel_tool_calls is False))
        return request

    MistralParser.adjust_request = mistral_adjust_request


def grammar(names, one_call):
    """At least one <|tool_call>call:NAME...<tool_call|>, for one of names, and nothing else."""
    return json.dumps({"type": "structural_tag", "format": {
        "type": "triggered_tags",
        "triggers": ["<|tool_call>"],
        "tags": [{"type": "tag", "begin": f"<|tool_call>call:{name}", "content": {"type": "any_text"},
                  "end": "<tool_call|>"} for name in names],
        "at_least_one": True,
        "stop_after_first": one_call,
    }})


def mistral_grammar(schemas, one_call):
    """At least one [TOOL_CALLS]NAME[ARGS]{...}, for one of the forced tools, and nothing else."""
    return json.dumps({"type": "structural_tag", "format": {
        "type": "triggered_tags",
        "triggers": ["[TOOL_CALLS]"],
        "tags": [{"type": "tag", "begin": f"[TOOL_CALLS]{name}[ARGS]",
                  "content": {"type": "json_schema", "json_schema": schema}, "end": ""}
                 for name, schema in schemas.items()],
        "at_least_one": True,
        "stop_after_first": one_call,
    }})
