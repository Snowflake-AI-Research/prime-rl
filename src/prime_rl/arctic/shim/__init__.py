"""OpenAI-compatible shim.

Translates OpenAI chat completions requests from Prime-RL's orchestrator
(via verifiers) into DSS /generate calls. Subprocess isolates the tokenizer
and survives independently of the orchestrator.
"""
