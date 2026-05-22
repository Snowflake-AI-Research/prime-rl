"""Prime-RL ↔ DSS Arctic adapter.

Opt-in integration that delegates training + inference to a remote DSS server
via HTTP through `ArcticRLClient`. Activated by setting `arctic.backend` in
`rl.toml`. When inactive (default), Prime-RL runs its native torchrun+FSDP2
trainer and local vLLM inference unchanged.
"""
