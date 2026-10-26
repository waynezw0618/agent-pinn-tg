"""Agent-PINN: persistent, auditable, LLM-driven structure discovery for the
TG-vortex PINN.

Layers
------
``db`` / ``repo``      SQLite scientific state (candidates, runs, audits,
                       Pareto, provenance) -- the SRHarness idea.
``runner`` / ``jobs``  deterministic subprocess execution of train/postprocess.
``audit``              deterministic physics-constraint auditing (Inspector).
``rag``                hybrid retrieval over the project literature.
``miner``/``distill``  Psi-NN structure distillation and extraction.
``orchestrator``       Creator-Inspector loop with LLM4eD search strategies.
``mcp_server``         MCP (stdio) tool surface for external agents.
``web``                FastAPI read-mostly dashboard.
"""

__version__ = "0.1.0"
