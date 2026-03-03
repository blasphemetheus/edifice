# Agent Swarm Patterns — Research Notes

## Overview

Agent swarm patterns are multi-agent systems where multiple specialized model
instances coordinate to solve complex tasks. Unlike single-model inference,
swarms decompose problems across agents that communicate, maintain state, and
can use external tools.

This is fundamentally different from Edifice's existing `MixtureOfAgents`,
which is a single-turn parallel ensemble (N proposers → 1 aggregator, one
forward pass, no inter-agent communication). True swarm patterns operate over
multiple turns with persistent state and coordination protocols.

## Landscape Survey

### Academic Foundations

**Multi-Agent Debate** (Du et al., 2023, arXiv:2305.14325)
- Multiple LLM instances generate independent responses, then critique each other's answers over multiple rounds
- Converges to more accurate answers than single-agent, especially on reasoning tasks
- Simple protocol: generate → share → critique → regenerate

**AutoGen** (Wu et al., Microsoft, 2023, arXiv:2308.08155)
- Framework for multi-agent conversation with customizable agent roles
- Agents can be LLMs, tool executors, or humans
- Key patterns: two-agent chat, group chat with speaker selection, nested conversations
- Introduced "conversable agent" abstraction

**CAMEL** (Li et al., 2023, arXiv:2303.17760)
- Role-playing framework: "inception prompting" assigns roles to two agents
- Coordinator (AI assistant) and user (AI user) collaborate via structured dialogue
- Showed emergent task decomposition without explicit planning

**MetaGPT** (Hong et al., 2023, arXiv:2308.00352)
- Software-engineering multi-agent system with explicit role assignment
- Product Manager → Architect → Engineer → QA, with structured outputs (PRDs, design docs, code)
- Key insight: structured intermediate artifacts prevent information loss between agents

**Swarm Intelligence** (classical, Bonabeau et al., 1999)
- Biological inspiration: ant colonies, bee swarms, bird flocking
- Decentralized control, local interactions, emergent global behavior
- Relevant patterns: stigmergy (indirect communication via environment), pheromone trails (shared state)

### Production Systems (2024-2026)

**OpenAI Swarm** (Oct 2024)
- Lightweight multi-agent orchestration framework
- Core abstractions: Agent (instructions + tools + handoffs), Handoff (transfer control between agents)
- Stateless between calls (context managed externally)
- Design philosophy: minimal abstraction over chat completions API

**Anthropic Agent SDK** (2025)
- Agent loop: model generates tool calls → tools execute → results fed back → repeat
- Orchestration patterns: single agent, handoffs, parallel dispatch
- Key abstractions: Agent, Tool, Handoff, GuardrailOutput

**CrewAI** (2024-2025)
- Role-based agent framework: each agent has role, goal, backstory, tools
- Task orchestration: sequential, hierarchical, or consensual process
- Memory: short-term (conversation), long-term (cross-session), entity memory

**LangGraph** (LangChain, 2024-2025)
- Graph-based agent orchestration (nodes = agents/tools, edges = transitions)
- Supports cycles (agent loops), branching, parallel execution
- Persistent state via checkpointing
- Multi-agent patterns: supervisor, hierarchical teams, collaboration

### Key Patterns Across Systems

| Pattern | Description | Example |
|---------|-------------|---------|
| **Coordinator-Worker** | Central agent decomposes task, dispatches to specialists | MetaGPT, CrewAI hierarchical |
| **Debate/Critique** | Agents independently solve, then critique each other | Multi-Agent Debate |
| **Relay/Pipeline** | Sequential handoff: Agent A → Agent B → Agent C | AutoGen two-agent, assembly line |
| **Blackboard** | Shared workspace agents read/write asynchronously | Stigmergy pattern |
| **Voting/Ensemble** | Multiple agents vote, majority or weighted aggregation | MoA (what Edifice has) |
| **Hierarchical** | Tree of agent teams, each team has supervisor | LangGraph hierarchical |

## Relationship to Edifice

### What Edifice Already Has

- **MixtureOfAgents** — Parallel ensemble (voting/ensemble pattern, single-turn)
- **MoE v2** — Expert routing at the layer level (related: dynamic dispatch)
- **MixtureOfDepths** — Adaptive compute allocation per token
- **FreeTransformer** — Latent variable enabling variable compute
- **Coconut** — Multi-step latent reasoning (related: iterative refinement)
- **RLHFHead** — Reward/policy outputs (related: RL-based agent training)
- **Memory architectures** — Titans, MIRAS, NTM, InfiniAttention (related: persistent agent state)

### The Core Tension

Edifice is an **architecture library** — it builds Axon computation graphs for
neural network forward passes. Agent swarm patterns are an **application
framework** concern — they orchestrate multiple model calls, manage state
between turns, and integrate external tools.

The question is: what building blocks can Edifice provide that make swarm
systems easier to build on top?

## Proposed Design

### Scope Decision: Building Blocks, Not Framework

Rather than building a full agent orchestration framework (which would compete
with LangGraph, CrewAI, etc. and isn't Edifice's core competency), we provide
**neural architecture components** that swarm systems need:

1. **Communication layers** — Attention-based message passing between agent hidden states
2. **Role conditioning** — Architecture support for role/instruction-conditioned generation
3. **State persistence** — Architectures designed for multi-turn state maintenance
4. **Routing networks** — Learned dispatch of subtasks to specialist models

### Module 1: `AgentSwarm` — Communication-Augmented Ensemble

The simplest extension of `MixtureOfAgents`: add inter-agent attention between
proposer outputs before aggregation. Each "communication round" lets agents
attend to each other's representations.

```
Input [batch, seq, embed_dim]
      |
      +----+----+----+
      |    |    |    |
      v    v    v    v
     P1   P2   P3   P4    (Proposer stacks)
      |    |    |    |
      v    v    v    v
  [Stack agent outputs along agent dim]
      |
      v
  [batch, num_agents, seq, hidden]
      |
  +---v---+
  | Inter-agent attention (repeat R rounds) |
  | - Each agent attends to all agents     |
  | - Per-agent FFN                        |
  | - Residual + LayerNorm                 |
  +---v---+
      |
      v
  Flatten → Aggregator
      |
      v
  Output [batch, hidden]
```

**Key architectural decisions:**
- Inter-agent attention: full cross-attention (each agent attends to all others)
- Communication rounds: configurable R (default 2), like message-passing GNN rounds
- Agent identity: learned agent embeddings added to representations (so attention can distinguish agents)
- Gating: optional learned gate per agent per round (can learn to ignore communication)

**Implementation notes:**
- Builds on existing `TransformerBlock` for inter-agent attention layers
- Agent dim is a new axis, not collapsed into batch (agents share parameters for communication layers but have separate proposer weights)
- This is still a single forward pass — but captures the "debate" pattern in a differentiable way

### Module 2: `RouterNetwork` — Learned Task Dispatch

A routing network that learns to assign input tokens/sequences to specialist
models. Unlike MoE (which routes at the hidden-state level within a single
model), this routes at the input level across separate models.

```
Input [batch, seq, embed_dim]
      |
      v
  Router (small transformer or MLP)
      |
      v
  Dispatch weights [batch, num_specialists]
      |
      +---- soft: weighted sum of specialist outputs
      |
      +---- hard: top-k selection (straight-through estimator for training)
```

**Builds on:** MoE v2 routing (auxiliary-loss-free), MixtureOfDepths (token-level routing)

### Module 3: `StatefulAgent` — Multi-Turn Architecture Wrapper

A wrapper that adds explicit state management to any Edifice architecture,
designed for multi-turn agent interactions. The key insight from memory
architectures (Titans, InfiniAttention, NTM) is that persistent state can be
maintained as tensor state rather than token-level KV cache.

```
Turn 1:  Input_1 + State_0 → Output_1 + State_1
Turn 2:  Input_2 + State_1 → Output_2 + State_2
Turn 3:  Input_3 + State_2 → Output_3 + State_3
```

**State types:**
- **Compressive memory** (InfiniAttention-style): fixed-size memory updated each turn
- **Episodic buffer** (NTM-style): addressable memory bank with read/write heads
- **Running statistics** (Titans-style): matrix state tracking prediction errors

This doesn't orchestrate multi-agent systems — it provides the stateful
backbone that each agent in a swarm would use.

### Module 4: `MessagePassingAgents` — GNN-Inspired Agent Graph

Model the agent swarm as a graph where agents are nodes and communication
channels are edges. Apply graph neural network message passing:

```
For each communication round:
  1. Each agent computes a message: m_ij = MLP(h_i, h_j)
  2. Each agent aggregates incoming messages: agg_i = sum(m_ji for j in neighbors)
  3. Each agent updates its state: h_i = GRU(h_i, agg_i)
```

**Builds on:** Existing graph family (MessagePassing, GAT, GraphTransformer)

This captures the "agent graph" topology that systems like LangGraph define
declaratively, but in a differentiable form that can be trained end-to-end.

## Implementation Plan

### Phase 1 — `AgentSwarm` (communication-augmented ensemble)

**Files:**
- `lib/edifice/meta/agent_swarm.ex` — Module with `build/1`
- `test/edifice/meta/agent_swarm_test.exs` — Shape, correctness, gradient tests

**Parameters:**
- `embed_dim` — Input embedding dimension
- `num_agents` — Number of parallel proposer agents (default: 4)
- `agent_hidden_size` — Hidden size per agent proposer (default: 128)
- `agent_layers` — Transformer layers per proposer (default: 2)
- `num_heads` — Attention heads in both proposer and inter-agent attention (default: 4)
- `communication_rounds` — Rounds of inter-agent attention (default: 2)
- `aggregator_hidden_size` — Aggregator transformer hidden size (default: 256)
- `aggregator_layers` — Aggregator transformer layers (default: 2)
- `use_agent_embeddings` — Add learned agent identity (default: true)
- `communication_gate` — Learnable per-agent gate on comm (default: false)

**Architecture flow:**
1. Shared input embedding (or per-agent if `shared_embedding: false`)
2. N independent proposer transformer stacks (reuse `TransformerBlock`)
3. Stack outputs: `[batch, num_agents, seq, hidden]`
4. Add learned agent embeddings if enabled
5. R rounds of inter-agent cross-attention + FFN (each agent attends to all)
6. Flatten agent dim, project to aggregator size
7. Aggregator transformer stack
8. Final norm → last-timestep output

**Complexity:** ~200-250 lines. Moderate — reuses TransformerBlock, adds agent
dim handling and inter-agent attention loop.

### Phase 2 — `RouterNetwork` (learned dispatch)

- Simpler module, ~100-150 lines
- Soft routing (weighted sum) and hard routing (top-k with straight-through)
- Builds directly on MoE v2 router patterns

### Phase 3 — `StatefulAgent` (multi-turn wrapper)

- Wrapper module that pairs any architecture with a state module
- ~150-200 lines
- State module choices: compressive (InfiniAttention), episodic (NTM), matrix (Titans)

### Phase 4 — `MessagePassingAgents` (GNN-inspired)

- Graph-based agent communication
- ~200-250 lines
- Reuses MessagePassing infrastructure from graph family

## What This Is NOT

- **Not an orchestration framework** — No HTTP calls, no tool execution, no prompt management
- **Not an agent runtime** — No event loops, no turn management, no conversation history
- **Not competing with LangGraph/CrewAI** — Those are application frameworks; this is architecture components

The target user builds a swarm system using LangGraph or similar, and uses
Edifice `AgentSwarm` / `StatefulAgent` as the neural backbone for each agent
or for the coordination layer itself.

## References

1. Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate", 2023
2. Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation", 2023
3. Li et al., "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society", 2023
4. Hong et al., "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework", 2023
5. OpenAI, "Swarm: Educational Framework for Lightweight Multi-Agent Orchestration", 2024
6. Anthropic, "Claude Agent SDK", 2025
7. Bonabeau et al., "Swarm Intelligence: From Natural to Artificial Systems", 1999
