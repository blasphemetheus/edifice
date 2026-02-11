# Graph and Set Networks
> Message passing on irregular structures -- from spectral convolutions on fixed graphs to attention over molecules to permutation-invariant processing of point clouds and sets.

## Overview

Graph neural networks (GNNs) generalize deep learning to data that lives on nodes and edges rather than in grids or sequences. Social networks, molecules, knowledge bases, protein structures, and 3D point clouds all have graph structure that convolutions and recurrences cannot naturally exploit. The GNN family in Edifice provides eight graph architectures and two set-processing architectures that handle these irregular domains.

The unifying abstraction is **message passing**: each node gathers information from its neighbors, transforms it, and updates its own representation. What varies across architectures is how messages are computed (linear projection, attention, continuous filters), how they are aggregated (sum, mean, max, or multiple aggregators), and how expressive the resulting model is (bounded by the Weisfeiler-Lehman graph isomorphism hierarchy).

Sets can be viewed as a special case: graphs with no edges, where the only structural constraint is permutation invariance. DeepSets and PointNet achieve this through symmetric aggregation functions (sum, max) applied after per-element transformations, with PointNet adding spatial alignment networks for 3D geometric data.

## Conceptual Foundation

The message passing framework decomposes a GNN layer into three stages:

    1. Message:    m_{ij} = MSG(h_i, h_j, e_{ij})    -- compute message from j to i
    2. Aggregate:  M_i    = AGG({m_{ij} : j in N(i)}) -- collect messages at node i
    3. Update:     h_i'   = UPD(h_i, M_i)             -- update node i's features

The key equation for the simplest case (GCN) reduces message passing to a single matrix operation:

    H' = sigma(D^{-1/2} A D^{-1/2} H W)

where A is the adjacency matrix (with self-loops), D is the degree matrix, H is the node feature matrix, W is a learnable weight matrix, and sigma is a nonlinearity. This formulation comes from approximating spectral graph convolutions with first-order Chebyshev polynomials.

For sets without edges, the message passing collapse to:

    output = rho( AGG_i( phi(x_i) ) )

where phi processes each element independently and AGG is a symmetric function (sum, mean, max). This is provably a universal approximator for permutation-invariant functions.

## Architecture Evolution

```
2005  Spectral Graph Theory foundations
  |     (Bruna et al., spectral convolutions on graphs)
  |
2017  GCN (Kipf & Welling, ICLR)
  |     - First-order Chebyshev approximation
  |     - Normalized adjacency propagation
  |     [GCN]
  |
2017  GraphSAGE (Hamilton et al., NeurIPS)       PointNet (Qi et al., CVPR)
  |     - Inductive learning via                    - Max pool over points
  |       neighborhood sampling                     - T-Net spatial alignment
  |     [GraphSAGE]                                 [PointNet]
  |
2017  MPNN (Gilmer et al.)                       DeepSets (Zaheer et al.)
  |     - Unifying message passing framework        - Permutation-invariant
  |     [MessagePassing]                              universal approximator
  |                                                 [DeepSets]
  |
2017  SchNet (Schutt et al., NeurIPS)
  |     - Continuous-filter convolutions
  |     - RBF distance expansion for molecules
  |     [SchNet]
  |
2018  GAT (Velickovic et al., ICLR)
  |     - Learned attention weights per edge
  |     - Multi-head attention on graphs
  |     [GAT]
  |
2019  GIN (Xu et al., ICLR)
  |     - Provably maximally powerful under WL test
  |     - Sum aggregation + learnable epsilon
  |     [GIN]
  |
2020  PNA (Corso et al., NeurIPS)
  |     - Multiple aggregators + degree scalers
  |     - Principled aggregation diversity
  |     [PNA]
  |
2021  Graph Transformer (Dwivedi & Bresson; Ying et al.)
        - Full attention over graph nodes
        - Adjacency as attention bias/mask
        [GraphTransformer]
```

## When to Use What

| Scenario | Module | Rationale |
|----------|--------|-----------|
| Simple, homogeneous graph (citation, social) | `GCN` | Fast, well-understood, good default |
| Heterogeneous neighbor importance | `GAT` | Learned attention weights adapt to each edge |
| Graph classification (isomorphism-sensitive) | `GIN` | Provably 1-WL expressive; sum aggregation distinguishes multisets |
| Large/evolving graphs, inductive setting | `GraphSAGE` | Neighborhood sampling; works on unseen nodes |
| Maximum aggregation expressiveness | `PNA` | Multiple aggregators x scalers capture diverse structural patterns |
| Molecular property prediction | `SchNet` | Continuous-filter convolutions on interatomic distances |
| Graph + global context needed | `GraphTransformer` | Full attention (not just neighbors); adjacency biases structure |
| Custom message/aggregate/update | `MessagePassing` | Generic MPNN framework; build your own GNN variant |
| Permutation-invariant set function | `DeepSets` | Simplest provably universal set architecture |
| 3D point cloud classification | `PointNet` | Per-point MLP + max pool + T-Net spatial alignment |

### Expressiveness Hierarchy

The Weisfeiler-Lehman (WL) graph isomorphism test provides a theoretical ceiling for message passing GNNs. The hierarchy from least to most expressive:

```
Expressiveness of message passing GNNs:

  GCN (mean aggregation)
    |  Cannot distinguish certain regular graphs
    |
  GAT (attention-weighted aggregation)
    |  Adaptive weighting, but still bounded by 1-WL
    |
  GIN (sum aggregation + learnable epsilon)  <==>  1-WL test
    |  Maximally powerful among standard MPNN architectures
    |
  PNA (multiple aggregators + scalers)
    |  Captures more structural info than any single aggregator
    |
  GraphTransformer (full attention + positional encoding)
       Can approximate higher-order WL with proper positional encoding
```

## Key Concepts

### Spectral vs Spatial Approaches

Graph convolutions originated from spectral graph theory: define convolution as multiplication in the graph Fourier domain (eigenvectors of the graph Laplacian). GCN approximates this with a first-order Chebyshev polynomial, yielding the one-hop neighbor aggregation rule.

Spatial approaches (GAT, GraphSAGE, GIN) operate directly in the node domain without referencing the spectrum. They define operations in terms of neighborhoods: for each node, look at its neighbors, compute something, and aggregate. Spatial methods are more flexible (they handle varying graph structures, inductive settings, and edge features more naturally) and have largely superseded purely spectral methods.

```
  Spectral                                Spatial
  --------                                -------
  Defined via graph Laplacian             Defined via neighborhoods
  Global operation (eigendecomposition)   Local operation (message passing)
  Fixed graph structure required          Handles dynamic/unseen graphs
  Smooth spectral filters                 Arbitrary message functions
  Example: GCN (1st-order approx)         Examples: GAT, GIN, GraphSAGE

  In practice, GCN is used as a spatial method -- the spectral
  derivation justifies the normalized adjacency propagation rule,
  but the implementation is purely spatial (matrix multiply with A).
```

### The Message Passing Framework

The `MessagePassing` module implements the generic MPNN framework that all other graph modules specialize. Understanding it clarifies the design space:

**Message function**: How does node j contribute to node i? Options range from simple (linear projection of j's features, as in GCN) to complex (attention-weighted projection using both i and j, as in GAT) to continuous (distance-dependent filter, as in SchNet).

**Aggregation function**: How are messages from multiple neighbors combined? Sum preserves multiset information (GIN). Mean normalizes by degree (GCN). Max captures the most salient neighbor (GraphSAGE with `:max`). PNA uses all of the above simultaneously plus standard deviation.

**Update function**: How does node i incorporate the aggregated messages? Simple approaches add or concatenate messages with self-features then project. GraphSAGE concatenates and L2-normalizes. GIN uses (1 + epsilon) * self + aggregated, passed through an MLP.

```
Message Passing decomposition for each GNN variant:

  GCN:       MSG = W * h_j          AGG = mean      UPD = sigma(agg)
  GAT:       MSG = alpha_ij * W*h_j AGG = sum        UPD = sigma(agg)
  GIN:       MSG = h_j              AGG = sum        UPD = MLP((1+eps)*h_i + agg)
  GraphSAGE: MSG = h_j              AGG = mean/max   UPD = W*[h_i || agg], L2-norm
  SchNet:    MSG = filter(d_ij)*h_j AGG = sum        UPD = MLP(agg) + h_i
  PNA:       MSG = h_j              AGG = [mean,max, UPD = project(concat(all))
                                          sum,std]
```

### Domain-Specific Architectures: SchNet and PointNet

**SchNet** is designed for molecular graphs where edges represent interatomic distances rather than binary connections. It uses radial basis functions (RBFs) to expand continuous distances into a feature vector, then generates per-edge filter weights from these RBF features. This continuous-filter approach means SchNet can distinguish atoms that are 2.1 angstroms apart from those 2.3 angstroms apart -- critical for chemistry where bond lengths determine molecular properties.

The cosine cutoff envelope smoothly zeros out interactions beyond a distance threshold, ensuring locality and preventing discontinuities in the energy surface.

**PointNet** processes 3D point clouds (unordered sets of xyz coordinates) for classification and segmentation. Its key contribution is the T-Net: a mini-PointNet that predicts a spatial transformation matrix, applied to the input to canonicalize the orientation. This makes the model robust to rotations and translations. After per-point feature extraction through shared MLPs, a global max pool creates a permutation-invariant representation.

### Sets as Edgeless Graphs

DeepSets and PointNet share the permutation-invariance constraint: the output must not change when input elements are reordered. DeepSets proves that any continuous permutation-invariant function on sets can be decomposed as:

    f({x_1, ..., x_n}) = rho( SUM_i phi(x_i) )

where phi processes elements independently and rho processes the aggregate. This is the theoretical foundation underlying both DeepSets (which implements it directly) and PointNet (which uses max instead of sum, and adds spatial alignment).

```
  Set processing as a special case of message passing:

  Graph (with edges):     Node i gathers from neighbors N(i)
  Set (no edges):         Node i gathers from ALL other nodes (or none)
  DeepSets:               No inter-element interaction; phi per element, then aggregate
  PointNet:               Same as DeepSets + learned spatial alignment (T-Net)

  The choice of symmetric aggregation function matters:

  SUM:   Preserves count information; |{a,a,b}| != |{a,b}|
  MEAN:  Normalized; loses count but more stable for variable-size sets
  MAX:   Captures extrema; ignores multiplicity entirely
```

## Complexity Comparison

| Module | Per-Layer Compute | Memory (adjacency) | Supports Inductive | Global Pooling |
|--------|------------------|--------------------|--------------------|----------------|
| `GCN` | O(E * d + N * d^2) | O(N^2) dense | No (spectral) | Via `MessagePassing` |
| `GAT` | O(E * d + N * d^2) | O(N^2) dense | Yes | Via `MessagePassing` |
| `GIN` | O(E * d + N * d^2) | O(N^2) dense | Yes | Built-in option |
| `GraphSAGE` | O(E * d + N * d^2) | O(N^2) dense | Yes (designed for it) | Built-in option |
| `PNA` | O(E * d * A * S + N * d^2) | O(N^2) dense | Yes | Built-in option |
| `SchNet` | O(E * R * d + N * d^2) | O(N^2) dense (distances) | Yes | Built-in option |
| `GraphTransformer` | O(N^2 * d) | O(N^2) dense | Yes | Built-in option |
| `MessagePassing` | O(E * d + N * d^2) | O(N^2) dense | Yes | `global_pool/2` |
| `DeepSets` | O(N * d^2) | None needed | N/A (no graph) | Built-in (sum/mean/max) |
| `PointNet` | O(N * d^2 + T-Net) | None needed | N/A (no graph) | Built-in (max) |

Where N = nodes, E = edges, d = hidden dimension, A = number of aggregators (PNA), S = number of scalers (PNA), R = number of RBF centers (SchNet).

Note: Edifice uses dense adjacency matrices (O(N^2) memory) because Nx does not support sparse tensors. For large graphs, batch subgraphs or use neighborhood sampling.

## Module Reference

- `Edifice.Graph.GCN` -- Spectral graph convolutions via normalized adjacency; node and graph classification with optional global pooling
- `Edifice.Graph.GAT` -- Multi-head attention over graph neighbors with LeakyReLU scoring; concatenated or averaged heads
- `Edifice.Graph.GIN` -- Maximally expressive MPNN (1-WL equivalent) using sum aggregation and learnable epsilon weighting
- `Edifice.Graph.GraphSAGE` -- Inductive graph learning with mean/max/pool aggregation, self-concatenation, and L2 normalization
- `Edifice.Graph.GraphTransformer` -- Full attention over graph nodes with adjacency-biased scores and random-walk positional encoding
- `Edifice.Graph.PNA` -- Principal Neighbourhood Aggregation combining mean/max/sum/std aggregators with identity/amplification/attenuation scalers
- `Edifice.Graph.SchNet` -- Continuous-filter convolutions for molecular graphs using RBF distance expansion and cosine cutoff envelopes
- `Edifice.Graph.MessagePassing` -- Generic MPNN framework with configurable message/aggregation/update functions and global pooling utilities
- `Edifice.Sets.DeepSets` -- Permutation-invariant set functions via independent phi network, symmetric aggregation, and rho post-processing
- `Edifice.Sets.PointNet` -- 3D point cloud processing with per-point shared MLPs, optional T-Net spatial alignment, and max pooling

## Cross-References

- **attention_mechanisms.md** -- GraphTransformer uses full multi-head attention with adjacency as a structural bias; the attention mechanism is the same as in standard transformers
- **building_blocks.md** -- Graph architectures can incorporate RMSNorm, SwiGLU, and other blocks as internal components of their message and update functions
- **meta_learning.md** -- GNNs compose naturally with Mixture-of-Experts for routing-based architectures on graph-structured inputs

## Further Reading

1. Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017) -- arxiv.org/abs/1609.02907. The foundational GCN paper deriving graph convolution from spectral theory.
2. Velickovic et al., "Graph Attention Networks" (ICLR 2018) -- arxiv.org/abs/1710.10903. Introduces learned attention weights for neighbor aggregation.
3. Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019) -- arxiv.org/abs/1810.00826. Proves GIN matches 1-WL expressiveness and analyzes limitations of other GNNs.
4. Gilmer et al., "Neural Message Passing for Quantum Chemistry" (ICML 2017) -- arxiv.org/abs/1704.01212. Unifying MPNN framework that subsumes GCN, GAT, and SchNet.
5. Corso et al., "Principal Neighbourhood Aggregation for Graph Nets" (NeurIPS 2020) -- arxiv.org/abs/2004.05718. Systematic analysis showing that combining diverse aggregators with degree scalers maximizes expressiveness.
