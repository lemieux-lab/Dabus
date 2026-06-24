# Dabus

A Julia package for visualizing [Flux.jl](https://fluxml.ai/) neural network architectures as GraphViz diagrams.

![Dense network with relu and sigmoid activations](examples/graph_1.png)

---

## What's new in 0.3.0

- **Custom `@layer` structs render automatically** - no boilerplate needed. Single-sublayer structs appear as labeled clusters; multi-sublayer structs fan out into parallel paths. See [Custom layers](#custom-layers).
- **Local GraphViz rendering** - if `dot` is installed, diagrams are rendered locally with no internet required. Falls back to the HTTP API with a warning if `dot` is not found. See [Local rendering](#local-rendering).
- **`network_to_dot`** - new exported function that returns the DOT source string, useful for debugging or piping into other tools.
- **`check_graphviz`** - new exported utility to verify your local GraphViz installation.
- **CI via GitHub Actions** - automated test runs on every push and pull request.

---

## Installation

This package is registered in [LabRegistry](https://github.com/lemieux-lab/LabRegistry).

```julia
# With LabRegistry added to your Julia registries
] add Dabus

# Otherwise
] add https://github.com/lemieux-lab/Dabus
```

Or for local development:

```julia
] dev path/to/Dabus
```

---

## Local rendering

Dabus can render diagrams locally using the `dot` binary from [GraphViz](https://graphviz.org/), or fall back to the [QuickChart.io](https://quickchart.io/) HTTP API when `dot` is not available.

### Installing GraphViz

| Platform      | Command                                 |
|:------------- |:--------------------------------------- |
| Linux (apt)   | `sudo apt install graphviz`             |
| Linux (conda) | `conda install -c conda-forge graphviz` |
| macOS         | `brew install graphviz`                 |
| Windows       | <https://graphviz.org/download/>        |

### Checking your installation

```julia
using Dabus

check_graphviz()
# [ Info: GraphViz `dot` is available at /usr/bin/dot
# true
```

### Renderer selection

The `renderer` keyword on `draw_network` controls which backend is used:

| Value      | Behaviour                                                                     |
|:---------- |:----------------------------------------------------------------------------- |
| `:auto`    | Uses local `dot` if found on PATH, otherwise falls back to HTTP with a warning |
| `:local`   | Always uses local `dot` (errors if GraphViz is not installed)                 |
| `:http`    | Always uses the QuickChart.io API (requires internet, silences the warning)   |

`:auto` is the default. Once GraphViz is installed, no code changes are needed.

---

## Usage

```julia
using Dabus, Flux

model = Chain(
    Dense(50, 10, relu),
    Dense(10, 1)
)

# Render and return image bytes (auto-detects local vs HTTP)
img = draw_network(model)

# Save directly to a file
draw_network(model, save_to="network.png")

# Export as SVG, force local rendering
draw_network(model, save_to="network.svg", output_type="svg", renderer=:local)

# Get the raw DOT source (no rendering)
dot = network_to_dot(model)
```

### Custom layers

Structs decorated with `Flux.@layer` render automatically without any extra code.

**Single sublayer field** - the inner layer is rendered inside a cluster labeled with the struct name:

```julia
struct MyEncoder
    layers::Chain
end
Flux.@layer MyEncoder

draw_network(Chain(Dense(128, 64), MyEncoder(Chain(Dense(64, 32, relu), Dense(32, 16)))))
```

**Multiple sublayer fields** - each sublayer becomes a parallel branch, merging at a summary node labeled with the struct name:

```julia
struct ResBlock
    main::Chain
    skip::Dense
end
Flux.@layer ResBlock

draw_network(Chain(Dense(64, 64), ResBlock(Chain(Dense(64, 64, relu), Dense(64, 64)), Dense(64, 64))))
```

Nesting works to any depth - a custom layer inside another custom layer inside a `Chain` all renders correctly.

---

## Supported layer types

| Category           | Layers                                                                   |
|:------------------ |:------------------------------------------------------------------------ |
| Standard           | `Dense`, `Embedding`                                                     |
| Convolutional      | `Conv`, `ConvTranspose`, `CrossCor`                                      |
| Pooling            | `MaxPool`, `MeanPool`, `AdaptiveMaxPool`, `AdaptiveMeanPool`, `GlobalMeanPool` |
| Attention          | `MultiHeadAttention`                                                     |
| Recurrent          | `LSTM`, `GRU`                                                            |
| Containers         | `Chain`, `Parallel`, `Maxout`, `PairwiseFusion`                          |
| Skip connections   | `SkipConnection`                                                         |
| Custom             | Any struct decorated with `Flux.@layer`                                  |
| Activations / misc | Any callable (e.g. `relu`, `softmax`, `Flux.flatten`)                   |

---

## API Reference

### `draw_network(network; save_to=nothing, output_type="png", renderer=:auto)`

Generates a diagram of a Flux neural network.

**Arguments:**
- `network` : A Flux model (`Chain`, any supported layer, or a custom `@layer` struct).
- `save_to` : Optional file path. Image bytes are written to this path if provided.
- `output_type` : Output format - `"png"` (default) or `"svg"`.
- `renderer` : `:auto` (default), `:local`, or `:http`. See [Renderer selection](#renderer-selection).

**Returns:** `Vector{UInt8}` - the raw image bytes.

---

### `network_to_dot(network) -> String`

Returns the GraphViz DOT source for a network without rendering it. Useful for debugging the graph structure or piping into external tools.

```julia
dot = network_to_dot(Chain(Dense(10, 5, relu), Dense(5, 1)))
println(dot)
```

---

### `check_graphviz() -> Bool`

Checks whether GraphViz's `dot` binary is installed and accessible on PATH. Prints an `@info` message with the binary path on success, or a `@warn` with installation instructions on failure.

```julia
check_graphviz()  # true or false
```

---

## Examples

**Embedding + Dense**

![Embedding followed by a Dense layer with leakyrelu](examples/graph_2.png)

**Dense + softmax activation**

![Dense layers with an intermediate softmax node](examples/graph_3.png)

**Convolutional layers (Conv, ConvTranspose, CrossCor)**

![Conv, ConvTranspose, and CrossCor layers in sequence](examples/graph_4.png)

**Pooling layers**

![MaxPool, GlobalMeanPool, and AdaptiveMaxPool](examples/graph_5.png)

**Multi-head attention**

![Two MultiHeadAttention layers followed by a Dense layer](examples/graph_6.png)

**LSTM recurrent network**

![Dense → LSTM → Dense with leakyrelu](examples/graph_7.png)

**Complex parallel architecture**

![Parallel branches with Embedding and LSTM paths merged into Dense layers](examples/graph_8.png)

---

## Requirements

- Julia ≥ 1.0
- [Flux.jl](https://github.com/FluxML/Flux.jl) ≥ 0.16
- [HTTP.jl](https://github.com/JuliaWeb/HTTP.jl) ≥ 1.0
- **GraphViz** (optional) - install for local rendering; falls back to the QuickChart.io API otherwise.
