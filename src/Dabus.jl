module Dabus

using HTTP; const http = HTTP
using Flux
using Functors

export draw_network, network_to_dot, check_graphviz, NetworkImage

# ---------------------------------------------------------------------------
# NetworkImage display-aware wrapper around rendered bytes
# ---------------------------------------------------------------------------

"""
    NetworkImage

Wraps the raw bytes returned by `draw_network`. Renders inline automatically
in IJulia, Pluto, and the VSCode Julia extension.

Access the raw bytes via `.bytes` for manual file I/O.
"""
struct NetworkImage
    bytes::Vector{UInt8}
    output_type::String
end

Base.show(io::IO, ::MIME"text/plain", img::NetworkImage) =
    print(io, "NetworkImage($(img.output_type), $(length(img.bytes)) bytes)")

Base.show(io::IO, ::MIME"image/png", img::NetworkImage) = write(io, img.bytes)
Base.show(io::IO, ::MIME"image/svg+xml", img::NetworkImage) = write(io, img.bytes)

Base.showable(::MIME"image/png", img::NetworkImage) = img.output_type == "png"
Base.showable(::MIME"image/svg+xml", img::NetworkImage) = img.output_type == "svg"

# ---------------------------------------------------------------------------
# Rendering backends
# ---------------------------------------------------------------------------

function http_graph(graph::String; output_type = "png")
    r = http.get("https://quickchart.io/graphviz", query = Dict("format" => output_type, "graph" => graph))
    return r.body
end

function render_dot_local(dot::String; output_type = "png")
    return read(pipeline(IOBuffer(dot), `dot -T$output_type`))
end

"""
    check_graphviz() -> Bool

Check whether GraphViz's `dot` binary is installed and accessible on PATH.
Returns `true` if found and executable, `false` otherwise.

When `dot` is available, `draw_network` uses it automatically (no internet needed).
When it is not, `draw_network` falls back to the QuickChart.io HTTP API.

# Installing GraphViz

| Platform      | Command                                      |
|:------------- |:-------------------------------------------- |
| Linux (apt)   | `sudo apt install graphviz`                  |
| Linux (conda) | `conda install -c conda-forge graphviz`      |
| macOS         | `brew install graphviz`                      |
| Windows       | <https://graphviz.org/download/>             |

After installing, verify with `check_graphviz()` or `which dot` in your shell.

# Examples
```julia-repl
julia> check_graphviz()
[ Info: GraphViz `dot` is available at /usr/bin/dot
true
```
"""
function check_graphviz()
    path = Sys.which("dot")
    if isnothing(path)
        @warn """GraphViz `dot` not found on PATH. Install it for local rendering:
          Linux (apt):   sudo apt install graphviz
          Linux (conda): conda install -c conda-forge graphviz
          macOS:         brew install graphviz
          Windows:       https://graphviz.org/download/"""
        return false
    end
    try
        run(pipeline(`dot -V`, stdout = devnull, stderr = devnull))
        @info "GraphViz `dot` is available at $path"
        return true
    catch e
        @warn "GraphViz `dot` found at $path but failed to execute: $e"
        return false
    end
end

# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

function count_params(node)
    cs = Functors.children(node)
    n = 0
    for v in values(cs)
        if v isa AbstractArray
            n += length(v)
        elseif !Functors.isleaf(v)
            n += count_params(v)
        end
    end
    n
end
count_params(::Function) = 0

function format_params(n::Int)
    n == 0 && return ""
    n >= 1_000_000 && return "$(round(n / 1_000_000, digits = 1))M params"
    n >= 10_000 && return "$(div(n, 1_000))K params"
    return "$n params"
end

# ---------------------------------------------------------------------------
# DOT generation
# ---------------------------------------------------------------------------

"""
    network_to_dot(network) -> String

Return the GraphViz DOT source for `network` without rendering it.
Useful for debugging or piping into external tools.
"""
function network_to_dot(network)
    headers, links, _ = symbol_analysis(network, 0, "", "")
    graph = """
    digraph network {
        node [fillcolor = ".7 .3 1.0", style = filled];
        overlap=scale;
        compound=true;

        subgraph repr {
            rankdir=TB;
            $headers
            $links
        }

    }
    """
    return replace(graph, " => " => ", ")
end

"""
    draw_network(network; save_to=nothing, output_type="png", renderer=:auto) -> NetworkImage

Draw a Flux network as a GraphViz diagram. Returns a `NetworkImage` that renders
inline in IJulia, Pluto, and the VSCode Julia extension automatically.

**Arguments:**
- `network`     : Any Flux model (`Chain`, layer, or custom `@layer` struct).
- `save_to`     : Optional file path — image bytes are written there if provided.
- `output_type` : Output format passed to GraphViz. Common values:
  - `"png"` (default) — raster image; displays inline in notebooks and VSCode.
  - `"svg"` — scalable vector; displays inline in notebooks and VSCode.
  - `"pdf"`, `"eps"`, `"jpg"` supported with local `dot`; HTTP fallback may not support these.
- `renderer`    : Controls how the DOT source is rendered:
  - `:auto` (default) — uses local `dot` if found on PATH, falls back to HTTP with a warning.
  - `:local` — always uses the local `dot` binary (errors if GraphViz is not installed).
  - `:http` — always uses the QuickChart.io API (requires internet).

Run `check_graphviz()` to verify whether local rendering is available.

# Examples
```julia-repl
julia> draw_network(Chain(Dense(50, 10, relu), Dense(10, 1)))   # displays inline
NetworkImage(png, 12345 bytes)

julia> draw_network(model, save_to="network.png")

julia> draw_network(model, output_type="svg", renderer=:local)

julia> draw_network(model, output_type="pdf", save_to="network.pdf", renderer=:local)
```
"""
function draw_network(network; save_to = nothing, output_type = "png", renderer = :auto)
    graph = network_to_dot(network)
    bytes = if renderer === :local
        render_dot_local(graph; output_type)
    elseif renderer === :http
        http_graph(graph; output_type)
    else  # :auto
        if !isnothing(Sys.which("dot"))
            render_dot_local(graph; output_type)
        else
            @warn "GraphViz `dot` not found on PATH: falling back to HTTP API (requires internet). " *
                  "Run `check_graphviz()` for installation instructions, or pass `renderer=:http` to suppress this warning."
            http_graph(graph; output_type)
        end
    end
    if !isnothing(save_to)
        open(save_to, "w") do handle
            write(handle, bytes)
        end
    end
    return NetworkImage(bytes, output_type)
end

# ---------------------------------------------------------------------------
# Per-node DOT rendering
# ---------------------------------------------------------------------------

function draw_type(node::Any, node_id)
    cs = Functors.children(node)
    if cs isa NamedTuple
        sublayers = [v for v in values(cs) if !Functors.isleaf(v)]
        if !isempty(sublayers)
            return draw_custom_composite(node, node_id, sublayers)
        end
    end
    if hasproperty(node, :weight) || hasproperty(node, :cell)
        return draw_standard_node(node, node_id)
    else
        return draw_misc_node(node, node_id)
    end
end
draw_type(node::Union{Conv, ConvTranspose, CrossCor}, node_id) = draw_convolution_node(node, node_id)
draw_type(node::Union{MultiHeadAttention}, node_id) = draw_transformer_node(node, node_id)
draw_type(node::Union{AdaptiveMaxPool, MaxPool, MeanPool, AdaptiveMeanPool}, node_id) = draw_pooling_node(node, node_id)
draw_type(node::Union{Parallel, Maxout, PairwiseFusion}, node_id) = draw_container_node(node, node_id)
draw_type(node::Union{RNNCell}, node_id) = draw_recursive_node(node, node_id)
draw_type(node::Union{SkipConnection}, node_id) = draw_skip_connection(node, node_id)
draw_type(node::Chain, node_id) = draw_chain(node, node_id)

function draw_custom_composite(node, node_id, sublayers)
    layer_type = String(nameof(typeof(node)))
    params = format_params(count_params(node))
    label = isempty(params) ? layer_type : "$layer_type<BR/><FONT POINT-SIZE=\"9\">$params</FONT>"

    if length(sublayers) == 1
        headers, links, node_id = symbol_analysis(sublayers[1], node_id, "", "")
        headers = """
        subgraph cluster_$(hash(node_id)) {
            label=<$label>;
            $headers
        }
        """
        return headers, links, node_id
    else
        prev_node_id = node_id
        firsts = []
        lasts = []
        headers = ""
        links = ""
        for (i, sublayer) in enumerate(sublayers)
            push!(firsts, node_id + 1)
            headers, links, node_id = symbol_analysis(sublayer, node_id, headers, links,
                remove_first_link = (prev_node_id != 0 || !(i == 1)))
            push!(lasts, node_id)
        end
        node_id += 1
        merge_header = """node$node_id [shape=none fillcolor="1.0 .6 1.0" margin=0 label=<
        <TABLE border="0" cellborder="1" style="rounded">
            <TR height="1"><TD port="up" border="0" height="1"></TD></TR>
            <TR><TD><B>$layer_type</B></TD></TR>
            <TR height="1"><TD port="down" border="0" height="1"></TD></TR>
        </TABLE>
        >];"""
        headers = "$headers\n\t$merge_header"
        for last in lasts
            links = "$links\nnode$last:down:c -> node$node_id:up:c;"
        end
        for first in firsts
            links = prev_node_id > 0 ? "$links\nnode$prev_node_id:down:c -> node$first:up:c;" : links
        end
        subgraph = """
        subgraph cluster_$node_id {
            label=<$label>;
            $headers
        }
        """
        return subgraph, links, node_id
    end
end

function symbol_analysis(node, node_id, headers, links; remove_first_link = false)
    header, link, node_id = draw_type(node, node_id)

    if remove_first_link  # removes the first link, useful when accumulating links via Parallel or similar
        tmp = IOBuffer(link)
        readline(tmp)
        link = String(read(tmp))
    end

    headers = "$headers\n\t$header"
    links = node_id > 1 && link != "" ? strip("\t$links\n$link") : links
    return headers, links, node_id
end

function draw_chain(node, node_id)
    headers = ""
    links = ""
    for layer in node
        headers, links, node_id = symbol_analysis(layer, node_id, headers, links)
    end
    params = format_params(count_params(node))
    label_attr = isempty(params) ? "label=\"\"" : "label=<$params>"
    headers = """
    subgraph cluster_$(hash(node_id)){
        $label_attr;
        $headers

    }
    """
    return headers, links, node_id
end

function draw_misc_node(node, node_id)
    node_id += 1
    header = """node$node_id [shape=none fillcolor = "1.0 .6 1.0" margin=0 label=<
    <TABLE border="0" cellborder= "1" style="rounded">
        <TR height="1"><TD port="up" border="0" height="1" colspan="2"></TD></TR>
        <TR>
            <TD colspan="2"><B>$(Symbol(node))</B></TD>
        </TR>
        <TR height="1"><TD port="down" border="0" height="1" colspan="2"></TD></TR>
    </TABLE>
    >];"""
    link = "node$(node_id-1):down:c -> node$node_id:up:c;"
    return header, link, node_id
end

function draw_convolution_node(node, node_id)
    layer_type = String(nameof(typeof(node)))
    node_id += 1
    activation = hasproperty(node, :σ) ? "\n<TD><I>$(Symbol(node.σ))</I></TD>" : ""
    layer_specs = size(node.weight)[end-1:end]
    layer_kernel = size(node.weight)[begin:end-length(layer_specs)]
    params = format_params(count_params(node))
    params_row = isempty(params) ? "" :
        "\n<TR><TD colspan=\"3\"><FONT POINT-SIZE=\"9\">$params</FONT></TD></TR>"
    header = """node$node_id [shape=none margin=0 label=<
                        <TABLE border="0" cellborder= "1" style="rounded">
                            <TR height="1"><TD port="up" border="0" height="1" colspan="3"></TD></TR>
                            <TR>
                                <TD colspan="3"><B>$layer_type</B></TD>
                            </TR>
                            <TR>
                                <TD>$layer_specs</TD><TD>$layer_kernel</TD>$activation
                            </TR>$params_row
                            <TR height="1"><TD port="down" border="0" height="1" colspan="3"></TD></TR>
                        </TABLE>
                        >];"""
    link = "node$(node_id-1):down:c -> node$node_id:up:c;"
    return header, link, node_id
end

function draw_transformer_node(node, node_id)
    layer_type = String(nameof(typeof(node)))
    node_id += 1
    k_proj = reverse(size(node.k_proj.weight))
    q_proj = reverse(size(node.q_proj.weight))
    v_proj = reverse(size(node.v_proj.weight))
    params = format_params(count_params(node))
    params_row = isempty(params) ? "" :
        "\n<TR><TD colspan=\"2\"><FONT POINT-SIZE=\"9\">$params</FONT></TD></TR>"
    header = """node$node_id [shape=none margin=0 label=<
                        <TABLE border="0" cellborder= "1" style="rounded">
                            <TR height="1"><TD port="up" border="0" height="1" colspan="2"></TD></TR>
                            <TR>
                                <TD colspan="2"><B>$layer_type</B></TD>
                            </TR>
                            <TR>
                                <TD>Query: $q_proj</TD><TD>Heads: $(node.nheads)</TD>
                            </TR>
                            <TR>
                                <TD>Key: $k_proj</TD><TD>Value: $v_proj</TD>
                            </TR>$params_row
                            <TR height="1"><TD port="down" border="0" height="1" colspan="2"></TD></TR>
                        </TABLE>
                        >];"""
    link = "node$(node_id-1):down:c -> node$node_id:up:c;"
    return header, link, node_id
end

function draw_pooling_node(node, node_id)
    layer_type = String(nameof(typeof(node)))
    node_id += 1
    layer_specs = hasproperty(node, :out) ? node.out : node.k
    header = """node$node_id [shape=none margin=0 label=<
                        <TABLE border="0" cellborder= "1" style="rounded">
                            <TR height="1"><TD port="up" border="0" height="1"></TD></TR>
                            <TR>
                                <TD><B>$layer_type</B></TD>
                            </TR>
                            <TR>
                                <TD>$layer_specs</TD>
                            </TR>
                            <TR height="1"><TD port="down" border="0" height="1"></TD></TR>
                        </TABLE>
                        >];"""
    link = "node$(node_id-1):down:c -> node$node_id:up:c;"
    return header, link, node_id
end

function draw_skip_connection(node, node_id)
    first = node_id + 1
    headers, links, node_id = symbol_analysis(node.layers, node_id, "", "")
    last = node_id + 1
    headers, links, node_id = symbol_analysis(node.connection, node_id, headers, links)
    links = "$links\nnode$first:up:c -> node$(last):up:c [color=\"red\" constraint=false];"
    return headers, links, node_id
end

function draw_standard_node(node, node_id)
    layer_type = String(nameof(typeof(node)))
    node_id += 1
    activation = hasproperty(node, :σ) ? "\n<TD><I>$(Symbol(node.σ))</I></TD>" : ""
    colspan = activation == "" ? 1 : 2
    layer_specs = if hasproperty(node, :weight)
        reverse(size(node.weight))
    else
        "($(size(node.cell.Wi, 2)),$(size(node.cell.Wi, 1)))"
    end
    params = format_params(count_params(node))
    params_row = isempty(params) ? "" :
        "\n<TR><TD colspan=\"$colspan\"><FONT POINT-SIZE=\"9\">$params</FONT></TD></TR>"
    header = """node$node_id [shape=none margin=0 label=<
                        <TABLE border="0" cellborder= "1" style="rounded">
                            <TR height="1"><TD port="up" border="0" height="1" colspan="$colspan"></TD></TR>
                            <TR>
                                <TD colspan="$colspan"><B>$layer_type</B></TD>
                            </TR>
                            <TR>
                                <TD>$layer_specs</TD>$activation
                            </TR>$params_row
                            <TR height="1"><TD port="down" border="0" height="1" colspan="$colspan"></TD></TR>
                        </TABLE>
                        >];"""
    link = "node$(node_id-1):down:c -> node$node_id:up:c;"
    return header, link, node_id
end

function draw_recursive_node(node, node_id)
    layer_type = String(nameof(typeof(node)))
    headers, links, final_node_id = symbol_analysis(node.cell, node_id, "", "", remove_first_link = true)
    headers = """
    subgraph cluster_$final_node_id {
        label="$layer_type"
        $headers
    }
    """
    links = "$links\nnode$(node_id):down:c -> node$(node_id+1):up:c [lhead=cluster_$final_node_id];"
    return headers, links, final_node_id
end

function draw_container_node(node, node_id)
    prev_node_id = node_id
    layer_type = String(nameof(typeof(node)))
    params = format_params(count_params(node))
    label = isempty(params) ? layer_type : "$layer_type<BR/><FONT POINT-SIZE=\"9\">$params</FONT>"
    firsts = []
    lasts = []
    headers = ""
    links = ""
    for (i, path) in enumerate(node.layers)
        push!(firsts, node_id + 1)
        headers, links, node_id = symbol_analysis(path, node_id, headers, links,
            remove_first_link = prev_node_id != 0 || !(i == 1))
        push!(lasts, node_id)
    end
    if hasproperty(node, :connection)
        headers, _, node_id = symbol_analysis(node.connection, node_id, headers, links)
    else
        headers, _, node_id = symbol_analysis(max, node_id, headers, links)
    end
    for last in lasts
        links = "$links\nnode$last:down:c -> node$node_id:up:c;"
    end
    for first in firsts
        links = prev_node_id > 0 ? "$links\nnode$prev_node_id:down:c -> node$first:up:c;" : links
    end
    subgraph = """
    subgraph cluster_$node_id {
        label=<$label>
        $headers
    }
    """
    return subgraph, links, node_id
end

# ---------------------------------------------------------------------------
# Test structs (to be moved to test/runtests.jl before General Registry release)
# ---------------------------------------------------------------------------

struct SingleBranchWrapper
    inner::Chain
end
Flux.@layer SingleBranchWrapper

struct ResidualBlock
    main::Chain
    skip::Dense
end
Flux.@layer ResidualBlock

struct MultiHeadBlock
    attn::MultiHeadAttention
    proj::Dense
    norm::LayerNorm
end
Flux.@layer MultiHeadBlock

# ---

function __tests_draw_network__()
    draw_network(Chain(
        Dense(20, 50, relu),
        Dense(50, 10, sigmoid),
        Dense(10, 1, identity)
    ), save_to="examples/graph_1.png")
    sleep(1)

    draw_network(Chain(
        Embedding(5=>10),
        Dense(10, 1, leakyrelu)
    ), save_to="examples/graph_2.png")
    sleep(1)

    draw_network(Chain(
        Dense(20, 30, relu),
        softmax,
        Dense(30, 1)
    ), save_to="examples/graph_3.png")
    sleep(1)

    draw_network(Chain(
        Conv((5,5,5), 3 => 7),
        ConvTranspose((2, 3), 5 => 3),
        CrossCor((2,), 3 => 6),
        Dense(6, 1),
    ), save_to="examples/graph_4.png")
    sleep(1)

    draw_network(Chain(
        MaxPool((5,)),
        GlobalMeanPool,
        AdaptiveMaxPool((25, 25))
    ), save_to="examples/graph_5.png")
    sleep(1)

    draw_network(Chain(
        MultiHeadAttention(64=>1024=>1024 ,nheads = 8),
        MultiHeadAttention(32 ,nheads = 2),
        Dense(32, 1)
    ), save_to="examples/graph_6.png")
    sleep(1)

    draw_network(Chain(
        Dense(10, 5),
        LSTM(5=>10),
        Dense(10, 1, leakyrelu)
    ), save_to="examples/graph_7.png")
    sleep(1)

    draw_network(Chain(
        Parallel(vcat,
            Chain(
                Embedding(4=>100),
                Flux.flatten,
                LSTM(31*100=>100+20)
            ),
            Chain(
                Embedding(50=>80),
                Flux.flatten,
            )
        ),
        Dense(100+80+20, 400, relu),
        Dense(400, 750, relu),
        Dense(750, 1, identity),
        vec
    ), save_to="examples/graph_8.png")
    sleep(1)

    draw_network(Chain(
        Dense(20, 50, relu),
        SingleBranchWrapper(Chain(Dense(50, 30, relu), Dense(30, 20))),
        Dense(20, 1)
    ), save_to="examples/graph_9.png")
    sleep(1)

    draw_network(Chain(
        Dense(20, 20, relu),
        ResidualBlock(
            Chain(Dense(20, 20, relu), Dense(20, 20)),
            Dense(20, 20)
        ),
        Dense(20, 1)
    ), save_to="examples/graph_10.png")
    sleep(1)

    draw_network(Chain(
        Dense(64, 64, relu),
        MultiHeadBlock(
            MultiHeadAttention(64, nheads=4),
            Dense(64, 64),
            LayerNorm(64)
        ),
        Dense(64, 1)
    ), save_to="examples/graph_11.png")
    sleep(1)

    draw_network(
        SingleBranchWrapper(Chain(
            Dense(32, 32, relu),
            ResidualBlock(
                Chain(Dense(32, 32, relu), Dense(32, 32)),
                Dense(32, 32)
            ),
            Dense(32, 1)
        ))
    , save_to="examples/graph_12.png")
end

end
