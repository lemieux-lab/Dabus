# using GraphViz

# Note: We're using an online GraphViz API since the GraphViz
# Julia wrapper hasn't been updated in years and doesn't work
# properly
using HTTP; const http=HTTP
using Flux

export draw_network

function http_graph(graph::String; output_type="png"::String)
    r = http.get("https://quickchart.io/graphviz", query=Dict("format"=>output_type, "graph"=>graph))
    return r.body
end


"""
    draw_network(network; save_to=nothing, output_type="png")

Draws a given Flux network using graphviz. Returns the image of the graph as the image
If `save_to` is a given a path, also writes the image bytes to that path.

# Examples
```julia-repl
julia> draw_network(Chain(Dense(50, 10, relu), Dense(10, 1)), save_to="my_path/network_graph.png")
4546-element Vector{UInt8}:
 0x89
 0x50
 0x4e
 0x47
 0x0d
 0x0a
    ⋮
 0x4e
 0x44
 0xae
 0x42
 0x60
 0x82
```
"""
function draw_network(network; save_to=nothing, output_type="png")
    headers, links, node_id = symbol_analysis(network, 0, "", "")
    # println(links)
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
    graph = replace(graph, " => " => ", ")
    # println(graph)
    img_bytes = http_graph(graph, output_type=output_type)
    if !isnothing(save_to)
        open(save_to, "w") do handle
            write(handle, img_bytes)
        end
    end
    return img_bytes
end

function draw_type(node::Any, node_id)
    if hasproperty(node, :weight) || hasproperty(node, :cell)
        return draw_standard_node(node, node_id)
    else
        draw_misc_node(node, node_id)
    end
end
draw_type(node::Union{Conv, ConvTranspose, CrossCor}, node_id) = draw_convolution_node(node, node_id)
draw_type(node::Union{MultiHeadAttention}, node_id) = draw_transformer_node(node, node_id)
draw_type(node::Union{AdaptiveMaxPool, MaxPool, MeanPool, AdaptiveMeanPool}, node_id) = draw_pooling_node(node, node_id)
draw_type(node::Union{Parallel, Maxout, PairwiseFusion}, node_id) = draw_container_node(node, node_id)
draw_type(node::Union{RNNCell}, node_id) = draw_recursive_node(node, node_id)
draw_type(node::Union{SkipConnection}, node_id) = draw_skip_connection(node, node_id)
draw_type(node::Chain, node_id) = draw_chain(node, node_id)

function symbol_analysis(node, node_id, headers, links; remove_first_link=false)
    # special_category_switch = Dict(
    #     ([:Conv, :ConvTranspose, :CrossCor, :DepthwiseConv] .=> draw_convolution_node)...,
    #     ([:MultiHeadAttention] .=> draw_transformer_node)...,
    #     ([:AdaptiveMaxPool, :MaxPool, :MeanPool, :AdaptiveMeanPool,] .=> draw_pooling_node)...,  # Global pooling is handled by misc
    #     ([:Parallel, :Maxout, :PairwiseFusion] .=> draw_container_node)...,
    #     ([:Recur] .=> draw_recursive_node)...,
    #     ([:SkipConnection] .=> draw_skip_connection)...,  # Technically a container, but way too different to handle in the same way
    #     ([:Chain] .=> draw_chain)... # Also technically a container, but is the base of the recursion
    # )
    
    header, link, node_id = draw_type(node, node_id)

    # if haskey(special_category_switch, nameof(typeof(node)))
    #     header, link, node_id = special_category_switch[nameof(typeof(node))](node, node_id)
    # elseif hasproperty(node, :weight) || hasproperty(node, :cell)
    #     header, link, node_id = draw_standard_node(node, node_id)
    # else
    #     header, link, node_id = draw_misc_node(node, node_id)
    # end

    if remove_first_link  # Removes the first link. Useful when this accumulates link with a Chain by a Parallel or similar
        tmp = IOBuffer(link)
        readline(tmp)
        link = String(read(tmp))
    end

    headers = "$headers\n\t$header"
    links = node_id > 1 && link!= "" ? strip("\t$links\n$link") : links
    return headers, links, node_id
end

function draw_chain(node, node_id)
    headers = ""
    links = ""
    for layer in node
       headers, links, node_id = symbol_analysis(layer, node_id, headers, links)
    #    headers = "$headers\n\t$sub_header"
    #    links = "\t$links\n$sub_link"
    end
    # Using the hash of the node id as a nice trick to get more unique ids out of the node id for clusters
    headers = """
    subgraph cluster_$(hash(node_id)){
        label="";
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
    header = """node$node_id [shape=none margin=0 label=<
                        <TABLE border="0" cellborder= "1" style="rounded">
                            <TR height="1"><TD port="up" border="0" height="1" colspan="3"></TD></TR>
                            <TR>
                                <TD colspan="3"><B>$layer_type</B></TD>
                            </TR>
                            <TR>
                                <TD>$layer_specs</TD><TD>$layer_kernel</TD>$activation
                            </TR>
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
                            </TR>
                            <TR height="1"><TD port="down" border="0" height="1" colspan="2"></TD></TR>
                        </TABLE>
                        >];"""
    link = "node$(node_id-1):down:c -> node$node_id:up:c;"
    return header, link, node_id
end

function draw_pooling_node(node, node_id)
    layer_type = String(nameof(typeof(node)))
    node_id += 1
    if hasproperty(node, :out)
        layer_specs = node.out
    else
        layer_specs = node.k
    end
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
    layer_type = String(nameof(typeof(node)))
    first = node_id+1
    headers, links, node_id = symbol_analysis(node.layers, node_id, "", "")
    last = node_id+1
    headers, links, node_id = symbol_analysis(node.connection, node_id, headers, links)

    links = "$links\nnode$first:up:c -> node$(last):up:c [color=\"red\" constraint=false];"
    return headers, links, node_id
end

function draw_standard_node(node, node_id)
    layer_type = String(nameof(typeof(node)))
    node_id += 1
    # activation = hasproperty(node, :σ) ? "|<activation> $(Symbol(node.σ))" : ""
    activation = hasproperty(node, :σ) ? "\n<TD><I>$(Symbol(node.σ))</I></TD>" : ""
    colspan = activation == "" ? 1 : 2
    if hasproperty(node, :weight)
        layer_specs = reverse(size(node.weight))
    else
        layer_specs = "($(size(node.cell.Wi, 2)),$(size(node.cell.Wi, 1)))"
    end
    header = """node$node_id [shape=none margin=0 label=<
                        <TABLE border="0" cellborder= "1" style="rounded">
                            <TR height="1"><TD port="up" border="0" height="1" colspan="$colspan"></TD></TR>
                            <TR>
                                <TD colspan="$colspan"><B>$layer_type</B></TD>
                            </TR>
                            <TR>
                                <TD>$layer_specs</TD>$activation
                            </TR>
                            <TR height="1"><TD port="down" border="0" height="1" colspan="$colspan"></TD></TR>
                        </TABLE>
                        >];"""
    link = "node$(node_id-1):down:c -> node$node_id:up:c;"
    return header, link, node_id
end

function draw_recursive_node(node, node_id)
    layer_type = String(nameof(typeof(node)))
    # activation = hasproperty(node, :σ) ? "|<activation> $(Symbol(node.σ))" : ""
    headers, links, final_node_id = symbol_analysis(node.cell, node_id, "", "", remove_first_link=true)
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
    firsts = []
    lasts = []
    headers = ""
    links = ""
    for (i, path) in enumerate(node.layers)
        push!(firsts, node_id+1)
        headers, links, node_id = symbol_analysis(path, node_id, headers, links, remove_first_link=prev_node_id!=0 || !(i==1))
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
    # We use node_id for subgraph id as a conveniant unique id number. This means that the id of the cluster_
    # is always the same as the id of its last node, which should be used for the next link anyway (hell yeah)
    subgraph = """
    subgraph cluster_$node_id {
        label="$layer_type"
        $headers
    }
    """
    return subgraph, links, node_id
end

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
end
