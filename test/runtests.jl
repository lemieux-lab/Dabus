using Test
using Flux
using Dabus

# Custom @layer structs used across tests

struct SequentialWrapper
    inner::Chain
end
Flux.@layer SequentialWrapper

struct ResidualBlock
    main::Chain
    skip::Dense
end
Flux.@layer ResidualBlock

struct TransformerBlock
    attn::MultiHeadAttention
    proj::Dense
    norm::LayerNorm
end
Flux.@layer TransformerBlock

# Helpers

node_count(dot) = count(r"<B>", dot)  # each rendered node has exactly one <B>label</B>

@testset "Dabus" begin

    @testset "DOT structure" begin
        dot = network_to_dot(Dense(10, 5, relu))
        @test occursin("digraph network {", dot)
        @test occursin("rankdir=TB", dot)
        @test occursin("subgraph repr", dot)
    end

    @testset "Standard layers" begin
        dot = network_to_dot(Chain(Dense(20, 10, relu), Dense(10, 1, sigmoid)))
        @test occursin("Dense", dot)
        @test occursin("relu", dot)
        @test occursin("σ", dot)
        @test node_count(dot) == 2
    end

    @testset "Convolution layers" begin
        dot = network_to_dot(Chain(Conv((3, 3), 1 => 8, relu), Dense(8, 1)))
        @test occursin("Conv", dot)
        @test occursin("Dense", dot)
    end

    @testset "Pooling layers" begin
        dot = network_to_dot(Chain(MaxPool((2,)), Dense(4, 1)))
        @test occursin("MaxPool", dot)
    end

    @testset "Transformer layers" begin
        dot = network_to_dot(Chain(MultiHeadAttention(32, nheads=4), Dense(32, 1)))
        @test occursin("MultiHeadAttention", dot)
        @test occursin("Dense", dot)
    end

    @testset "LSTM" begin
        dot = network_to_dot(Chain(Dense(10, 5), LSTM(5 => 10), Dense(10, 1)))
        @test occursin("LSTM", dot)
        @test occursin("Dense", dot)
    end

    @testset "Chain produces a cluster" begin
        dot = network_to_dot(Chain(Dense(5, 5), Dense(5, 1)))
        @test occursin("subgraph cluster_", dot)
    end

    @testset "Parallel container" begin
        dot = network_to_dot(Parallel(+, Dense(5, 5), Dense(5, 5)))
        @test occursin("Parallel", dot)
        @test occursin("subgraph cluster_", dot)
        @test node_count(dot) >= 2
    end

    @testset "SkipConnection" begin
        dot = network_to_dot(SkipConnection(Dense(8, 8, relu), +))
        @test occursin("Dense", dot)
    end

    @testset "Custom @layer - single sublayer" begin
        layer = SequentialWrapper(Chain(Dense(10, 5, relu), Dense(5, 3)))
        dot = network_to_dot(layer)
        @test occursin("SequentialWrapper", dot)  # cluster label
        @test occursin("Dense", dot)
        @test node_count(dot) == 2
    end

    @testset "Custom @layer - multiple sublayers" begin
        layer = ResidualBlock(
            Chain(Dense(10, 10, relu), Dense(10, 10)),
            Dense(10, 10)
        )
        dot = network_to_dot(layer)
        @test occursin("ResidualBlock", dot)  # cluster label + merge node
        @test count(r"Dense", dot) >= 3  # 2 in main chain + 1 skip
        @test occursin("subgraph cluster_", dot)
        @test node_count(dot) >= 4  # 2 Dense in chain + 1 skip Dense + 1 ResidualBlock merge
    end

    @testset "Custom @layer - three sublayers" begin
        layer = TransformerBlock(
            MultiHeadAttention(32, nheads=4),
            Dense(32, 32),
            LayerNorm(32)
        )
        dot = network_to_dot(layer)
        @test occursin("TransformerBlock", dot)
        @test occursin("MultiHeadAttention", dot)
        @test occursin("Dense", dot)
    end

    @testset "Custom @layer - nested" begin
        layer = SequentialWrapper(Chain(
            Dense(16, 16, relu),
            ResidualBlock(
                Chain(Dense(16, 16, relu), Dense(16, 16)),
                Dense(16, 16)
            )
        ))
        dot = network_to_dot(layer)
        @test occursin("SequentialWrapper", dot)
        @test occursin("ResidualBlock", dot)
        @test occursin("Dense", dot)
    end

    @testset "Custom @layer - embedded in outer Chain" begin
        dot = network_to_dot(Chain(
            Dense(20, 20, relu),
            ResidualBlock(
                Chain(Dense(20, 20, relu), Dense(20, 20)),
                Dense(20, 20)
            ),
            Dense(20, 1)
        ))
        @test occursin("ResidualBlock", dot)
        @test occursin("Dense", dot)
        @test node_count(dot) >= 5  # outer Dense + 2 in chain + 1 skip + merge + outer Dense
    end

end
