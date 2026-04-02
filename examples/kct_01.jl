using JuBox
using Kmers

@time kct_a = Kct("data/kmers.31.jf")
@time kct_b = Kct("data/second.31.jf")
# To test, this can link to: /home/golem/GDC/brca/0aabc72c-442a-4a87-8bb8-9c4fd512e0bf/kmers.31.jf

using BenchmarkTools

b = DNA31mer("TTTTTTTTTTAAAAAAAAAGAAAAAAAAAAA")

n = 48000
seed = rand(UInt64, n)
seq = [DNA31mer((s,)) for s in seed]

@btime z = [kct_a[s] for s in seq]

@time JuBox.save(kct_a, "data/test.bson")
@time test = JuBox.load("data/test.bson")

@btime z = [test[s] for s in seq]


@time ab = merge(kct_a, kct_b)
@time abab = merge(ab, ab)

save(ab, "data/test3.bson")
test = load("data/test3.bson")