using Flux
include("hupo.jl")

function collectData!(net_top, net_bot, memory_buffer)
  k = 1
  while k <= memory_buffer.N
    k = game!(net_top, net_bot, memory_buffer, k)
  end
end

mb = memory_buffer(500)
collectData!(net_top, net_bot, mb)
sum(mb.rewards.==0.)
find(mb.rewards.==0)
