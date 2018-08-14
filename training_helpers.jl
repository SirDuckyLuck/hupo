using Flux
include("hupo.jl")

function collectData!(net_top, net_bot, memory_buffer)
  k = 1
  while k <= memory_buffer.N
    k = game!(net_top, net_bot, memory_buffer, k)
  end
end
