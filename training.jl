using Flux
using Flux.Tracker
include("hupo.jl")

function collectData(numOfGames, net_top, net_bot, exploration)
  states_top = Array{Int}(undef,0,18)
  rewards_top = Vector{Float64}()
  probability_top = Vector{Float64}()
  states_bot = Array{Int}(undef,0,18)
  rewards_bot = Vector{Float64}()
  probability_bot = Vector{Float64}()

  for i in 1:numOfGames
    st, rt, pt, sb, rb, pb = game(net_top, net_bot, exploration)
    states_top = vcat(states_top,st)
    states_bot = vcat(states_bot,sb)
    rewards_top = vcat(rewards_top,rt)
    rewards_bot = vcat(rewards_bot,rb)
    probability_top = vcat(probability_top,pt)
    probability_bot = vcat(probability_bot,pb)
  end

  states_top, rewards_top, probability_top, states_bot, rewards_bot, probability_bot
end

net_top = Chain(
  Dense(18, 100, relu),
  Dense(100, 24),
  softmax)

net_bot = Chain(
  Dense(18, 100, relu),
  Dense(100, 24),
  softmax)

st, rt, pt, sb, rb, pb = collectData(50, net_top, net_bot, 0.1)

-sum(log.(pt) .* rt)
Params(net_top)

f(x) = 3x^2 + 2x + 1

# df/dx = 6x + 2
f′(x) = Tracker.gradient(f, x)[1]

f′(2) # 14.0 (tracked)

# d²f/dx² = 6
f′′(x) = Tracker.gradient(f′, x)[1]

f′′(2) # 6.0 (tracked)

f(W, b, x) = W * x + b

Tracker.gradient(f, 2, 3, 4)

W = param(2) # 2.0 (tracked)
b = param(3) # 3.0 (tracked)

f(m,r) = r

p = Params(net_top)
grads = Tracker.gradient(() -> f(m,-1), p)

grads[W] # 4.0
grads[b] # 1.0
