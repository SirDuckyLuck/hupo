using Flux
using Flux.Tracker
include("hupo.jl")

function collectData(numOfGames, net_top, net_bot, exploration)
  states_top = Array{Int}(undef,0,18)
  rewards_top = Vector{Float64}()
  move_top = Vector{Float64}()
  states_bot = Array{Int}(undef,0,18)
  rewards_bot = Vector{Float64}()
  move_bot = Vector{Float64}()

  for i in 1:numOfGames
    st, rt, pt, sb, rb, pb = game(net_top, net_bot, exploration)
    states_top = vcat(states_top,st)
    states_bot = vcat(states_bot,sb)
    rewards_top = vcat(rewards_top,rt)
    rewards_bot = vcat(rewards_bot,rb)
    move_top = vcat(move_top,pt)
    move_bot = vcat(move_bot,pb)
  end

  states_top, rewards_top, move_top, states_bot, rewards_bot, move_bot
end

net_top = Chain(
  Dense(18, 100, relu),
  Dense(100, 24),
  softmax)

net_bot = Chain(
  Dense(18, 100, relu),
  Dense(100, 24),
  softmax)

st, rt, mt, sb, rb, mb = collectData(50, net_top, net_bot, 0.0)

function loss(x,y)
    p_all = net_top(x[:,1:end-1]')
    p = [p_all[Int(x[i,end]),i] for i in 1:size(p_all,2)]
    -sum(log.(p) .* y)
end

x = hcat(st,pt)
y = rt
data = Iterators.repeated((x, y), 1)
opt = ADAM(Flux.params(net_top))

Flux.train!(loss, data, opt)
loss(x,rt)
