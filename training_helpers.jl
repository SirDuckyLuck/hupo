using Flux
include("hupo.jl")

function collectData(numOfGames, net_top, net_bot)
  states_top = Array{Int}(undef,0,18)
  rewards_top = Vector{Float64}()
  move_top = Vector{Float64}()
  states_bot = Array{Int}(undef,0,18)
  rewards_bot = Vector{Float64}()
  move_bot = Vector{Float64}()

  for i in 1:numOfGames
    st, rt, pt, sb, rb, pb = game(net_top, net_bot)
    states_top = vcat(states_top,st)
    states_bot = vcat(states_bot,sb)
    rewards_top = vcat(rewards_top,rt)
    rewards_bot = vcat(rewards_bot,rb)
    move_top = vcat(move_top,pt)
    move_bot = vcat(move_bot,pb)
  end

  states_top, rewards_top, move_top, states_bot, rewards_bot, move_bot
end
