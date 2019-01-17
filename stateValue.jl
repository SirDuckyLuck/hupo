
include("hupo.jl")

mutable struct sval_memory_buffer
  N::Int
  k::Int
  states::Matrix{Int}
  rewards::Vector{Float64}
  start_states::Matrix{Int}
  actions::Vector{Int}
end

abstract type AbstractPlayer end

include("actorPlayer.jl")
include("statePlayer.jl")


function action2idx(move::Move, pass::Int)
  (Int(move) - 1)*6 + pass
end

function sval_memory_buffer(N::Int)
  sval_memory_buffer(N, 0, zeros(Int, 18, N), zeros(N), zeros(Int, 18, N), zeros(N))
end

function state2hash(state::Array{Int})
  # here implement transforming state into a UInt64 integer
  return hash(state)
end

const epsilon = 0.1

const length_of_game_tolerance = 300
const discount = 0.95
const r_end = 1.
const alpha = 0.1


global sval_net_top = Chain(
    Dense(72, 100, relu),
    Dense(100,1))
    # Dense(72,1))
global net_player_top = NetPlayer(sval_net_top,ADAM(Flux.params(sval_net_top)),transformState)

global sval_net_bot = Chain(
    Dense(72, 100, relu),
    Dense(100,1))
global net_player_bot = NetPlayer(sval_net_bot,ADAM(Flux.params(sval_net_bot)),transformState)

global action_net_top = Chain(
    Dense(72, 100, relu),
    Dense(100,30),
    softmax)

global action_player_top = ActionPlayer(action_net_top,ADAM(Flux.params(action_net_top)),transformState)


global ac_player_top = ACPlayer(action_net_top,ADAM(Flux.params(action_net_top)),sval_net_top,ADAM(Flux.params(sval_net_top)),transformState)


global sval_top = Dict{UInt64,Float64}()

# global sval_bot = StatePlayer(Dict{UInt64,Float64}())

global sval_bot = Dict{UInt64,Float64}()


function computeStateValue!(;n_epochs = 1000, train_bot = false, train_top = false, level = :original)

  player_top = ac_player_top
  player_bot = sval_bot

  epoch = 0
  while epoch <= n_epochs

    if epoch % 1000 == 0
      top_wins = 0
      n_games = 0
      sum_game_lenghts = 0
    end
    epoch += 1

    mb_top = sval_memory_buffer(length_of_game_tolerance)
    mb_bot = sval_memory_buffer(length_of_game_tolerance)

    won, game_length = sval_game!(player_top, player_bot, mb_top, mb_bot, r_end, discount, length_of_game_tolerance, level)

    top_wins += won == :top_player_won ? 1 : 0
    n_games += 1
    sum_game_lenghts += game_length

    # check whether mb contains the right rewards at the right states
    global mb_top = mb_top
    global mb_bot = mb_bot

    if train_top
      train!(player_top, mb_top)
    end
    if train_bot
      train_monte_carlo_online!(player_bot,mb_bot)
    end

    if epoch % 1000 == 0
      println("Epoch: $(epoch)")
      top_win = round(top_wins / n_games * 100,1)
      avg_length = round(sum_game_lenghts / n_games,1)
      println("top won $top_win % against bot in $avg_length rounds")
    end
  end

end


function sval_game!(sval_top, sval_bot, mb_top, mb_bot, r_end, discount, length_of_game_tolerance, level)
  state = state_beginning(level)
  active_player = :top
  won = Symbol()
  game_length = 0

  while true
    game_length += 1
    move, pass = active_player == :top ?
                 sval_sample_action(state, sval_top) :
                 sval_sample_action(state, sval_bot)
    if (active_player == :top)
      mb = mb_top
    else
      mb = mb_bot
    end

    mb.k +=1
    mb.start_states[:,mb.k] = state
    mb.actions[mb.k] = action2idx(move, pass)

    active_stone = apply_move!(state, move)
    apply_pass!(state, active_stone, pass)
    won = check_state(state)
    active_player = get_active_player(state)

    mb.states[:,mb.k] = state

    if game_length > length_of_game_tolerance
      won = :bottom_player_won
    end

    if won ∈ (:top_player_won, :bottom_player_won)
      mb = mb_top
      if mb.k <= mb.N
        reward = (discount .^ ((mb.k - 1):-1:0)) .* r_end
        if won == :top_player_won
            mb.rewards[1:mb.k] .+= reward
        else
            mb.rewards[1:mb.k] .-= reward
        end
      end
      mb = mb_bot
      if mb.k <= mb.N
        reward = (discount .^ ((mb.k - 1):-1:0)) .* r_end
        if won == :bottom_player_won
            mb.rewards[1:mb.k] .+= reward
        else
            mb.rewards[1:mb.k] .-= reward
        end
      end
      return won, game_length
    end

  end
end
