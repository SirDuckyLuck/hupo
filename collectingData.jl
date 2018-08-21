mutable struct memory_buffer
  N::Int
  states_before_moves::Matrix{Int}
  moves::Vector{Int}
  states_before_passes::Matrix{Int}
  passes::Vector{Int}
  rewards::Vector{Float64}
end

function memory_buffer(N::Int)
  memory_buffer(N, zeros(Int, 18,N), zeros(Int, N), zeros(Int, 18,N), zeros(Int, N), zeros(N))
end


function game!(net_top_move, net_top_pass, net_bot_move, net_bot_pass,
               mb, k,  r_end = 1., discount = 0.8, length_of_game_tolerance = 500)
  state = Array{Int}(undef,6*2+6)
  fill_state_beginning!(state)
  active_player = :top
  active_stone = 2
  k_init = k
  won = Symbol()
  game_length = 0

  while true
    game_length += 1

    move = active_player == :top ? sample_move(state, active_stone, net_top_move) : sample_move(state, active_stone, net_bot_move)
    if (active_player == :top) && (k <= mb.N)# collect data for top player
      mb.states_before_moves[:,k] = state
      mb.moves[k] = Int(move)
    end
    active_stone = apply_move!(state, active_stone, move)
    pass = active_player == :top ? sample_pass(state, active_stone, net_top_pass) : sample_pass(state, active_stone, net_bot_pass)
    if (active_player == :top) && (k <= mb.N)
      mb.states_before_passes[:,k] = state
      mb.passes[k] = pass
      k += 1
    end
    active_stone = apply_pass!(state, active_stone, pass)
    won, active_player = check_state(state, active_stone)

    if k - k_init > length_of_game_tolerance
      won = :bottom_player_won
    end

    if won ∈ (:top_player_won, :bottom_player_won)
      if k_init + game_length <= mb.N
        reward = (discount .^ ((k - k_init - 1):-1:0)) .* r_end
        if won == :top_player_won
            mb.rewards[k_init:(k - 1)] .+= reward
        else
            mb.rewards[k_init:(k - 1)] .-= reward
        end
      end
      return k
    end
  end
end


function collectData(net_top_move, net_top_pass, net_bot_move, net_bot_pass,
                      lengthOfBuffer, r_end, discount, length_of_game_tolerance)
  mb = memory_buffer(lengthOfBuffer)
  k = 1
  while k <= mb.N
    k = game!(net_top_move, net_top_pass, net_bot_move, net_bot_pass, mb, k, r_end, discount, length_of_game_tolerance)
  end

  data = mb.states_before_moves, mb.moves, mb.states_before_passes, mb.passes, mb.rewards #(mb.rewards .- mean(mb.rewards))./std(mb.rewards)
end


# data = collectData(net_top_move, net_top_pass, net_bot_move, net_bot_pass, lengthOfBuffer, r_end, discount, length_of_game_tolerance)
