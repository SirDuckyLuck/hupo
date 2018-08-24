mutable struct memory_buffer
  N::Int
  states_before_moves::Matrix{Int}
  moves::Vector{Int}
  rewards::Vector{Float64}
end

function memory_buffer(N::Int)
  memory_buffer(N, zeros(Int, 5, N), zeros(Int, N), zeros(N))
end


function game!(net_top_move, net_bot_move, mb, k, r_end, discount, length_of_game_tolerance)
  state = Array{Int}(undef,5)
  fill_state_beginning!(state)
  active_player = :top
  k_init = k
  game_length = 0

  while true
    game_length += 1

    move = active_player == :top ? sample_move(state, net_top_move) : sample_move(state, net_bot_move)
    if (active_player == :top) && (k <= mb.N)# collect data for top player
      mb.states_before_moves[:,k] = state
      mb.moves[k] = Int(move)
      k += 1
    end
    apply_move!(state, move)
    won = check_state(state)

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


function collectData(net_top_move, net_bot_move,
                     lengthOfBuffer, r_end, discount, length_of_game_tolerance)
  mb = memory_buffer(lengthOfBuffer)
  k = 1
  while k <= mb.N
    k = game!(net_top_move, net_bot_move, mb, k, r_end, discount, length_of_game_tolerance)
  end

  data = mb.states_before_moves, mb.moves, mb.rewards #(mb.rewards .- mean(mb.rewards))./std(mb.rewards)
end


# data = collectData(net_top_move, net_bot_move, lengthOfBuffer, r_end, discount, length_of_game_tolerance)
