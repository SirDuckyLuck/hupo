mutable struct memory_buffer
  N::Int
  states::Matrix{Int}
  actions::Vector{Int}
  rewards::Vector{Float64}
end


function action2idx(move::Move, pass::Int)
  (Int(move) - 1)*6 + pass
end


function memory_buffer(N::Int)
  memory_buffer(N, zeros(Int, 18, N), zeros(N), zeros(N))
end


function game!(net_top, net_bot, mb::memory_buffer,
               k::Int,  r_end::Float64 = 1., discount::Float64 = 0.8, length_of_game_tolerance::Int = 500)
  state = state_beginning(:random)
  active_player = :top
  k_init = k
  won = Symbol()
  game_length = 0

  while true
    game_length += 1
    move, pass = active_player == :top ?
                 sample_action(state, net_top) :
                 sample_action(state, net_bot)
    if (active_player == :top) && (k <= mb.N)
      mb.states[:,k] = state
      mb.actions[k] = action2idx(move, pass)
      k += 1
    end
    active_stone = apply_move!(state, move)
    apply_pass!(state, active_stone, pass)
    won, active_player = check_state(state)

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


function collectData(net_top, net_bot, lengthOfBuffer::Int = 300,
                     r_end::Float64 = 1., discount::Float64 = 0.8, length_of_game_tolerance::Int = 500)
  mb = memory_buffer(lengthOfBuffer)
  k = 1
  while k <= mb.N
    k = game!(net_top, net_bot, mb, k, r_end, discount, length_of_game_tolerance)
  end

  data = mb.states, mb.actions, (mb.rewards .- mean(mb.rewards))./std(mb.rewards)
end


# data = collectData(net_top, net_bot)
