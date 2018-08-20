mutable struct memory_buffer
  N::Int
  states::Matrix{Int}
  moves::Vector{Int}
  passes::Vector{Int}
  rewards::Vector{Float64}
end

function memory_buffer(N::Int)
  memory_buffer(N, zeros(Int, 18,N), zeros(Int, N), zeros(Int, N), zeros(N))
end


function game!(net_top_move, net_top_pass, net_bot_move, net_bot_pass,
               mb, k,  r_end = 1., r_add = 5., discount = 0.8, length_of_game_tolerance = 500)
    state = Array{Int}(undef,6*2+6)
    fill_state_beginning!(state)
    active_player = :top
    k_init = k
    won = Symbol()

    while true
        a = active_player == :top ? action(state, net_top_move, net_top_pass) : action(state, net_bot_move, net_bot_pass)
        if (active_player == :top) && (k <= mb.N)# collect data for top player
            mb.states[:,k] = state
            mb.moves[k] = Int(a[1])+1
            mb.passes[k] = a[2]
            k += 1
        end

        won, active_player = execute!(state,a)
        if k - k_init > length_of_game_tolerance
          won = :bottom_player_won
        end

        if won ∈ (:top_player_won, :bot_player_lost_a_stone)
            mb.rewards[k-1] += r_add
        end

        if won ∈ (:top_player_won, :bottom_player_won)
            reward = [discount.^collect((length(k_init:min(k - 1,mb.N))-1):-1:1); r_end]
            if won==:top_player_won
                mb.rewards[k_init:min(k - 1,mb.N)] .+= reward
            else
                mb.rewards[k_init:min(k - 1,mb.N)] .-= reward
            end
            return k
        end
    end
end


function collectData!(net_top_move, net_top_pass, net_bot_move, net_bot_pass,
                      lengthOfBuffer, r_end, r_add, discount, length_of_game_tolerance)
  mb = memory_buffer(lengthOfBuffer)
  k = 1
  while k <= mb.N
    k = game!(net_top_move, net_top_pass, net_bot_move, net_bot_pass, mb, k, r_end, r_add, discount, length_of_game_tolerance)
  end

  data = mb.states, mb.moves, mb.passes, (mb.rewards .- mean(mb.rewards))./std(mb.rewards)
end
