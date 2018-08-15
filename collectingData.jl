mutable struct memory_buffer
  N::Int
  states::Array{Int}
  actions::Vector{Float64}
  rewards::Vector{Float64}
end

function memory_buffer(N::Int)
  memory_buffer(N, Array{Int}(undef,18,N), zeros(N), zeros(N))
end


function game!(net_top, net_bot, memory_buffer, k,  r_end = 1., r_add = 5., discount = 0.8)
    state = Array{Int}(undef,6*2+6)
    fill_state_beginning!(state)
    active_player = "top"
    k_init = copy(k)

    while true
        a, a_idx = active_player == "top" ? action(state, net_top) : action(state, net_bot)
        if (active_player=="top") && (k <= memory_buffer.N)# collect data for top player
            memory_buffer.states[:,k] = state
            memory_buffer.actions[k] = a_idx
            k += 1
        end

        won = execute!(state,a)

        if won ∈ ["top player won" "bot player lost a stone"]
            memory_buffer.rewards[k-1] += r_add
        end

        if won ∈ ["top player won" "bottom player won"]
            reward = [r_end.^collect((length(k_init:min(k - 1,memory_buffer.N))-1):-1:1); r_end]
            if won=="top player won"
                memory_buffer.rewards[k_init:min(k - 1,memory_buffer.N)] .+= reward
            else
                memory_buffer.rewards[k_init:min(k - 1,memory_buffer.N)] .-= reward
            end
            return k
        end

        active_player = active_player == "top" ? "bottom" : "top"
    end
end


function collectData!(net_top, net_bot, memory_buffer, r_end = 1., r_add = 5., discount = 0.8)
  k = 1
  while k <= memory_buffer.N
    k = game!(net_top, net_bot, memory_buffer, k, r_end, r_add, discount)
  end
end
