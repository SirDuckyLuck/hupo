mutable struct memory_buffer
  N::Int
  states::Array{Float64,2}
  moves::Vector{Float64}
  passes::Vector{Float64}
  rewards::Vector{Float64}
end

function memory_buffer(N::Int)
  memory_buffer(N, zeros(18,N), zeros(N), zeros(N), zeros(N))
end


function game!(net_top_move, net_top_pass, net_bot_move, net_bot_pass,
               mb, k,  r_end = 1., r_add = 5., discount = 0.8)
    state = Array{Int}(undef,6*2+6)
    fill_state_beginning!(state)
    active_player = "top"
    k_init = copy(k)

    while true
        a = active_player == "top" ? action(state, net_top_move, net_top_pass) : action(state, net_bot_move, net_bot_pass)
        if (active_player == "top") && (k <= mb.N)# collect data for top player
            mb.states[:,k] = state
            mb.moves[k] = Int(a[1])+1
            mb.passes[k] = a[2]
            k += 1
        end

        won, active_player = execute!(state,a)

        if won ∈ ["top player won" "bot player lost a stone"]
            mb.rewards[k-1] += r_add
        end

        if won ∈ ["top player won" "bottom player won"]
            reward = [discount.^collect((length(k_init:min(k - 1,mb.N))-1):-1:1); r_end]
            if won=="top player won"
                mb.rewards[k_init:min(k - 1,mb.N)] .+= reward
            else
                mb.rewards[k_init:min(k - 1,mb.N)] .-= reward
            end
            return k
        end
    end
end


function collectData!(net_top_move, net_top_pass, net_bot_move, net_bot_pass,
                      lengthOfBuffer, r_end, r_add, discount)
  mb = memory_buffer(lengthOfBuffer)
  k = 1
  while k <= mb.N
    k = game!(net_top_move, net_top_pass, net_bot_move, net_bot_pass, mb, k, r_end, r_add, discount)
  end

  data = mb.states, mb.moves, mb.passes, (mb.rewards .- mean(mb.rewards))./std(mb.rewards)
end
