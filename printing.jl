const d_moves = Dict(up => "↑", right => "→", down => "↓", left => "←", out => "~")

function game_show(net_top, net_bot)
  state = Array{Int}(6*2+6)
  fill_state_beginning!(state)
  active_player = :top
  round_number = 1

  println()
  println("Round $(round_number)")
  print_state(state)
  println()
  println("press <Enter>")

  while true
    readline()
    clear(16)

    idx = get_active_stone(state)
    pos = (state[2*idx - 1], state[2*idx])

    move, pass = active_player == :top ?
           sample_action(state, net_top) :
           sample_action(state, net_bot)
    active_stone = apply_move!(state, move)
    apply_pass!(state, active_stone, pass)
    won, active_player = check_state(state)

    println("Round $(round_number)")
    print_state(state)
    println("player $idx moves $(d_moves[move])  and passes token to player $(pass)")
    active_player == :top ? println(get_probabilities(state, net_top)) : println(get_probabilities(state, net_bot))

    if won ∈ (:top_player_won, :bottom_player_won)
        println(won)
        break
    end
    round_number += 1
  end
end


function print_state(state::Array{Int})
  M = fill(" ", 5, 3)
  M[3,2] = "x"
  for i in 1:6
    state[12+i] == -1 && continue
    c = string(i)
    state[12+i] == 1 && (c = aesRed * c * aesClear)
    state[12+i] == 2 && (c = aesBold * aesYellow * c * aesClear)
    id = i*2-1
    M[state[id],state[id+1]] = c
  end
  print_grid(M)
end


"Clear `n` lines above cursor."
function clear(n::Int)
  println("\033[$(n)A" * "\033[K\n" ^ n * "\033[$(n)A")
end


function get_probabilities(state::Array{Int}, net, k::Int = 3)
  p = policy(state, net)
  v = Array{Any}(0,3)

  for i in eachindex(p)
    if p[i] > 0
      v = vcat(v, [p[i] idx2MovePass(i)[1] idx2MovePass(i)[2]])
    end
  end

  v[sortperm(-v[:,1])[1:min(k,size(v,1))], :]
end
