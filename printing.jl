include("UnicodeGrids.jl")
using .UnicodeGrids

const d_moves = Dict(up => "↑", right => "→", down => "↓", left => "←", out => "~")
const d_used_stones = Dict(1 => "₁", 2 => "₂", 3 => "₃", 4 => "₄", 5 => "₅", 6 => "₆")


function game_show(net_top, net_bot)
  state = state_beginning()
  active_player = :top
  round_number = 0

  println()
  println("Round $(round_number)")
  print_state(state)
  print_action_probs(get_action_probs(state))
  println("press <Enter>\n\n\033[2A")

  while true
    idx = get_active_stone(state)
    pos = (state[2*idx - 1], state[2*idx])

    move, pass = active_player == :top ?
           sample_action(state, net_top) :
           sample_action(state, net_bot)
    active_stone = apply_move!(state, move)
    apply_pass!(state, active_stone, pass)
    won = check_state(state)
    active_player = get_active_player(state)

    round_number += 1
    readline()
    clear(16)
    println("Round $(round_number)")
    print_state(state)
    print_action_probs(get_action_probs(state))
    println("player $idx moves $(d_moves[move])  and passes token to player $(pass)")

    if won ∈ (:top_player_won, :bottom_player_won)
      println(won)
      break
    end
  end
end


function print_state(state::Array{Int})
  M = fill(" ", 5, 3)
  M[3,2] = "x"
  x, y = 0, 0
  for i in 1:6
    state[12+i] == -1 && continue
    c = string(i)
    state[12+i] == 1 && (c = d_used_stones[i])
    state[12+i] == 2 && ((x, y) = (state[2i-1], state[2i]))
    M[state[2i-1], state[2i]] = c
  end
  println(UnicodeGrids.grid(M, (x, y)))
end


"Clear `n` lines above cursor."
function clear(n::Int)
  println("\033[$(n)A" * "\033[K\n" ^ n * "\033[$(n)A")
end


function get_action_probs(state)
  active_player = get_active_player(state)
  net = active_player == :top ? net_top : net_bot
  return policy(state, net)
end


function print_action_probs(probs, width = 80)
  perm = sortperm(probs, rev = true)
  sb = "\e[1m"
  width_left = width
  p_left = 1.0
  for (i, a) ∈ enumerate(perm)
    p = probs[a]
    move, pass = idx2MovePass(a)
    c = i % 2 == 0 ? "\e[32m\e[41m" : "\e[31m\e[42m"
    w = min(ceil(Int, p / p_left * width_left), width_left)
    w == 0 && break
    s = w ≥ 3 ? "$(d_moves[move]) $pass" : ""
    sb *= c * rpad(s, w)
    p_left -= p
    width_left -= w
    width_left == 0 && break
  end
  sb *= " " ^ width_left * "\e[0m"
  println(sb)
end
