@enum Move up = 1 right = 2 down = 3 left = 4 out = 5
const d_moves = Dict(up => "↑", right => "→", down => "↓", left => "←", out => "~")

function game_show(net_top_move, net_top_pass, net_bot_move, net_bot_pass)
    state = Array{Int}(undef,6*2+6)
    fill_state_beginning!(state)
    active_player = :top
    active_stone = 2
    round_number = 1

    println()
    println("Round $(round_number)")
    print_state(state)
    println("press <Enter>")

    while true
        readline()
        clear(15)

        idx = get_active_stone(state)
        pos = (state[2*idx - 1], state[2*idx])

        move = active_player == :top ? sample_move(state, active_stone, net_top_move) : sample_move(state, active_stone, net_bot_move)
        active_stone = apply_move!(state, active_stone, move)
        pass = active_player == :top ? sample_pass(state, active_stone, net_top_pass) : sample_pass(state, active_stone, net_bot_pass)
        active_stone = apply_pass!(state, active_stone, pass)
        won, active_player = check_state(state, active_stone)

        println("Round $(round_number)")
        print_state(state)
        println("player $idx moves $(d_moves[move])  and passes token to player $(pass)")

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
