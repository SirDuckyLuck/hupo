include("input.jl")


function play(opponent, as_top = false)
  state = state_beginning()
  my_turn = as_top

  print_state(state)
  print("\n\n\033[2A")

  while true
    if my_turn
      new_state = copy(state)
      move = get_move(state)
      active_stone = apply_move!(new_state, move)
      clear(11)
      print_state(new_state)
      is_end_state(new_state) && break
      pass = get_pass(state)
      apply_pass!(new_state, active_stone, pass)
      clear(11)
      print_state(new_state)
      state = new_state
    else
      move, pass = sval_sample_action(state, opponent)
      new_state = copy(state)
      active_stone = apply_move!(new_state, move)
      readline()
      clear(13)
      print_state(new_state)
      is_end_state(new_state) && break
      apply_pass!(new_state, active_stone, pass)
      readline()
      clear(13)
      print_state(new_state)
      state = new_state
    end
    my_turn = get_active_stone(state) < 4 ? as_top : !as_top
  end
end


function get_move(state)
  valid_moves = get_valid_moves(state)
  valid_moves == [out] && return out
  move = out
  while true
    key = get_key()
    key == key_up     && (move = up)
    key == key_down   && (move = down)
    key == key_right  && (move = right)
    key == key_left   && (move = left)
    move ∈ valid_moves && return move
  end
end


function get_pass(state)
  while true
    key = get_key()
    key == key_1 && is_valid_pass(1, state) && return 1
    key == key_2 && is_valid_pass(2, state) && return 2
    key == key_3 && is_valid_pass(3, state) && return 3
    key == key_4 && is_valid_pass(4, state) && return 4
    key == key_5 && is_valid_pass(5, state) && return 5
    key == key_6 && is_valid_pass(6, state) && return 6
  end
end
