@enum Key key_up key_down key_left key_right key_enter key_backspace key_1 key_2 key_3 key_4 key_5 key_6


function get_key()
  local key
  if is_linux()
    run(`stty raw -echo`)
    while true
      c = ccall("getchar", Cint, ())
      if c == 3 # Ctrl+C
        run(`stty -raw echo`)
        throw(InterruptException())
      elseif c == 13
        key = key_enter
        break
      elseif c == 127
        key = key_backspace
        break
      elseif c == 49
        key = key_1
        break
      elseif c == 50
        key = key_2
        break
      elseif c == 51
        key = key_3
        break
      elseif c == 52
        key = key_4
        break
      elseif c == 53
        key = key_5
        break
      elseif c == 54
        key = key_6
        break
      elseif c == 27
        c = ccall("getchar", Cint, ())
        c == 91 || continue
        c = ccall("getchar", Cint, ())
        if c == 65
          key = key_up
          break
        elseif c == 66
          key = key_down
          break
        elseif c == 67
          key = key_right
          break
        elseif c == 68
          key = key_left
          break
        end
      end
    end
    run(`stty -raw echo`)
    return key
  elseif is_windows()
    println("Windows is not supported yet")
  else
    println("unsupported OS")
  end
end
