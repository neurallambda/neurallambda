'''

Debugging /  Probing Tools

'''

##################################################
# Misc

da = lambda at_addr: vec_to_address(N.from_mat(at_addr[0]), N.from_mat(addresses))


##################################################
# Colors

# Dictionary of color gradients
colors = {
    "black_to_blue": [
        "\u001b[0;1m",  # bold black
        "\u001b[0m",    # black
        "\u001b[34m",   # blue
        "\u001b[34;1m", # bold blue
    ],
    "black_to_gold": [
        "\u001b[0;1m",  # bold black
        "\u001b[0m",    # black
        "\u001b[33m",   # yellow
        "\u001b[33;1m", # bold yellow
    ],
    "black_to_red": [
        "\u001b[0;1m",  # bold black
        "\u001b[0m",    # black
        "\u001b[31m",   # red
        "\u001b[31;1m", # bold red
    ],
    "black_to_green": [
        "\u001b[0;1m",  # bold black
        "\u001b[0m",    # black
        "\u001b[32m",   # green
        "\u001b[32;1m", # bold green
    ],
    "red_to_green": [
        "\u001b[31;1;4m", # Bold UL Red
        "\u001b[31;1m", # Bold Red
        "\u001b[31m",   # Red
        "\u001b[31;3m", # Italics Red
        "\u001b[33m",   # Yellow
        "\u001b[33m",   # Yellow
        "\u001b[32;3m", # Italics Green
        "\u001b[32m",   # Green
        "\u001b[32;1m", # Bold Green
        "\u001b[32;1;4m", # Bold Green
    ]
}

def interpolate_ansi(value, gradient_key):
    """Interpolate between colors in the specified gradient over intervals."""

    value = max(0, min(value, 1))
    gradient_colors = colors[gradient_key]
    index = round(value * (len(gradient_colors) - 1))
    return gradient_colors[index]

def colorize(text, value, gradient_key='red_to_green'):
    """ Colorize text based on a value between 0 and 1, using a specified color gradient. """
    if value is None:
        return text
    return f"{interpolate_ansi(value, gradient_key)}{text}\u001b[0m"
