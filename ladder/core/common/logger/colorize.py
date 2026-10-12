COLORS = {
    'gray': 37,
    'red': 31,
    'green': 32,
    'yellow': 93,
    'blue': 94,
    'magenta': 35,
    'cyan': 36,
    'white': 97,
}


def colorize(string: str, color: str, bold: bool = False, highlight: bool = False) -> str:
    """
    Modify string so it appears in color on the console
    """
    attr = []
    code = COLORS[color]

    if highlight:
        code += 10
    attr.append(str(code))

    if bold:
        attr.append('1')

    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)
