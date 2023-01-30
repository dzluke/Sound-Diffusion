def clear_dir(p):
    """
    Delete the contents of the directory at p
    """
    if not p.is_dir():
        return
    for f in p.iterdir():
        if f.is_file():
            f.unlink()
        else:
            clear_dir(f)


def format_time(seconds):
    """

    :param seconds:
    :return: a dictionary with the keys 'h', 'm', 's', that is the amount of hours, minutes, seconds equal to 'seconds'
    """
    hms = [seconds // 3600, (seconds // 60) % 60, seconds % 60]
    hms = [int(t) for t in hms]
    labels = ['h', 'm', 's']
    return {labels[i]: hms[i] for i in range(len(hms))}


def time_string(seconds):
    """
    Returns a string with the format "0h 0m 0s" that represents the amount of time provided
    :param seconds:
    :return: string
    """
    t = format_time(seconds)
    return "{}h {}m {}s".format(t['h'], t['m'], t['s'])
