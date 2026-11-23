from colorama import Style, Back, Fore


"""************** terminal style Interface*********************"""
def printProgressBar (iteration, total, prefix = '', suffix = '',
                      decimals = 1, length = 100, fill = '>',
                      printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals
                                    in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = Fore.BLUE + fill * filledLength + Fore.YELLOW + '-' * (length - filledLength)
    print(f'\r{ Back.LIGHTBLACK_EX + Fore.BLUE + prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print(Fore.GREEN + "Complete " + Style.RESET_ALL)
Input_style = Back.LIGHTBLACK_EX + Fore.WHITE
end_style = Style.RESET_ALL
"""***********************************************"""
