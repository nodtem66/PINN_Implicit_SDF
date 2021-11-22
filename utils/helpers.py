# -*- coding: utf-8 -*-

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        module = get_ipython().__class__.__module__

        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif module == "google.colab._shell":
            return True
        return False
    except NameError:
        return False      # Probably standard Python interpreter