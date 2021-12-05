import sys
import os
import dotenv

try:
    dotenv.load_dotenv()
    iglpath = os.environ['PYIGL_PATH'] if 'PYIGL_PATH' in os.environ else ''
    iglpath = os.path.abspath(os.path.normpath(iglpath))
    if os.path.exists(iglpath):
        #print(iglpath)
        sys.path.insert(0, os.path.abspath(iglpath))
    else:
        raise ModuleNotFoundError('pyigl not found')  
    import pyigl as igl
except ImportError as e:
    raise e