import sys
import os
import dotenv

try:
    KEYS = ('PYIGL_PATH', 'SDF_PATH')
    
    dotenv.load_dotenv(verbose=True)
    for key in KEYS:
        _path = os.environ.get(key, '')
        _path = os.path.abspath(os.path.normpath(_path))
        if os.path.exists(_path):
            sys.path.insert(0, os.path.abspath(_path))

    try:
        import igl
    except:
        print("[pyigl_import] module igl not found. trying to import pyigl")
        import pyigl as igl

except ImportError as e:
    raise e