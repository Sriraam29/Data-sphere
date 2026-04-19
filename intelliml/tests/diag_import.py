import sys, os, importlib
print('cwd:', os.getcwd())
print('listdir root:', os.listdir())
print('intelliml exists:', os.path.isdir('intelliml'))
print('sys.path[0]:', sys.path[0])
print('sys.path[:5]:', sys.path[:5])
try:
    m = importlib.import_module('intelliml')
    print('import ok:', m)
except Exception as e:
    print('import error:', type(e), e)
    import traceback; traceback.print_exc()
