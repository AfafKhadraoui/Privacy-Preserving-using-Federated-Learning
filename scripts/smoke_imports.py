modules = ['torch','torchvision','facenet_pytorch','flwr','cv2','fastapi','uvicorn']
for m in modules:
    try:
        mod = __import__(m)
        v = getattr(mod, '__version__', None)
        print(m, 'OK', v)
    except Exception as e:
        print(m, 'ERROR', type(e).__name__, e)

print('SMOKE TEST COMPLETE')
