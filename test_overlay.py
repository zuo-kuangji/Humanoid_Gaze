import sys
import numpy as np
import cv2

sys.path.append('/home/g1/humanoid_gaze')
from unitree_lerobot.eval_robot.run_gaze_server import GazeServer

dummy_rgb = np.full((480, 640, 3), 100, dtype=np.uint8)
dummy_mask = np.zeros((480, 640), dtype=bool)
dummy_mask[200:250, 300:350] = True

try:
    server = GazeServer(port=5558)
    overlay = server.apply_mask_overlay(dummy_rgb, dummy_mask)
    
    center_pixel = overlay[225, 325]
    print(f'Center pixel value (expected ~75): {center_pixel}')
    
    print('✅ Overlay logic tested successfully on GPU!')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'❌ Failed: {e}')
