"""
conda activate glasses
cd /home/g1/unitree_groot1.5
python unitree_lerobot/eval_robot/run_gaze_server.py --port 5556
"""

import os
import sys
import time
import zmq
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import argparse
import tempfile
import shutil

# Overlay colors for multi-object tracking (obj1=red, obj2=green, then cycle)
OBJECT_COLORS = [
    np.array([255, 0, 0], dtype=np.uint8),      # red
    np.array([0, 255, 0], dtype=np.uint8),      # green
    np.array([0, 128, 255], dtype=np.uint8),    # blue
    np.array([255, 196, 64], dtype=np.uint8),   # yellow
]

# --- PATH SETUP FOR GLASSES ENV ---
# Update this path if your SAM2 installation is elsewhere
SAM2_PATH = "/home/g1/zuo/glasses/sam2"
sys.path.append(SAM2_PATH) 

try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("[ERROR] Could not import SAM2. Make sure you activated 'conda activate glasses'!")
    print(f"[ERROR] Also verify SAM2_PATH is correct: {SAM2_PATH}")
    sys.exit(1)

class GazeServer:
    def __init__(self, port=5556, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
        # Update these paths if your SAM2 checkpoint/config are elsewhere
        self.checkpoint = "/home/g1/zuo/glasses/sam2/models/sam2.1_hiera_small.pt"
        self.config = "configs/sam2.1/sam2.1_hiera_s.yaml"  # Relative to SAM2_PATH
        
        print(f"Loading SAM2 Video Predictor (48GB VRAM optimized) from {self.checkpoint}...")
        self.video_predictor = build_sam2_video_predictor(self.config, self.checkpoint, device=self.device)
        self.image_predictor = SAM2ImagePredictor(self.video_predictor)
        
        self.tracker_initialized = False
        self.inference_state = None
        self.temp_dir = None
        self.frame_idx = 0
        self.last_masks_by_obj = {}
        self.obj_ids = []
        
        # Normalization constants (matching SAM2 defaults)
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        self.img_size = self.video_predictor.image_size

        print(f" Gaze Server Ready on port {port}!")

    def cleanup_state(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.tracker_initialized = False
        self.inference_state = None
        self.frame_idx = 0
        self.last_masks_by_obj = {}
        self.obj_ids = []

    def run(self):
        print("Waiting for connection from Robot Client...")
        while True:
            try:
                msg = self.socket.recv_pyobj()
            except Exception as e:
                print(f"Recv error: {e}")
                continue
                
            command = msg.get('cmd')
            image_rgb = msg.get('image') # RGB numpy array
            
            response = {"status": "ok"}
            
            if command == 'init':
                print("Received INIT request. Launching GUI...")
                self.cleanup_state()
                self.temp_dir = tempfile.mkdtemp()
                success = self.interactive_init(image_rgb)
                response["success"] = success
                self.socket.send_pyobj(response)
                
            elif command == 'track':
                if not self.tracker_initialized:
                    response["mask"] = None
                    response["masks_by_obj"] = None
                    response["overlayed_image"] = image_rgb
                else:
                    masks_by_obj = self.step(image_rgb)
                    overlayed_image = self.apply_mask_overlay(image_rgb, masks_by_obj, alpha=0.35)
                    combined_mask = self._combine_masks(masks_by_obj, image_rgb.shape[:2])
                    response = {
                        "status": "ok",
                        "mask": combined_mask,
                        "masks_by_obj": masks_by_obj,
                        "overlayed_image": overlayed_image,
                    }
                self.socket.send_pyobj(response)
                
            elif command == 'ping':
                 self.socket.send_pyobj({"status": "pong"})
            elif command == 'reset':
                 self.cleanup_state()
                 self.socket.send_pyobj({"status": "ok"})
            else:
                 self.socket.send_pyobj({"status": "error", "msg": "Unknown command"})

    def _get_obj_color(self, obj_id):
        return OBJECT_COLORS[(int(obj_id) - 1) % len(OBJECT_COLORS)]

    def _prompt_single_object(self, first_frame_rgb, object_index):
        point_coords = []
        point_labels = []  # 1 for FG, 0 for BG
        selected_mask = None

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(first_frame_rgb)
        ax.set_title(
            f"Object {object_index}: L=FG | R=BG | SPACE=Preview | C=Clear | Enter=Confirm | Esc=Finish"
        )
        mask_overlay_layer = None

        def show_mask(mask):
            nonlocal mask_overlay_layer
            if mask_overlay_layer is not None:
                mask_overlay_layer.remove()
            color = self._get_obj_color(object_index).astype(np.float32) / 255.0
            rgba = np.array([color[0], color[1], color[2], 0.5], dtype=np.float32)
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * rgba.reshape(1, 1, 4)
            mask_overlay_layer = ax.imshow(mask_image)
            fig.canvas.draw()

        def onclick(event):
            if event.xdata is None or event.ydata is None:
                return
            if fig.canvas.toolbar and fig.canvas.toolbar.mode != "":
                return
            x, y = int(event.xdata), int(event.ydata)
            label = 1 if event.button == 1 else 0
            point_coords.append([x, y])
            point_labels.append(label)
            color = "green" if label == 1 else "blue"
            marker = "*" if label == 1 else "x"
            ax.scatter(x, y, c=color, marker=marker, s=150)
            fig.canvas.draw()

        def onkey(event):
            nonlocal point_coords, point_labels, selected_mask
            if event.key == " ":
                if not point_coords:
                    return
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    masks, scores, _ = self.image_predictor.predict(
                        point_coords=np.array(point_coords),
                        point_labels=np.array(point_labels),
                        multimask_output=True,
                    )
                best_idx = int(np.argmax(scores))
                selected_mask = masks[best_idx]
                show_mask(selected_mask)
            elif event.key in ["c", "C"]:
                point_coords.clear()
                point_labels.clear()
                selected_mask = None
                ax.clear()
                ax.imshow(first_frame_rgb)
                ax.set_title(
                    f"Object {object_index}: L=FG | R=BG | SPACE=Preview | C=Clear | Enter=Confirm | Esc=Finish"
                )
                fig.canvas.draw()
            elif event.key == "enter":
                plt.close(fig)
            elif event.key in ["escape", "q"]:
                point_coords.clear()
                point_labels.clear()
                selected_mask = None
                plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", onclick)
        fig.canvas.mpl_connect("key_press_event", onkey)
        plt.show(block=True)

        if selected_mask is None:
            return None
        return selected_mask

    def _ask_add_next_object_gui(self, image_rgb, objects):
        preview = image_rgb.copy()
        for obj in objects:
            mask = obj["mask"]
            color = self._get_obj_color(obj["obj_id"]).astype(np.float32)
            preview[mask] = (preview[mask].astype(np.float32) * 0.65 + color * 0.35).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(preview)
        ax.set_title(
            f"{len(objects)} object(s) selected\nEnter/Y: Add next object | N/Esc: Start tracking",
            fontsize=13,
        )
        ax.axis("off")
        plt.tight_layout()

        action = {"add_next": False}

        def on_key(event):
            if event.key in ["enter", "y", "Y"]:
                action["add_next"] = True
                plt.close(fig)
            elif event.key in ["n", "N", "escape", "q"]:
                action["add_next"] = False
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()
        return action["add_next"]

    def interactive_init(self, first_frame_rgb):
        # ImagePredictor is great for real-time clicks preview
        self.image_predictor.set_image(first_frame_rgb)
        self.obj_ids = []
        
        # Save first frame for VideoPredictor initialization requirement
        cv2.imwrite(os.path.join(self.temp_dir, "00000.jpg"), cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2BGR))

        print("Initializing Video Tracking State...")
        self.inference_state = self.video_predictor.init_state(video_path=self.temp_dir)
        
        # Convert to list for dynamic growth (streaming hack)
        if torch.is_tensor(self.inference_state["images"]):
            self.inference_state["images"] = list(self.inference_state["images"])

        objects = []
        obj_id = 1
        while True:
            try:
                mask = self._prompt_single_object(first_frame_rgb, obj_id)
            except Exception as e:
                print(f"GUI Error: {e}")
                return False

            if mask is None:
                if len(objects) == 0:
                    print("No object selected.")
                    return False
                print("Finish object collection.")
                break

            mask_bool = mask.astype(bool)
            if not mask_bool.any():
                print(f"Object {obj_id} mask is empty, retrying this object.")
                continue

            print(f"Adding object {obj_id}...")
            self.video_predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=obj_id,
                mask=mask_bool,
            )
            objects.append({"obj_id": obj_id, "mask": mask_bool})
            self.obj_ids.append(obj_id)

            add_next = self._ask_add_next_object_gui(first_frame_rgb, objects)
            if not add_next:
                break
            obj_id += 1
        
        self.frame_idx = 0
        self.tracker_initialized = True
        print(f" Tracker Initialized successfully. objects={len(self.obj_ids)}")
        return True

    def apply_mask_overlay(self, frame_rgb, mask_data, alpha=0.35):
        """
        Apply colored overlay to masked regions.
        Supports:
            - single mask ndarray (backward compatibility -> red)
            - dict[obj_id] = mask ndarray (multi-object)
        
        Args:
            frame_rgb: Original RGB image (H, W, 3) uint8
            mask_data: mask or dict of masks
            alpha: Overlay intensity (0.3 means 30% color)
        Returns:
            Overlayed RGB image (H, W, 3) uint8
        """
        if mask_data is None:
            return frame_rgb
        
        output = frame_rgb.copy()
        h, w = frame_rgb.shape[:2]

        if isinstance(mask_data, dict):
            if len(mask_data) == 0:
                return frame_rgb
            for obj_id in sorted(mask_data.keys()):
                mask = mask_data[obj_id]
                if mask is None:
                    continue
                if mask.shape != (h, w):
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                else:
                    mask_resized = mask.astype(bool)
                if not mask_resized.any():
                    continue
                color = np.zeros_like(output)
                rgb = self._get_obj_color(obj_id)
                color[:, :, 0] = rgb[0]
                color[:, :, 1] = rgb[1]
                color[:, :, 2] = rgb[2]
                output[mask_resized] = (
                    output[mask_resized] * (1 - alpha) + color[mask_resized] * alpha
                ).astype(np.uint8)
            return output
        else:
            mask = np.asarray(mask_data)
            if mask.shape != (h, w):
                mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask_resized = mask.astype(bool)
            if not mask_resized.any():
                return frame_rgb
            red_mask = np.zeros_like(output)
            red_mask[:, :, 0] = 255
            red_mask[:, :, 1] = 0
            red_mask[:, :, 2] = 0
            output[mask_resized] = (
                output[mask_resized] * (1 - alpha) + red_mask[mask_resized] * alpha
            ).astype(np.uint8)
            return output

    def _combine_masks(self, masks_by_obj, target_hw):
        if masks_by_obj is None:
            return None
        if isinstance(masks_by_obj, dict):
            h, w = target_hw
            combined = np.zeros((h, w), dtype=bool)
            for obj_id in masks_by_obj:
                mask = masks_by_obj[obj_id]
                if mask is None:
                    continue
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    mask = mask.astype(bool)
                combined |= mask
            return combined
        mask = np.asarray(masks_by_obj)
        return mask.astype(bool) if mask.size > 0 else None

    def step(self, frame_rgb):
        self.frame_idx += 1
        
        # 1. Preprocess & Manual Injection
        img_resized = cv2.resize(frame_rgb, (self.img_size, self.img_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor.to(self.device) - self.img_mean) / self.img_std
        
        # 1.5 Frame Skipping (Target ~10Hz assuming ~30FPS input)
        # We only run SAM2 every 3rd frame. For other frames, return last masks.
        if self.last_masks_by_obj and (self.frame_idx % 3 != 0):
            if len(self.inference_state["images"]) <= self.frame_idx:
                self.inference_state["images"].append(img_tensor)
            else:
                self.inference_state["images"][self.frame_idx] = img_tensor
            self.inference_state["num_frames"] = len(self.inference_state["images"])
            return self.last_masks_by_obj

        if len(self.inference_state["images"]) <= self.frame_idx:
            self.inference_state["images"].append(img_tensor)
        else:
            self.inference_state["images"][self.frame_idx] = img_tensor
        self.inference_state["num_frames"] = len(self.inference_state["images"])
        
        # 2. Memory Pruning (Prevent indefinite growth)
        if self.frame_idx > 0 and self.frame_idx % 100 == 0:
            self._prune_memory(keep_last_n=30)

        # 3. Tracking Step with bfloat16
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                propagation_gen = self.video_predictor.propagate_in_video(
                    self.inference_state,
                    start_frame_idx=self.frame_idx,
                    max_frame_num_to_track=1
                )
                _, out_obj_ids, out_mask_logits = next(propagation_gen)

                masks_by_obj = {}
                for i, obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    if mask is not None and np.asarray(mask).any():
                        masks_by_obj[int(obj_id)] = mask.astype(bool)

                if len(masks_by_obj) == 0:
                    self.last_masks_by_obj = {}
                    return None

                self.last_masks_by_obj = masks_by_obj
                return masks_by_obj
        except Exception as e:
            print(f"Tracking error: {e}")
            return None

    def _prune_memory(self, keep_last_n=30):
        """
        Surgically remove old frames from inference_state to save memory.
        We MUST keep frame 0 (the initial prompt) and the most recent N frames.
        """
        if self.inference_state is None: return
        
        # Determine which frames to keep
        # Always keep frame 0 (initial conditioning)
        frames_to_keep = {0}
        # Keep a window of recent frames
        start_keep = max(1, self.frame_idx - keep_last_n)
        for f in range(start_keep, self.frame_idx + 1):
            frames_to_keep.add(f)
            
        # 1. Prune images (Replace old images with None to reclaim VRAM/RAM)
        for f in range(1, start_keep):
            if f < len(self.inference_state["images"]):
                self.inference_state["images"][f] = None
        
        # 2. Prune internal prediction results (The biggest VRAM consumers)
        for obj_idx in self.inference_state["output_dict_per_obj"]:
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                d = self.inference_state["output_dict_per_obj"][obj_idx][storage_key]
                for f in list(d.keys()):
                    if f not in frames_to_keep:
                        del d[f]
            
            # Prune tracking metadata
            tracked_frames = self.inference_state["frames_tracked_per_obj"][obj_idx]
            for f in list(tracked_frames.keys()):
                if f not in frames_to_keep:
                    del tracked_frames[f]

        # 3. Clear transient interaction data
        for obj_idx in self.inference_state["point_inputs_per_obj"]:
            d = self.inference_state["point_inputs_per_obj"][obj_idx]
            for f in list(d.keys()):
                if f not in frames_to_keep:
                    del d[f]
        
        # Note: cached_features is automatically handled by SAM2 (always size 1)
        # print(f"Memory Pruned at frame {self.frame_idx}. Kept {len(frames_to_keep)} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5556)
    args = parser.parse_args()
    
    server = GazeServer(port=args.port)
    try:
        server.run()
    finally:
        server.cleanup_state()
