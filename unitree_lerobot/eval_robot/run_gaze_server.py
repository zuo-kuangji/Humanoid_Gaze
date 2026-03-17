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
                return_pick_mask = bool(msg.get("return_pick_mask", False))
                if not self.tracker_initialized:
                    response["mask"] = None
                    response["masks_by_obj"] = None
                    response["overlayed_image"] = image_rgb
                    if return_pick_mask:
                        h, w = image_rgb.shape[:2]
                        response["pick_mask"] = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    masks_by_obj = self.step(image_rgb)
                    overlayed_image = self.apply_mask_overlay(image_rgb, masks_by_obj)
                    combined_mask = self._combine_masks(masks_by_obj, image_rgb.shape[:2])
                    response = {
                        "status": "ok",
                        "mask": combined_mask,
                        "masks_by_obj": masks_by_obj,
                        "overlayed_image": overlayed_image,
                    }
                    if return_pick_mask:
                        response["pick_mask"] = self._mask_to_pick_rgb(combined_mask, image_rgb.shape[:2])
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
        img_layer = [ax.imshow(first_frame_rgb)]
        ax.set_title(
            f"Object {object_index}: L=FG | R=BG | SPACE=Preview | C=Clear | Enter=Confirm | Esc=Finish"
        )

        def show_mask(mask):
            # Generate the true morphological outline preview
            overlay_rgb = self.apply_mask_overlay(first_frame_rgb, {object_index: mask})
            img_layer[0].set_data(overlay_rgb)
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
                img_layer[0] = ax.imshow(first_frame_rgb)
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
        mask_dict = {obj["obj_id"]: obj["mask"] for obj in objects}
        preview = self.apply_mask_overlay(image_rgb, mask_dict)

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

    def apply_mask_overlay(self, frame_rgb, mask_data, darken_ratio=1.0, k1_size=5, k2_size=7):
        """
        Apply mathematical morphology visual prompting consistent with
        pipeline_mask2overlay.py in its default outline_bw mode
        (Mask 0: outer white ring + inner black edge, Mask 1: Red/Green).
        
        Args:
            frame_rgb: Original RGB image (H, W, 3) uint8 as NumPy array
            mask_data: None, single mask ndarray, or dict[obj_id] = mask ndarray
            darken_ratio: Kept for API compatibility; outline_bw does not darken mask 0
            k1_size: Outer white margin / inner black edge kernel size
            k2_size: Outer place-mask green margin kernel size
        Returns:
            Overlayed RGB image (H, W, 3) uint8
        """
        if mask_data is None:
            return frame_rgb

        h, w = frame_rgb.shape[:2]
        
        # Extract masks into a list (order is important: obj_id 1 -> mask 0, obj_id 2 -> mask 1)
        masks_list = []
        if isinstance(mask_data, dict):
            if len(mask_data) == 0:
                return frame_rgb
            max_id = max(mask_data.keys())
            for i in range(1, max_id + 1):
                if i in mask_data and mask_data[i] is not None:
                    mask = mask_data[i]
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        mask = mask.astype(bool)
                    masks_list.append(mask)
                else:
                    masks_list.append(np.zeros((h, w), dtype=bool))
        else:
            mask = np.asarray(mask_data)
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask = mask.astype(bool)
            masks_list.append(mask)

        if len(masks_list) == 0:
            return frame_rgb

        # Stack masks to (1, num_masks, H, W)
        masks_np = np.stack(masks_list, axis=0)[np.newaxis, ...] # (1, M, H, W)

        # --- Fast GPU Morphology Equivalent of pipeline_mask2overlay ---
        # 1. H2D (Numpy to Tensor)
        image_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)
        masks_t = torch.from_numpy(masks_np).to(self.device, dtype=torch.float32)

        # 2. Dilation Operations
        import torch.nn.functional as F
        mask_k1 = F.max_pool2d(masks_t, kernel_size=k1_size, stride=1, padding=k1_size//2)
        mask_k2 = F.max_pool2d(masks_t, kernel_size=k2_size, stride=1, padding=k2_size//2)
        eroded_k1 = 1.0 - F.max_pool2d(1.0 - masks_t, kernel_size=k1_size, stride=1, padding=k1_size//2)

        # 3. Boolean Masks for rendering
        S_inner = masks_t > 0.5
        S_k1_border = (mask_k1 > 0.5) & (~S_inner)
        S_k2_border = (mask_k2 > 0.5) & (~(mask_k1 > 0.5))

        # 4. Pixel-wise Override
        I_prompt = image_t.clone()
        num_masks = masks_np.shape[1]
        
        S0_total = None
        # --- Mask 0 (Pick) -> pipeline_mask2overlay.py outline_bw ---
        if num_masks > 0:
            S0_inner = S_inner[:, 0:1, :, :]
            S0_white = S_k1_border[:, 0:1, :, :]
            # Inner black contour computed from erosion, matching outline_bw.
            S0_black = S0_inner & (~(eroded_k1[:, 0:1, :, :] > 0.5))

            # Match pipeline exclusion footprint: mask interior plus outer white ring.
            S0_total = S0_inner | S0_white

            I_prompt = torch.where(S0_white.expand_as(I_prompt), torch.tensor([255.0, 255.0, 255.0], device=self.device).view(1,3,1,1), I_prompt)
            I_prompt = torch.where(S0_black.expand_as(I_prompt), torch.tensor([0.0, 0.0, 0.0], device=self.device).view(1,3,1,1), I_prompt)
            
        # --- Mask 1 (Place) -> Red / Green ---
        if num_masks > 1:
            S1_inner = S_inner[:, 1:2, :, :]
            S1_red = S_k1_border[:, 1:2, :, :]
            S1_green = S_k2_border[:, 1:2, :, :]
            
            # Apply Boolean Exclusion: Mask 1's borders cannot intersect Mask 0's footprint
            if S0_total is not None:
                S1_red = S1_red & (~S0_total)
                S1_green = S1_green & (~S0_total)
                
            I_prompt = torch.where(S1_red.expand_as(I_prompt), torch.tensor([255.0, 0.0, 0.0], device=self.device).view(1,3,1,1), I_prompt)
            I_prompt = torch.where(S1_green.expand_as(I_prompt), torch.tensor([0.0, 255.0, 0.0], device=self.device).view(1,3,1,1), I_prompt)

        # 5. D2H (Tensor to Numpy)
        I_prompt_uint8 = torch.clamp(I_prompt, 0, 255).to(torch.uint8)
        output_np = I_prompt_uint8.squeeze(0).permute(1, 2, 0).cpu().numpy()

        return output_np

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

    def _mask_to_pick_rgb(self, mask, target_hw):
        h, w = target_hw
        if mask is None:
            return np.zeros((h, w, 3), dtype=np.uint8)

        mask_np = np.asarray(mask)
        if mask_np.ndim == 3:
            if mask_np.shape[-1] >= 1:
                mask_np = mask_np[..., 0]
            elif mask_np.shape[0] >= 1:
                mask_np = mask_np[0]
            else:
                mask_np = np.squeeze(mask_np)
        if mask_np.shape != (h, w):
            mask_np = cv2.resize(mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        pick_u8 = (mask_np.astype(bool).astype(np.uint8)) * 255
        return np.repeat(pick_u8[:, :, None], 3, axis=2)

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
