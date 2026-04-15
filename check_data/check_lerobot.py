import numpy as np
import torch
import cv2
import os
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, default="/mnt/public/zhoujiakai/0224/real_new_ctrl_pp_dice_10hz/episode_0/data.npz")
    parser.add_argument("--lerobot_root", type=str, default="/mnt/public/zhoujiakai/0224/datasets/real-pnp")
    parser.add_argument("--lerobot_repo", type=str, default="GSEnv-pnp", help="Usually the folder name of the dataset")
    parser.add_argument("--ep_idx", type=int, default=0, help="Episode index in LeRobot to compare")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to compare and save images")
    args = parser.parse_args()

    print(f"Loading NPZ from {args.npz_path}...")
    try:
        npz_file = np.load(args.npz_path, allow_pickle=True)
        npz_data = npz_file['data'].item()
    except Exception as e:
        print(f"Failed to load NPZ: {e}")
        sys.exit(1)

    npz_len = npz_data['observation']['rgb'].shape[0]
    print(f"NPZ data length: {npz_len}")

    print(f"\nLoading LeRobot dataset from {args.lerobot_root}...")
    try:
        dataset = LeRobotDataset(args.lerobot_repo, root=args.lerobot_root)
    except Exception as e:
        print(f"Failed to load LeRobot Dataset: {e}")
        sys.exit(1)
    
    try:
        from_idx = dataset.episode_data_index["from"][args.ep_idx].item()
        to_idx = dataset.episode_data_index["to"][args.ep_idx].item()
        lr_len = to_idx - from_idx
        print(f"LeRobot Episode {args.ep_idx} length: {lr_len} (from {from_idx} to {to_idx})")
    except Exception as e:
        print(f"Error accessing episode {args.ep_idx}: {e}")
        sys.exit(1)

    print("\n--- Length Check ---")
    if npz_len == lr_len:
        print("✅ Pased! Length matches.")
    else:
        print(f"❌ Mismatch! NPZ len: {npz_len}, LeRobot len: {lr_len}")

    print(f"\n--- Comparing Frame {args.frame_idx} ---")
    
    if args.frame_idx >= lr_len or args.frame_idx >= npz_len:
        print(f"Error: frame_idx {args.frame_idx} out of bounds (LeRobot len: {lr_len}, NPZ len: {npz_len})")
        sys.exit(1)
        
    lr_frame = dataset[from_idx + args.frame_idx]
    
    print("[Image Comparison]")
    if "observation.image" in lr_frame:
        lr_img = lr_frame["observation.image"]
        if isinstance(lr_img, torch.Tensor):
            lr_img = lr_img.numpy()
        if lr_img.ndim == 3 and lr_img.shape[0] == 3:
            lr_img = lr_img.transpose(1, 2, 0)
        if lr_img.dtype == np.float32 or lr_img.dtype == np.float64:
            lr_img = (lr_img * 255).astype(np.uint8)
        
        lr_img_bgr = cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"lerobot_img_cam0_frame{args.frame_idx}.png", lr_img_bgr)
        print(f"  Saved LeRobot image to lerobot_img_cam0_frame{args.frame_idx}.png")
        
        npz_img_cam0 = npz_data['observation']['rgb'][args.frame_idx, 0]
        cv2.imwrite(f"npz_img_cam0_frame{args.frame_idx}.png", cv2.cvtColor(npz_img_cam0, cv2.COLOR_RGB2BGR))
        print(f"  Saved NPZ image cam0 to npz_img_cam0_frame{args.frame_idx}.png")
        
        if npz_data['observation']['rgb'].shape[1] > 1:
            npz_img_cam1 = npz_data['observation']['rgb'][args.frame_idx, 1]
            cv2.imwrite(f"npz_img_cam1_frame{args.frame_idx}.png", cv2.cvtColor(npz_img_cam1, cv2.COLOR_RGB2BGR))
            print(f"  Saved NPZ image cam1 to npz_img_cam1_frame{args.frame_idx}.png")
        
        match_shape = lr_img.shape == npz_img_cam0.shape
        diff = np.mean(np.abs(lr_img.astype(float) - npz_img_cam0.astype(float)))
        print(f"  LeRobot img shape: {lr_img.shape}, dtype: {lr_img.dtype}")
        print(f"  NPZ img shape (cam 0): {npz_img_cam0.shape}, dtype: {npz_img_cam0.dtype}")
        print(f"  Shape match: {'✅' if match_shape else '❌'}, Mean pixel diff: {diff:.2f}")
    else:
        print("  No 'observation.image' found in LeRobot dataset.")

    print("\n[State Comparison]")
    if "observation.state" in lr_frame:
        lr_state = lr_frame["observation.state"]
        if isinstance(lr_state, torch.Tensor):
            lr_state = lr_state.numpy()
        print(f"  LeRobot State (shape {lr_state.shape}):\n    {lr_state}")
        
        npz_joint = npz_data['state']['joint']['position'][args.frame_idx]
        npz_grip = npz_data['state']['end_effector']['gripper_width'][args.frame_idx]
        
        expected_state = np.concatenate([npz_joint, [npz_grip / 2.0, npz_grip / 2.0]]).astype(np.float32)
        print(f"  NPZ Joint Position: {npz_joint}")
        print(f"  NPZ Gripper Width:  {npz_grip}")
        print(f"  Expected State:   \n    {expected_state}")
        
        state_match = np.allclose(lr_state, expected_state, atol=1e-5)
        print(f"  State match: {'✅' if state_match else '❌'}")
    else:
        print("  No 'observation.state' found in LeRobot dataset.")

    print("\n[Action Comparison]")
    if "actions" in lr_frame:
        lr_action = lr_frame["actions"]
        if isinstance(lr_action, torch.Tensor):
            lr_action = lr_action.numpy()
        print(f"  LeRobot Action (shape {lr_action.shape}):\n    {lr_action}")
        
        npz_a_ee_pos = npz_data['action']['end_effector']['delta_position'][args.frame_idx]
        npz_a_ee_ori = npz_data['action']['end_effector']['delta_orientation'][args.frame_idx]
        npz_a_ee_eul = npz_data['action']['end_effector']['delta_euler'][args.frame_idx]
        npz_a_grip = npz_data['action']['end_effector']['gripper_control'][args.frame_idx]
        npz_a_joint = npz_data['action']['joint']['position'][args.frame_idx]
        
        print(f"  NPZ Action Delta Position:\n    {npz_a_ee_pos}")
        print(f"  NPZ Action Delta Euler:\n    {npz_a_ee_eul}")
        print(f"  NPZ Action Gripper Control:\n    {npz_a_grip}")
        print(f"  NPZ Action Joint Position:\n    {npz_a_joint}")
    else:
        print("  No 'actions' found in LeRobot dataset.")

if __name__ == "__main__":
    main()
