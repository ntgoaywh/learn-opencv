import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 暗通道先验 (DCP) ---
# (使用我们之前定义的 dehaze_dcp_main 函数及其辅助函数)
# <editor-fold desc="DCP Helper Functions (from previous answer)">
def get_dark_channel_dcp(image, patch_size): # Renamed to avoid conflict
    min_channel_img = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel_img, kernel)
    return dark_channel

def estimate_atmospheric_light_dcp(image_hazy, dark_channel, percentile=0.001): # Renamed
    rows, cols, _ = image_hazy.shape
    num_pixels = rows * cols
    num_brightest = int(max(1, num_pixels * percentile))
    dark_channel_flat = dark_channel.flatten()
    brightest_indices = np.argsort(dark_channel_flat)[-num_brightest:]
    image_hazy_flat = image_hazy.reshape(num_pixels, 3)
    A_candidates = image_hazy_flat[brightest_indices]
    A = np.mean(A_candidates, axis=0)
    return A.reshape(1, 1, 3)

def estimate_transmission_dcp(image_hazy, A, patch_size, omega=0.95): # Renamed
    A_safe = np.maximum(A, 1e-6)
    image_norm_A = image_hazy / A_safe
    dark_channel_norm_A = get_dark_channel_dcp(image_norm_A, patch_size)
    t_tilde = 1.0 - omega * dark_channel_norm_A
    return t_tilde

def refine_transmission_guided_filter_dcp(image_hazy, t_tilde, radius, eps): # Renamed
    gray_image = cv2.cvtColor(image_hazy.astype(np.float32), cv2.COLOR_BGR2GRAY)
    t_tilde_float32 = t_tilde.astype(np.float32)
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
        t_refined = cv2.ximgproc.guidedFilter(guide=gray_image, src=t_tilde_float32,
                                              radius=radius, eps=eps*eps)
    else:
        print("Warning (DCP): cv2.ximgproc.guidedFilter not found. Using Gaussian blur.")
        ksize = 2 * radius + 1
        t_refined = cv2.GaussianBlur(t_tilde_float32, (ksize, ksize), 0)
    return t_refined

def recover_dehazed_image_dcp(image_hazy, A, t_refined, t0=0.1): # Renamed
    t_bounded = np.maximum(t_refined, t0)
    t_bounded_3channel = np.expand_dims(t_bounded, axis=2)
    J = (image_hazy - A) / t_bounded_3channel + A
    J = np.clip(J, 0, 1)
    return J

def dehaze_dcp(image_hazy_bgr, patch_size=15, omega=0.95, t0=0.1,
               guided_filter_radius=60, guided_filter_eps=0.001,
               atmospheric_light_percentile=0.001):
    if image_hazy_bgr.dtype == np.uint8:
        image_hazy_norm = image_hazy_bgr.astype(np.float64) / 255.0
    elif image_hazy_bgr.max() > 1.0 :
        image_hazy_norm = image_hazy_bgr.astype(np.float64) / 255.0
    else:
        image_hazy_norm = image_hazy_bgr.astype(np.float64)

    if image_hazy_norm.ndim == 2:
        image_hazy_norm = cv2.cvtColor(image_hazy_norm, cv2.COLOR_GRAY2BGR)
    elif image_hazy_norm.shape[2] == 4:
        image_hazy_norm = cv2.cvtColor(image_hazy_norm, cv2.COLOR_RGBA2BGR)

    dark_channel = get_dark_channel_dcp(image_hazy_norm, patch_size)
    A = estimate_atmospheric_light_dcp(image_hazy_norm, dark_channel, percentile=atmospheric_light_percentile)
    t_tilde = estimate_transmission_dcp(image_hazy_norm, A, patch_size, omega)
    t_refined = refine_transmission_guided_filter_dcp(image_hazy_norm, t_tilde,
                                                  radius=guided_filter_radius,
                                                  eps=guided_filter_eps)
    J_dehazed = recover_dehazed_image_dcp(image_hazy_norm, A, t_refined, t0)
    return (J_dehazed * 255).astype(np.uint8) # Return uint8 image
# </editor-fold>


# --- 2. 对比度受限的自适应直方图均衡化 (CLAHE) ---
def dehaze_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    if image_bgr.dtype != np.uint8:
        image_bgr = (image_bgr * 255).astype(np.uint8)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_img

# --- 3. 颜色衰减先验 (CAP) ---
# 这是一个简化的CAP实现，实际论文中的细节更复杂，特别是大气光估计和透射率优化部分
# 这里的大气光A的估计和透射率优化可以复用DCP的类似方法或更简单的方法
def get_depth_map_cap(image_bgr_norm, patch_size=15):
    # image_bgr_norm is float [0,1]
    # d(x) = V(x) - S(x)
    img_hsv = cv2.cvtColor(image_bgr_norm.astype(np.float32), cv2.COLOR_BGR2HSV)
    v = img_hsv[:, :, 2]
    s = img_hsv[:, :, 1]
    depth_map = v - s # This is the raw depth map, higher values mean more haze
    # Refine depth map, e.g., using a min filter or guided filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    # In CAP, areas with more haze have larger d(x).
    # To make it similar to dark channel (smaller values for hazer regions before 1-()),
    # we can take the negative or use it as is.
    # Original CAP uses this depth map to estimate transmission.
    # For simplicity here, let's treat it similar to how dark channel leads to transmission.
    # Brighter areas in (v-s) are hazier.
    refined_depth_map = cv2.erode(depth_map, kernel) # Simplistic refinement
    return refined_depth_map


def dehaze_cap(image_bgr, patch_size=15, omega=0.85, t0=0.1,
               guided_filter_radius=60, guided_filter_eps=0.01,
               atmospheric_light_percentile=0.001):
    if image_bgr.dtype == np.uint8:
        image_bgr_norm = image_bgr.astype(np.float64) / 255.0
    elif image_bgr.max() > 1.0 :
        image_bgr_norm = image_bgr.astype(np.float64) / 255.0
    else:
        image_bgr_norm = image_bgr.astype(np.float64)

    # 1. Estimate depth map based on V-S
    # For CAP, higher d(x) means more haze.
    # The transmission t(x) = exp(-beta * d(x)). Beta is a scene depth parameter.
    # Or, similar to DCP, t(x) = 1 - omega * normalized_depth_map.
    # Let's try a simpler approach adapting DCP's framework.
    raw_scene_depth = get_depth_map_cap(image_bgr_norm, patch_size) # Higher means hazier

    # 2. Estimate Atmospheric Light (can use DCP's method with raw_scene_depth)
    # Pixels with highest raw_scene_depth values are candidates for A
    rows, cols, _ = image_bgr_norm.shape
    num_pixels = rows * cols
    num_brightest = int(max(1, num_pixels * atmospheric_light_percentile))
    depth_flat = raw_scene_depth.flatten()
    # Sort for highest depth values (haziest according to V-S difference)
    brightest_indices = np.argsort(depth_flat)[-num_brightest:]
    image_hazy_flat = image_bgr_norm.reshape(num_pixels, 3)
    A_candidates = image_hazy_flat[brightest_indices]
    A = np.mean(A_candidates, axis=0).reshape(1,1,3)
    # print(f"CAP Atm. Light (BGR): {A.flatten()}")

    # 3. Estimate Transmission
    # Normalize raw_scene_depth to [0,1] to act like a "haze density map"
    # where 1 is max haze and 0 is no haze.
    # raw_scene_depth can be negative if S > V. Clip at 0.
    haze_density_map = np.clip(raw_scene_depth, 0, None)
    if np.max(haze_density_map) > 1e-6 : # Avoid division by zero if map is all zeros
        haze_density_map = haze_density_map / np.max(haze_density_map)
    else:
        haze_density_map = np.zeros_like(haze_density_map)


    # t_tilde = 1 - omega * haze_density_map
    # Another common model for CAP is t(x) = exp(-beta * d(x)).
    # For a simpler analogy to DCP's t = 1 - omega * dark_channel_of_normalized_image:
    # We need to normalize image by A and then compute its "CAP channel"
    A_safe = np.maximum(A, 1e-6)
    image_norm_A = image_bgr_norm / A_safe
    cap_channel_norm_A = get_depth_map_cap(image_norm_A, patch_size) # Higher means hazier in normalized image
    cap_channel_norm_A_clipped = np.clip(cap_channel_norm_A,0,None)
    if np.max(cap_channel_norm_A_clipped) > 1e-6:
        cap_channel_norm_A_normalized = cap_channel_norm_A_clipped / np.max(cap_channel_norm_A_clipped)
    else:
        cap_channel_norm_A_normalized = np.zeros_like(cap_channel_norm_A_clipped)


    t_tilde = 1.0 - omega * cap_channel_norm_A_normalized # Smaller t for hazier regions

    # 4. Refine Transmission (using guided filter)
    t_refined = refine_transmission_guided_filter_dcp(image_bgr_norm, t_tilde,
                                                      guided_filter_radius, guided_filter_eps)

    # 5. Recover Image
    J_dehazed = recover_dehazed_image_dcp(image_bgr_norm, A, t_refined, t0)
    return (J_dehazed * 255).astype(np.uint8)


# --- 4. 全局直方图均衡化 (Global HE) ---
def dehaze_global_he(image_bgr):
    if image_bgr.dtype != np.uint8:
        image_bgr = (image_bgr * 255).astype(np.uint8)

    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    final_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return final_img


# --- Main execution and comparison ---
if __name__ == '__main__':
    # --- Configuration ---
    img_path = 'Project_X/2.png' # REPLACE WITH YOUR IMAGE
    output_folder = "dehazing_comparison_results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_filename = os.path.splitext(os.path.basename(img_path))[0]

    # --- Load Image ---
    try:
        image_hazy_orig_bgr = cv2.imread(img_path)
        if image_hazy_orig_bgr is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
    except FileNotFoundError as e:
        print(e)
        print("Using a synthetic hazy image for demonstration.")
        clean_img = np.zeros((300, 400, 3), dtype=np.float64)
        clean_img[:, :150, 0] = 0.8; clean_img[:, 150:250, 1] = 0.7; clean_img[:, 250:, 2] = 0.9
        X, Y = np.meshgrid(np.linspace(0,1,400), np.linspace(0,1,300))
        transmission_syn = 0.3 + 0.5 * (X + Y) / 2
        transmission_syn_3ch = np.expand_dims(transmission_syn, axis=2)
        A_syn = np.array([[[0.8, 0.85, 0.9]]]) # BGR
        image_hazy_orig_float = clean_img * transmission_syn_3ch + A_syn * (1 - transmission_syn_3ch)
        image_hazy_orig_float = np.clip(image_hazy_orig_float, 0, 1)
        image_hazy_orig_bgr = (image_hazy_orig_float * 255).astype(np.uint8)
        img_path = "synthetic_hazy_image.png"
        base_filename = os.path.splitext(os.path.basename(img_path))[0]

    results = {"Original": image_hazy_orig_bgr}
    print(f"Processing image: {img_path}")

    print("Applying DCP...")
    results["DCP"] = dehaze_dcp(image_hazy_orig_bgr.copy(), patch_size=15, omega=0.95, t0=0.1, guided_filter_radius=60, guided_filter_eps=0.001)

    print("Applying CLAHE...")
    results["CLAHE"] = dehaze_clahe(image_hazy_orig_bgr.copy(), clip_limit=2.0, tile_grid_size=(8,8))

    print("Applying CAP (simplified)...")
    results["CAP"] = dehaze_cap(image_hazy_orig_bgr.copy(), patch_size=15, omega=0.9, t0=0.1, guided_filter_radius=60, guided_filter_eps=0.001)

    print("Applying Global HE...")
    results["Global HE"] = dehaze_global_he(image_hazy_orig_bgr.copy())

    # --- Display and Save Results ---
    num_algorithms = len(results)
    plt.figure(figsize=(5 * num_algorithms, 5)) # Adjust figure size

    for i, (name, img_bgr) in enumerate(results.items()):
        plt.subplot(1, num_algorithms, i + 1)
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis('off')
        # Save individual results
        save_path = os.path.join(output_folder, f"{base_filename}_{name.replace(' ', '_')}.png")
        cv2.imwrite(save_path, img_bgr)
        print(f"Saved: {save_path}")

    plt.tight_layout()
    plt.show()

    print("Comparison complete. Results saved in:", output_folder)