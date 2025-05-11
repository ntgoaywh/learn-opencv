import cv2
import numpy as np
import matplotlib.pyplot as plt
import os  # For path operations


# ... (所有函数定义: get_dark_channel, estimate_atmospheric_light,
#      estimate_transmission, refine_transmission_guided_filter,
#      recover_dehazed_image, dehaze_dcp_main 都和之前一样)
# 我将把这些函数折叠起来以便聚焦于修改的部分
# <editor-fold desc="Helper Functions (unchanged from previous version)">
def get_dark_channel(image, patch_size):
    min_channel_img = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel_img, kernel)
    return dark_channel


def estimate_atmospheric_light(image_hazy, dark_channel, percentile=0.001):
    rows, cols, _ = image_hazy.shape
    num_pixels = rows * cols
    num_brightest = int(max(1, num_pixels * percentile))
    dark_channel_flat = dark_channel.flatten()
    brightest_indices = np.argsort(dark_channel_flat)[-num_brightest:]
    image_hazy_flat = image_hazy.reshape(num_pixels, 3)
    A_candidates = image_hazy_flat[brightest_indices]
    A = np.mean(A_candidates, axis=0)
    return A.reshape(1, 1, 3)


def estimate_transmission(image_hazy, A, patch_size, omega=0.95):
    A_safe = np.maximum(A, 1e-6)
    image_norm_A = image_hazy / A_safe
    dark_channel_norm_A = get_dark_channel(image_norm_A, patch_size)
    t_tilde = 1.0 - omega * dark_channel_norm_A
    return t_tilde


def refine_transmission_guided_filter(image_hazy, t_tilde, radius, eps):
    gray_image = cv2.cvtColor(image_hazy.astype(np.float32), cv2.COLOR_BGR2GRAY)
    t_tilde_float32 = t_tilde.astype(np.float32)
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
        t_refined = cv2.ximgproc.guidedFilter(guide=gray_image, src=t_tilde_float32,
                                              radius=radius, eps=eps * eps)  # eps is squared inside for OpenCV
    else:
        print("Warning: cv2.ximgproc.guidedFilter not found. Using Gaussian blur (suboptimal).")
        ksize = 2 * radius + 1
        t_refined = cv2.GaussianBlur(t_tilde_float32, (ksize, ksize), 0)
    return t_refined


def recover_dehazed_image(image_hazy, A, t_refined, t0=0.1):
    t_bounded = np.maximum(t_refined, t0)
    t_bounded_3channel = np.expand_dims(t_bounded, axis=2)
    J = (image_hazy - A) / t_bounded_3channel + A
    J = np.clip(J, 0, 1)
    return J


def dehaze_dcp_main(image_hazy_bgr,
                    patch_size=15,
                    omega=0.95,
                    t0=0.1,
                    guided_filter_radius=40,
                    guided_filter_eps=0.001,
                    atmospheric_light_percentile=0.001):
    if image_hazy_bgr.dtype == np.uint8:
        image_hazy_norm = image_hazy_bgr.astype(np.float64) / 255.0
    elif image_hazy_bgr.max() > 1.0:
        print("Warning: Input float image seems not to be in [0,1] range. Normalizing assuming [0,255] range.")
        image_hazy_norm = image_hazy_bgr.astype(np.float64) / 255.0
    else:
        image_hazy_norm = image_hazy_bgr.astype(np.float64)

    if image_hazy_norm.ndim == 2:
        image_hazy_norm = cv2.cvtColor(image_hazy_norm, cv2.COLOR_GRAY2BGR)
    elif image_hazy_norm.shape[2] == 4:
        image_hazy_norm = cv2.cvtColor(image_hazy_norm, cv2.COLOR_RGBA2BGR)

    # print("1. Computing Dark Channel...") # Can be commented out for cleaner output
    dark_channel = get_dark_channel(image_hazy_norm, patch_size)
    # print("2. Estimating Atmospheric Light...")
    A = estimate_atmospheric_light(image_hazy_norm, dark_channel, percentile=atmospheric_light_percentile)
    # print(f"   Estimated Atmospheric Light (BGR): {A.flatten()}")
    # print("3. Estimating Initial Transmission Map...")
    t_tilde = estimate_transmission(image_hazy_norm, A, patch_size, omega)
    # print("4. Refining Transmission Map using Guided Filter...")
    t_refined = refine_transmission_guided_filter(image_hazy_norm, t_tilde,
                                                  radius=guided_filter_radius,
                                                  eps=guided_filter_eps)
    # print("5. Recovering Haze-Free Image...")
    J_dehazed = recover_dehazed_image(image_hazy_norm, A, t_refined, t0)
    return J_dehazed  # , dark_channel, t_refined (return these if still needed elsewhere)


# </editor-fold>


if __name__ == '__main__':
    # --- Configuration ---
    # 用你自己的图片路径替换
    img_path = '/Users/limttkx/Downloads/图像去雾/Project_X/2.png'  # 例如 'hazy_forest.png'
    # 保存去雾后图像的路径和文件名
    output_dir = 'dehazed_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_dir, f'{base_filename}_dehazed_dcp.png')

    # --- Load Image ---
    try:
        image_hazy_orig = cv2.imread(img_path)
        if image_hazy_orig is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
    except FileNotFoundError as e:
        print(e)
        print("Creating a synthetic hazy image for demonstration.")
        clean_img = np.zeros((300, 400, 3), dtype=np.float64)
        clean_img[:, :150, 0] = 0.8
        clean_img[:, 150:250, 1] = 0.7
        clean_img[:, 250:, 2] = 0.9
        X, Y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 300))
        transmission_syn = 0.3 + 0.5 * (X + Y) / 2
        transmission_syn_3ch = np.expand_dims(transmission_syn, axis=2)
        A_syn = np.array([[[0.8, 0.85, 0.9]]])
        image_hazy_orig_float = clean_img * transmission_syn_3ch + A_syn * (1 - transmission_syn_3ch)
        image_hazy_orig_float = np.clip(image_hazy_orig_float, 0, 1)
        image_hazy_orig = (image_hazy_orig_float * 255).astype(np.uint8)
        img_path = "synthetic_hazy_image.png"  # Update for synthetic image saving
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f'{base_filename}_dehazed_dcp.png')

    # --- Dehaze Parameters ---
    patch_sz = 15
    omega_val = 0.95
    t0_val = 0.1
    gf_radius = 60
    gf_eps_val = 0.001  # K. He 论文中的原始 eps
    atm_light_perc = 0.001

    print(f"Processing image: {img_path}")
    print("Dehazing in progress...")

    # --- Dehaze the image ---
    # J_dehazed, _, _ = dehaze_dcp_main( # If dehaze_dcp_main returns 3 values
    J_dehazed = dehaze_dcp_main(  # If dehaze_dcp_main is modified to return only J_dehazed
        image_hazy_orig,
        patch_size=patch_sz,
        omega=omega_val,
        t0=t0_val,
        guided_filter_radius=gf_radius,
        guided_filter_eps=gf_eps_val,
        atmospheric_light_percentile=atm_light_perc
    )
    print("Dehazing complete.")

    # --- Prepare for Display (BGR to RGB for Matplotlib) ---
    image_hazy_rgb_disp = cv2.cvtColor(image_hazy_orig, cv2.COLOR_BGR2RGB)
    # J_dehazed is already in [0,1] float format, convert to uint8 for display and saving
    J_dehazed_display = (J_dehazed * 255).astype(np.uint8)
    J_dehazed_rgb_disp = cv2.cvtColor(J_dehazed_display, cv2.COLOR_BGR2RGB)

    # --- Display Results ---
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image_hazy_rgb_disp)
    plt.title('Original Hazy Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(J_dehazed_rgb_disp)
    plt.title(f'Dehazed Image (DCP)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # --- Save the Dehazed Image ---
    # J_dehazed is float [0,1]. Convert to uint8 [0,255] for saving with cv2.imwrite
    J_dehazed_to_save = (J_dehazed * 255).astype(np.uint8)
    try:
        cv2.imwrite(save_path, J_dehazed_to_save)
        print(f"Dehazed image saved to: {save_path}")
    except Exception as e:
        print(f"Error saving image: {e}")