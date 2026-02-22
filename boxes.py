import cv2
import numpy as np
import os
import glob
import json

# ================= CONFIGURATION =================
BASE_PATH = "data/emre_dataset/my_seq_1Emre"
OUTPUT_FOLDER = "calibration_visual_test"
CALIBRATION_FILE = "/data/home/student_colak/MVGFormer/data/emre_dataset/my_seq_1Emre/calibration_my_seq_1YEDEK.json"
CUBE_SIZE = 100 
THICKNESS = 4

# Added a 6th slot here if you have one, otherwise the code handles 5 just fine!
CAMERA_NAMES = ["00_19", "00_25", "00_30", "00_31", "00_39"]

# Grid settings
GRID_IMG_WIDTH = 800 # Resize each image to this width so the final grid isn't massive

def get_cube_points(size, center):
    x, y, z = center
    hs = size / 4.0
    return np.float32([
        [x-hs, y-hs, z-hs], [x+hs, y-hs, z-hs], [x+hs, y+hs, z-hs], [x-hs, y+hs, z-hs],
        [x-hs, y-hs, z+hs], [x+hs, y-hs, z+hs], [x+hs, y+hs, z+hs], [x-hs, y+hs, z+hs]
    ])

def draw_cube(img, imgpts, color):
    if np.isnan(imgpts).any() or np.isinf(imgpts).any():
        return img
    
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # Draw bottom face
    cv2.drawContours(img, [imgpts[:4]], -1, color, THICKNESS)
    # Draw pillars
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, THICKNESS)
    # Draw top face
    cv2.drawContours(img, [imgpts[4:]], -1, color, THICKNESS)
    return img

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    with open(CALIBRATION_FILE, 'r') as f:
        calibration_dict = json.load(f)
        
    calib_data = {cam["name"]: cam for cam in calibration_dict["cameras"]}

    processed_images = []
    target_h = 0 # Will be calculated based on the first image's aspect ratio

    for cam_name in CAMERA_NAMES:
        print(f"--- Processing Camera: {cam_name} ---")
        
        if cam_name not in calib_data:
            print(f"Warning: Camera {cam_name} not found in calibration file.")
            continue
            
        search_path = os.path.join(BASE_PATH, "hdImgs", cam_name, "*.*")
        files = sorted(glob.glob(search_path))
        if not files:
            print(f"Warning: No images found in {search_path}")
            continue
        
        img = cv2.imread(files[0])
        h, w = img.shape[:2]
        
        # Calculate target height for resizing to maintain aspect ratio
        if target_h == 0:
            aspect_ratio = h / w
            target_h = int(GRID_IMG_WIDTH * aspect_ratio)

        cam_data = calib_data[cam_name]
        
        K = np.array(cam_data["K"], dtype=np.float64)
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]

        if cx > w or cy > h:
            K = np.array([[fx/2.0, 0, cx/2.0], [0, fy/2.0, cy/2.0], [0, 0, 1]], dtype=np.float64)
        
        dist_coeffs = np.array(cam_data["distCoef"], dtype=np.float64)
        R = np.array(cam_data["R"], dtype=np.float64)
        T = np.array(cam_data["t"], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(R)

        # Origin Cube (Red)
        pts_origin = get_cube_points(CUBE_SIZE, (0,0,0))
        proj_origin, _ = cv2.projectPoints(pts_origin, rvec, T, K, dist_coeffs)

        # Look-At Cube (Green)
        P_cam_center = np.array([[0], [0], [3000]], dtype=np.float64) 
        P_world_center = np.dot(R.T, (P_cam_center - T))
        pts_center = get_cube_points(CUBE_SIZE/2, P_world_center.flatten()) 
        proj_center, _ = cv2.projectPoints(pts_center, rvec, T, K, dist_coeffs)

        # Draw Cubes
        img = draw_cube(img, proj_origin, (0, 0, 255))
        #img = draw_cube(img, proj_center, (0, 255, 0))
        
        # Resize image and add to our list
        img_resized = cv2.resize(img, (GRID_IMG_WIDTH, target_h))
        
        # Add a text label to the image so you know which camera is which
        cv2.putText(img_resized, cam_name, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        
        processed_images.append(img_resized)

    # ================= BUILD THE GRID =================
    if not processed_images:
        print("No images were processed. Exiting.")
        return

    print("--- Building Layout ---")
    
    # Pad the list with black images until we have exactly 6 images (for a 2x3 grid)
    while len(processed_images) < 6:
        black_img = np.zeros((target_h, GRID_IMG_WIDTH, 3), dtype=np.uint8)
        processed_images.append(black_img)

    # Stack into 2 rows of 3 columns
    row1 = np.hstack((processed_images[0], processed_images[1], processed_images[2]))
    row2 = np.hstack((processed_images[3], processed_images[4], processed_images[5]))
    final_grid = np.vstack((row1, row2))

    save_path = os.path.join(OUTPUT_FOLDER, "calibration_layout.jpg")
    cv2.imwrite(save_path, final_grid)
    print(f"Saved combined layout: {save_path}")

if __name__ == "__main__":
    main()