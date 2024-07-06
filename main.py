import time
import dxcam
import pygetwindow
import cv2

orig_screen_width, orig_screen_height = 1920, 1032

def is_template_in_region(image, template_path, region):
    current_height, current_width = image.shape[:2]
    orig_x, orig_y, orig_width, orig_height = region
    width_ratio, height_ratio = current_width / orig_screen_width, current_height / orig_screen_height
    new_x, new_y = int(orig_x * width_ratio), int(orig_y * height_ratio)
    new_width, new_height = int(orig_width * width_ratio), int(orig_height * height_ratio)
    cropped_image = image[new_y:new_y + new_height, new_x:new_x + new_width]
    result = cv2.matchTemplate(cropped_image, load_image(template_path, current_width, current_height), cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val > 0.8

def load_image(image_path, width, height):
    width_ratio, height_ratio = width / orig_screen_width, height / orig_screen_height
    image = cv2.imread(image_path)
    orig_height, orig_width = image.shape[:2]
    resized_image = cv2.resize(image, (int(orig_width * width_ratio), int(orig_height * height_ratio)))
    return resized_image

def get_in_game_state(image):
    if is_in_offer_popup(image): return "popup"
    if is_in_shop(image): return "shop"
    if is_in_lobby(image): return "lobby"
    if is_in_end_of_a_match(image): return "end"
    if is_in_brawler_selection(image): return "brawler_selection"
    return "match"

def is_in_shop(image) -> bool:
    return is_template_in_region(image, 'images_to_detect/powerpoint.png', (977, 28, 109, 87))

def is_in_brawler_selection(image) -> bool:
    return is_template_in_region(image, 'images_to_detect/brawler_menu_task.png', (1384, 28, 200, 127))

def is_in_offer_popup(image) -> bool:
    return is_template_in_region(image, 'images_to_detect/close_popup.png', (1235, 70, 555, 278))

def is_in_lobby(image) -> bool:
    return is_template_in_region(image, 'images_to_detect/lobby_menu.png', (1654, 31, 186, 105))

def is_in_end_of_a_match(image):
    return is_template_in_region(image, 'images_to_detect/end_battle_top_left_continue_corner.png', (1695, 881, 181, 138))


time.sleep(1)
camera = dxcam.create()
screenshot = camera.grab()
screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
start_time = time.time()
print(get_in_game_state(screenshot_bgr))
print(f"Time taken: {time.time() - start_time}")
