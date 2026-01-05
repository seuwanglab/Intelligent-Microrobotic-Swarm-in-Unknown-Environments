import cv2
import torch
import time
import os
import numpy as np
from datetime import datetime
from collections import deque
from threading import Thread, Lock
import csv
import signal
import threading
import sys
import socket
from config import YamlParser
from tool import create_env
from agent.ppo import PPO, _Normal
from agent.transformer import TransformerBlock


def save_attention_to_csv(block_attention, num_blocks, num_heads, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Save only once per run, so overwrite previous files
    for block_idx in range(num_blocks):
        for head_idx in range(num_heads):
            data = np.array(block_attention[block_idx][head_idx])
            columns = [f'Key {i}' for i in range(128)]
            csv_filename = os.path.join(output_dir, f'block{block_idx}_head{head_idx}.csv')
            with open(csv_filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Head'] + columns)
                for i, row in enumerate(data):
                    writer.writerow([f'Head {head_idx}'] + row.tolist())
            print(f'Saved attention for block {block_idx} head {head_idx} to {csv_filename}')


def init_memory(config, max_episode_steps, device):
    """
    Initialize memory, mask, and indices required for the transformer module
    """
    mem_len = config['memory_length']
    # Lower triangular matrix as mask
    memory_mask = torch.tril(torch.ones((mem_len, mem_len)))
    memory = torch.zeros((1, max_episode_steps, config['n_blocks'], config['embedding_dim'])).to(device)
    # repetitions shape: (memory_length-1, memory_length)
    repetitions = torch.repeat_interleave(torch.arange(0, mem_len).unsqueeze(0),
                                          mem_len - 1, dim=0).long()
    # Generate continuous memory indices
    memory_indices = torch.stack([torch.arange(i, i + mem_len) for i in range(max_episode_steps - mem_len + 1)]).long()
    # Merge repetitions and memory_indices into complete index matrix, ensuring total length is max_episode_steps
    memory_indices = torch.cat((repetitions, memory_indices))
    return memory, memory_mask, memory_indices


def data_handler(info):
    pass


def transfer_dict_to_tensor(info):
    """
    Convert information from info dictionary to tensor
    Note: Uses global variables last_action_0, last_action_1
    """
    global last_action_0, last_action_1  # Explicitly declare global variables
    observation_position = np.array([float(info['observation_x']), float(info['observation_y'])])
    target_position = np.array([float(info['target_x']), float(info['target_y'])])
    target_obs_vector = target_position - observation_position
    target_obs_angle = np.arctan2(target_obs_vector[1], target_obs_vector[0])
    target_obs_distance = np.linalg.norm(target_obs_vector)
    data = [
        float(last_action_0),
        float(last_action_1),
        float(info['observation_x']) / 40,  # convert pixel to mm
        float(info['observation_y']) / 40,
        float(info['observation_w']) / 40,
        float(info['observation_h']) / 40,
        float(info['vx']) / 40,
        float(info['vy']) / 40,  
        target_obs_distance / 40,
        target_obs_angle
    ]
    # Add obstacle data, sorted by distance
    obstacles = sorted(
        info['obstacles'],
        key=lambda obs: np.sqrt((float(obs['x']) - float(info['observation_x'])) ** 2 +
                                (float(obs['y']) - float(info['observation_y'])) ** 2)
    )
    for obs in obstacles:
        distance = np.sqrt((float(obs['x']) - float(info['observation_x'])) ** 2 +
                           (float(obs['y']) - float(info['observation_y'])) ** 2)
        if distance <= 200:
            obstacle = np.array([float(obs['x']), float(obs['y'])])
            obst_obs_vector = obstacle - observation_position
            obst_obs_angle = np.arctan2(obst_obs_vector[1], obst_obs_vector[0])
            data += [
                float(obs['state']),
                distance / 40,
                obst_obs_angle,
                float(obs['w']) / 40,
                float(obs['h']) / 40
            ]
    # Ensure data length is 35, pad with 0 if insufficient
    while len(data) < 35:
        data.append(0.0)
    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # shape [1, 35]
    print(tensor)
    tensor = tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    return tensor


class YOLOv5DetectionModule:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        # Load YOLOv5 model locally
        self.yolo_v5 = torch.hub.load('C:/Users/1/Desktop/NMI/opencv/yolov5', 'custom',
                                      path=model_path, source='local')
        self.yolo_v5.eval()
        self.yolo_v5.to(self.device)

        self.target_x = 400
        self.target_y = 500
        self.obs_x = 400
        self.obs_y = 300
        self.list_vx = deque(maxlen=30)
        self.list_vy = deque(maxlen=30)
        self.last_target_update = self.last_stats_time = time.time()

        self.lock = Lock()
        self.latest_info = None

        self.prev_class1_info = {
            'x': 400,
            'y': 300,
            'w': 60,
            'h': 60,
            'vx': 0,
            'vy': 0
        }

        self.csv_file = './attention_csv/observation_info.csv'
        self.initialize_csv()
        # Variables to save the last frame for saving images when the program ends
        self.last_display_frame1 = None
        self.last_display_frame2 = None

    def initialize_csv(self):
        """
        Initialize CSV file and write header
        """
        header = [
            'timestamp', 'yolo_delay', 'target_x', 'target_y',
            'observation_x', 'observation_y', 'observation_w', 'observation_h',
            'vx', 'vy'
        ]
        for i in range(1, 6):
            header += [f'obstacle_{i}_state', f'obstacle_{i}_x',
                       f'obstacle_{i}_y', f'obstacle_{i}_w', f'obstacle_{i}_h']
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)

    def data_to_csv(self, info):
        """
        Write collected data to CSV file
        """
        row = [
            info['timestamp'], info['yolo_delay'], info['target_x'], info['target_y'],
            info['observation_x'], info['observation_y'],
            info['observation_w'], info['observation_h'],
            info['vx'], info['vy']
        ]
        for obs in info['obstacles']:
            row += [obs['state'], obs['x'], obs['y'], obs['w'], obs['h']]
        while len(row) < 10 + 5 * 5:
            row += [0, 0, 0, 0, 0]
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def draw_dashed_circle(self, img, center, radius, color, thickness, segments=20):
        """
        Draw dashed circle on image
        """
        for i in range(segments):
            start_angle = i * (2 * np.pi / segments)
            end_angle = start_angle + (np.pi / segments)
            start_point = (
                int(center[0] + radius * np.cos(start_angle)),
                int(center[1] + radius * np.sin(start_angle))
            )
            end_point = (
                int(center[0] + radius * np.cos(end_angle)),
                int(center[1] + radius * np.sin(end_angle))
            )
            cv2.line(img, start_point, end_point, color, thickness)

    def create_circular_mask(self, img, center, radius):
        """
        Create circular mask based on center and radius
        """
        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = dist_from_center > radius
        return mask

    def calculate_velocity(self):
        """
        Calculate the accumulated displacement of the target in the queue, approximated as velocity
        """
        if len(self.list_vx) < 2 or len(self.list_vy) < 2:
            return 0, 0
        vx = self.list_vx[-1] - self.list_vx[0]
        vy = self.list_vy[-1] - self.list_vy[0]
        return vx, vy

    def process_frame1(self, frame):
        """
        Process camera frame: detect target (class1) and nearby obstacles (class0)
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.draw_dashed_circle(frame, (self.target_x, self.target_y), 50, (150, 36, 63), 3)

        with torch.no_grad():
            results = self.yolo_v5(img_rgb, size=640)
        detections = results.pandas().xyxy[0]

        class1_info = None
        max_area = 0  # Store maximum area
        for _, row in detections.iterrows():
            if int(row['class']) == 1 and float(row['confidence']) > 0.5:
                xmin, ymin, xmax, ymax = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']].values)
                w = xmax - xmin
                h = ymax - ymin
                area = w * h  # Calculate area
                if area > max_area:
                    max_area = area
                    center_x = xmin + w / 2
                    center_y = ymin + h / 2
                    self.obs_x = center_x
                    self.obs_y = center_y
                    class1_info = {'x': center_x, 'y': center_y, 'w': w, 'h': h}

        # If class1 is not found, use previous frame information and set velocity to 0
        if class1_info is None:
            class1_info = self.prev_class1_info.copy()
            vx, vy = 0, 0
        else:
            self.list_vx.append(class1_info['x'])
            self.list_vy.append(class1_info['y'])
            self.prev_class1_info = class1_info.copy()
            vx, vy = self.calculate_velocity()

        # Collect all class0 detection results and filter obstacles closest to class1
        obstacles_near_class1 = []
        if class1_info:
            for _, row in detections.iterrows():
                if int(row['class']) == 0:
                    xmin, ymin, xmax, ymax = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']].values)
                    w = xmax - xmin
                    h = ymax - ymin
                    center_x = xmin + w / 2
                    center_y = ymin + h / 2
                    dist = np.sqrt((center_x - class1_info['x']) ** 2 + (center_y - class1_info['y']) ** 2)
                    if dist <= 225:
                        if dist >=25:
                            obstacles_near_class1.append({
                                'state': 1.0,
                                'x': center_x,
                                'y': center_y,
                                'w': w,
                                'h': h
                            })

        # Ensure obstacle data always has 5 entries, pad with 0 if insufficient
        while len(obstacles_near_class1) < 5:
            obstacles_near_class1.append({'state': 0, 'x': 0, 'y': 0, 'w': 0, 'h': 0})
        obstacles_near_class1 = obstacles_near_class1[:5]

        return class1_info, obstacles_near_class1, vx, vy

    def process_frame2(self, frame, observation_center):
        h, w = frame.shape[:2]

        if observation_center is None:
            masked_frame = frame.copy()
            visible_mask = np.ones((h, w), dtype=bool)
        else:
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - observation_center[0]) ** 2 +
                                       (Y - observation_center[1]) ** 2)
            visible_mask = dist_from_center <= 225
            masked_frame = np.zeros_like(frame, dtype=np.uint8)
            masked_frame[visible_mask] = frame[visible_mask]

            # Convert to RGB before passing to detection model
        img_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            results = self.yolo_v5(img_rgb, size=640)
        detections = results.pandas().xyxy[0]

        max_area = 0
        max_area_bbox = None
        center_obs_x = 0
        center_obs_y = 0

        for _, row in detections.iterrows():
            cls = int(row['class'])
            xmin, ymin, xmax, ymax = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']].values)
            w_box = xmax - xmin
            h_box = ymax - ymin
            
            area = w_box * h_box

            if cls == 1 and float(row['confidence']) > 0.5:
                if area > max_area:
                    max_area = area
                    max_area_bbox = (xmin, ymin, xmax, ymax)
                    center_obs_x = xmin + w_box/2
                    center_obs_y = ymin + h_box/2

        for _, row in detections.iterrows():
            cls = int(row['class'])
            xmin, ymin, xmax, ymax = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']].values)
            w_box = xmax - xmin
            h_box = ymax - ymin
            obst_x = xmin + w_box/2
            obst_y = ymin + h_box/2
            dist = np.sqrt((center_obs_x - obst_x) ** 2 + (center_obs_y -obst_y) ** 2)
            
            area = w_box * h_box

            if cls == 0:
                if dist >= 25:
                    cv2.rectangle(masked_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        if max_area_bbox:
            xmin, ymin, xmax, ymax = max_area_bbox
            cv2.rectangle(masked_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # **⚠️ Draw target position markers after YOLO detection**
        

        # Draw arrow on independent arrow layer to ensure it doesn't participate in detection
        arrow_layer = np.zeros_like(frame, dtype=np.uint8)
        arrow_center = (740, 60)
        arrow_size = 30

        dx = self.target_x - self.obs_x
        dy = self.target_y - self.obs_y
        self.draw_dashed_circle(masked_frame, (self.target_x, self.target_y), 50, (150, 36, 63), 3)
        angle = np.degrees(np.arctan2(-dy, dx))

        # Calculate arrow vertex coordinates
        tip = (arrow_center[0] + arrow_size * np.cos(np.radians(angle)),
               arrow_center[1] - arrow_size * np.sin(np.radians(angle)))
        left = (arrow_center[0] - arrow_size * np.cos(np.radians(angle - 40)),
                arrow_center[1] + arrow_size * np.sin(np.radians(angle - 40)))
        base = (arrow_center[0] - arrow_size * 0.35 * np.cos(np.radians(angle)),
                arrow_center[1] + arrow_size * 0.35 * np.sin(np.radians(angle)))
        right = (arrow_center[0] - arrow_size * np.cos(np.radians(angle + 40)),
                 arrow_center[1] + arrow_size * np.sin(np.radians(angle + 40)))

        arrow_points = np.array([tip, left, base, right], dtype=np.int32)

        # Fill arrow shape
        cv2.fillPoly(arrow_layer, [arrow_points], color=(255, 165, 0))
        cv2.polylines(arrow_layer, [arrow_points], isClosed=True, color=(255, 165, 0), thickness=1)

        # **Final composition: arrow layer + masked area**
        final_frame = arrow_layer.copy()
        final_frame[visible_mask] = masked_frame[visible_mask]

        return final_frame

    def detection_state(self, data_callback=None):
        """
        Continuously capture video frames using camera, perform detection, and record results via callback and CSV
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if not cap.isOpened():
            print("Cannot open camera")
            return

        # Video recording settings
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out1 = cv2.VideoWriter('output_cv1.avi', fourcc, 20.0, (800, 600))
        out2 = cv2.VideoWriter('output_cv2.avi', fourcc, 20.0, (800, 600))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            if current_time - self.last_target_update >= 2:
                self.target_x += 1.25
                self.last_target_update = current_time


            display_frame1 = frame.copy()
            display_frame2 = frame.copy()

            # Process first frame: detect target and obstacles
            class1_info, obstacles_near_class1, vx, vy = self.process_frame1(display_frame1)

            # Determine observation center based on target information
            observation_center = (int(class1_info['x']), int(class1_info['y'])) if class1_info else None

            # Process second frame: masked detection
            display_frame2 = self.process_frame2(display_frame2, observation_center)

            # Save current processed frames for saving images when program exits
            self.last_display_frame1 = display_frame1.copy()
            self.last_display_frame2 = display_frame2.copy()

            # Calculate and record data once per second
            if current_time - self.last_stats_time >= 1:
                self.last_stats_time = current_time
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                target_x = self.target_x
                target_y = self.target_y

                info = {
                    'timestamp': timestamp,
                    'yolo_delay': (time.time() - current_time) * 1000,
                    'target_x': target_x,
                    'target_y': target_y,
                    'observation_x': class1_info['x'],
                    'observation_y': class1_info['y'],
                    'observation_w': class1_info['w'],
                    'observation_h': class1_info['h'],
                    'vx': vx,
                    'vy': vy,
                    'obstacles': obstacles_near_class1
                }
                with self.lock:
                    self.latest_info = info
                if data_callback:
                    data_callback(info)
                self.data_to_csv(info)

            # Save video frames
            out1.write(display_frame1)
            out2.write(display_frame2)

            # Display windows
            cv2.imshow("CV1 - Global View", display_frame1)
            cv2.imshow("CV2 - Partial View", display_frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out1.release()
        out2.release()
        cv2.destroyAllWindows()

    def get_latest_info(self):
        with self.lock:
            return self.latest_info


if __name__ == "__main__":
    # Declare global variables for sharing latest action information across functions
    global last_action_0, last_action_1
    last_action_0 = 0
    last_action_1 = 0
    start_time = time.time()

    # Create UDP Socket for sending action commands
    s_pattern = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_pattern.bind(('127.0.0.1', 61556))
    s_pattern.settimeout(1)
    print('Bind UDP on 61556...')


    # Define Ctrl+C exit handler function
    def client_exit(num, frame):
        print("Exit data collection")
        s_pattern.close()
        sys.exit(0)


    signal.signal(signal.SIGINT, client_exit)
    exit_event = threading.Event()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path1 = 'models/yolov5.pt'
    model_path2 = 'models/Turbo.pt'
    yolo_detection = YOLOv5DetectionModule(model_path1, device)

    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)

    config = YamlParser(path='config/dynamic_obstacle.yaml').get_config()
    env = create_env(config['environment'])

    jit_model = torch.jit.load(model_path2, map_location=device)
    model = PPO(config, env.observation_space, env.action_space.shape[0]).to(device)
    model.load_state_dict(jit_model.state_dict())
    model.to(device)
    model.eval()

    memory, memory_mask, memory_indices = init_memory(config['transformer'], 2048, device)
    memory_length = config['transformer']['memory_length']
    t = 0  # Current time step
    direction_angle = 0.
    pitch_angle = 0.

    num_blocks = config['transformer']['n_blocks']
    num_heads = config['transformer']['n_heads']
    block_attention = {block_idx: {head_idx: [] for head_idx in range(num_heads)}
                       for block_idx in range(num_blocks)}

    # Start detection thread
    detection_thread = Thread(target=yolo_detection.detection_state, args=(data_handler,))
    detection_thread.start()
    last_print_time = time.time()

    try:
        last_print_time = time.time()  # Last print time
        while True:
            # Send action commands to corresponding ports
            a_angle = str(direction_angle).encode('utf-8')
            p_pitch_angle = str(pitch_angle).encode('utf-8')
            s_pattern.sendto(a_angle, ('127.0.0.1', 61557))
            s_pattern.sendto(p_pitch_angle, ('127.0.0.1', 61558))

            latest_info = yolo_detection.get_latest_info()
            current_time = time.time()
            # Wait for a certain time before starting model inference to avoid premature processing
            if latest_info and current_time - last_print_time >= 1 and current_time - start_time >= 10:
                last_print_time = current_time

                observation = transfer_dict_to_tensor(latest_info)
                print(observation)
                # Ensure t is within memory_indices range
                if t >= memory_indices.shape[0]:
                    t = memory_indices.shape[0] - 1
                in_memory = memory[0, memory_indices[t].unsqueeze(0)]
                t_ = max(0, min(t, memory_length - 1))
                mask = memory_mask[t_].unsqueeze(0).bool()
                indices = memory_indices[t].unsqueeze(0)
                model_start_time = time.time()
                # Model inference
                policy, value, new_memory, attention = model(observation, in_memory, indices, mask)
                attention_np = attention.detach().cpu().numpy()
                mask_np = mask.cpu().numpy().squeeze()

                for block_idx in range(attention_np.shape[1]):
                    block_attn = attention_np[0, block_idx, :, 0, :]
                    for head_idx in range(block_attn.shape[0]):
                        head_attn = block_attn[head_idx, :]
                        head_attn_masked = head_attn * mask_np
                        block_attention[block_idx][head_idx].append(head_attn_masked)
                model_inference_time = (time.time() - model_start_time) * 1000

                memory[:, t] = new_memory
                t += 1

                # Calculate action based on policy output, assuming policy contains mean attribute
                mean = policy.mean
                action = torch.tanh(mean)
                action = action.detach().cpu().numpy()[0]
                print(action)
                vel = (action[0] + 1) / 2
                angle = np.degrees(action[1] * np.pi)
                last_action_0 = action[0]
                last_action_1 = action[1]
                pitch_angle = vel * 2.25
                direction_angle = -angle
                formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_print_time))
                with open('./attention_csv/action.txt', 'a') as f:
                    f.write(f"{formatted_time} {np.linalg.norm(action):.2f} {direction_angle:.2f} "
                            f"Model_inference_time: {model_inference_time:.2f} ms\n")

            # Check if 'q' key is pressed to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit and save attention data")
                break

    except KeyboardInterrupt:
        print("Program interrupted, saving attention data")
        # Save attention data when program is interrupted
        output_dir = './attention_csv'
        save_attention_to_csv(block_attention, num_blocks, num_heads, output_dir)
        detection_thread.join()

    finally:
        # Save attention data when program exits
        print("Program ended, saving attention data")
        output_dir = './attention_csv'
        save_attention_to_csv(block_attention, num_blocks, num_heads, output_dir)
        s_pattern.close()
