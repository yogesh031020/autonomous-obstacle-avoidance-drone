import airsim
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import time

class ObstacleAvoidanceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

print("Loading trained model...")
model = ObstacleAvoidanceCNN()
model.load_state_dict(torch.load(
    "models/best_model.pth",
    map_location=torch.device('cpu')))
model.eval()
print("Model loaded!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

ACTIONS = {
    0: (2,  0,  0, 0.5),
    1: (0, -2,  0, 0.5),
    2: (0,  2,  0, 0.5),
    3: (0,  0, -1, 0.5),
    4: (-2, 0,  0, 0.5),
}
ACTION_NAMES = ["Forward", "Left", "Right", "Up", "Back"]
ACTION_COLORS = [
    (0, 255, 0),
    (255, 165, 0),
    (0, 165, 255),
    (255, 0, 0),
    (0, 0, 255),
]

print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()

print("Resetting drone...")
client.reset()
time.sleep(3)

client.enableApiControl(True)
time.sleep(1)
client.armDisarm(True)
time.sleep(2)

print("Taking off...")
client.takeoffAsync().join()
time.sleep(4)

state = client.getMultirotorState()
pos = state.kinematics_estimated.position
print("Drone height: " + str(round(-pos.z_val, 2)) + "m")

print("Going up higher...")
client.moveByVelocityAsync(0, 0, -8, 5).join()
time.sleep(2)

state = client.getMultirotorState()
pos = state.kinematics_estimated.position
print("New height: " + str(round(-pos.z_val, 2)) + "m")

print("Moving into neighborhood...")
client.moveByVelocityAsync(5, 0, 0, 4).join()
time.sleep(1)

responses = client.simGetImages([
    airsim.ImageRequest(
        "front_center",
        airsim.ImageType.DepthPlanar,
        True)
])
depth_check = np.array(
    responses[0].image_data_float,
    dtype=np.float32)
print("Min distance ahead: " + str(round(float(depth_check.min()), 2)) + "m")
print("Starting autonomous flight...")
print("-" * 45)

step = 0
try:
    while True:
        responses = client.simGetImages([
            airsim.ImageRequest(
                "front_center",
                airsim.ImageType.DepthPlanar,
                True)
        ])

        depth = np.array(
            responses[0].image_data_float,
            dtype=np.float32)
        h = responses[0].height
        w = responses[0].width
        depth = depth.reshape(h, w)

        depth_resized = cv2.resize(depth, (224, 224))
        depth_norm = (
            np.clip(depth_resized / 50.0, 0, 1) * 255
        ).astype(np.uint8)

        img_pil = Image.fromarray(depth_norm)
        tensor = transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            action_id = probs.argmax().item()
            confidence = probs[action_id].item()

        min_dist = depth.min()

        if min_dist < 0.5:
            action_id = 4
            print("CRITICAL - BACKING UP!")
        elif min_dist < 2.5:
            left_half  = depth[:, :w//2].min()
            right_half = depth[:, w//2:].min()
            top_half   = depth[:h//2, :].min()
            if top_half > 5.0:
                action_id = 3
            elif left_half > right_half:
                action_id = 1
            else:
                action_id = 2
            print("OBSTACLE OVERRIDE -> " + ACTION_NAMES[action_id])

        display = cv2.cvtColor(
            depth_norm, cv2.COLOR_GRAY2BGR)
        color = ACTION_COLORS[action_id]

        cv2.putText(display,
            "Action: " + ACTION_NAMES[action_id],
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2)
        cv2.putText(display,
            "Confidence: " + str(round(confidence * 100, 1)) + "%",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1)
        cv2.putText(display,
            "Min Dist: " + str(round(float(min_dist), 2)) + "m",
            (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1)
        cv2.putText(display,
            "Step: " + str(step),
            (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (200, 200, 200), 1)

        cv2.imshow("Autonomous Drone View", display)

        vx, vy, vz, dur = ACTIONS[action_id]
        client.moveByVelocityAsync(vx, vy, vz, dur)

        print("Step " + str(step).zfill(3) +
              " | " + ACTION_NAMES[action_id].ljust(8) +
              " | Confidence: " + str(round(confidence * 100, 1)) + "%" +
              " | Min dist: " + str(round(float(min_dist), 2)) + "m")

        step += 1

        if cv2.waitKey(1) == 27:
            print("Stopped by user!")
            break

        time.sleep(0.3)

except KeyboardInterrupt:
    print("Interrupted!")

cv2.destroyAllWindows()
print("Landing...")
client.landAsync().join()
client.armDisarm(False)
print("Mission Complete!")