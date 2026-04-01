import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected!")

client.reset()
time.sleep(2)

client.enableApiControl(True)
time.sleep(1)

client.armDisarm(True)
time.sleep(1)

print("Taking off...")
client.takeoffAsync().join()
print("Flying!")
time.sleep(3)

print("Moving forward...")
client.moveByVelocityAsync(3, 0, 0, 5).join()
print("Done!")

client.landAsync().join()
client.armDisarm(False)
print("Landed!")