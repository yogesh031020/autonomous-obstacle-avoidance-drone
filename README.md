\# Autonomous Obstacle Avoidance Drone using Deep Learning in AirSim



A drone that flies autonomously and avoids obstacles in real time using a CNN trained on depth camera images inside Microsoft AirSim simulator.



\## Demo

!\[Autonomous Flight](results/demo.png)



\## Project Overview

\- Drone captures depth images from front camera

\- CNN model predicts flight direction (Forward, Left, Right, Up)

\- Safety override system detects obstacles and reacts in real time

\- Drone flies autonomously without any human control



\## Tech Stack

\- Python 3.10

\- Microsoft AirSim (drone simulator)

\- PyTorch (CNN model)

\- OpenCV (image processing)

\- NumPy, Pandas



\## Project Structure

```

obstacle-avoidance-drone/

├── README.md

├── settings.json

├── requirements.txt

├── src/

│   ├── test\_connection.py

│   ├── capture\_images.py

│   ├── collect\_data.py

│   ├── train\_model.py

│   └── autonomous\_flight.py

├── data/

│   ├── images/

│   └── labels.csv

├── models/

│   └── best\_model.pth

└── results/

&#x20;   └── training\_graph.png

```



\## How It Works

1\. Depth images captured from drone front camera

2\. Images resized to 224x224 and normalized

3\. CNN predicts one of 4 actions: Forward, Left, Right, Up

4\. Safety override triggers when obstacle closer than 5 meters

5\. Stuck detection system escapes left/right loops automatically



\## Model Performance

\- Training samples: 512

\- Validation samples: 129

\- Best validation accuracy: 92%

\- Training epochs: 25



\## Setup Instructions



\### 1. Install dependencies

pip install -r requirements.txt



\### 2. Download AirSim

Download AirSimNH from:

https://github.com/microsoft/AirSim/releases/tag/v1.8.1-windows



\### 3. Run autonomous flight

python src/autonomous\_flight.py



\## Results

\- Drone successfully navigates neighborhood environment

\- Avoids trees, buildings and road obstacles

\- Stuck detection prevents infinite left/right loops

\- Height control keeps drone at optimal flying altitude



\## Author

Yogesh E S

