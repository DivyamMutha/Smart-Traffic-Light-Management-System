# Smart-Traffic-Light-Management-System
Traffic congestion is a major issue in urban areas, causing delays, fuel wastage, and accidents. Traditional fixed-timer traffic signals fail to adapt to real-time traffic flow. This project proposes a Smart Traffic Light Management System using Artificial Intelligence and Machine Learning (AI/ML) to optimize signal timings dynamically.
The system employs computer vision with YOLOv10 to detect vehicles, estimate speed, identify ambulances, and manage green light allocation adaptively. A priority mechanism ensures immediate passage for ambulances. The system also detects violations such as over speeding and red-light jumping, logging details into CSV files with snapshots for evidence.
This model also detects emergency services (police patrol,ambulances) an gives them first priority for safe passage.
The prototype was implemented using Python, OpenCV, Py.Torch, and YOLOv10, processing multiple video streams representing lanes. Results show improved efficiency in lane switching, emergency handling, and violation detection, proving its potential for future smart city deployments.
  
Complete Data Flow
1.	Video Feed → YOLOv10 → State (Vehicle Counts).
2.	State → DQN Agent → Action (Next Phase/Duration).
3.	Action → System API → Traffic Simulator (Actuation).
4.	Simulator → Reward → DQN Agent (Learning).

