# Drone-Payload-Based-Anomaly-Detection-System
Unmanned Aerial Vehicles (UAVs) are increasingly deployed across a wide range of applications including surveillance, disaster management, traffic monitoring, and environmental analysis. These applications demand systems capable of real-time situational awareness, anomaly detection, and reliable remote communication. However, achieving such capabilities on a drone platform presents several challenges, including limited onboard computational resources, constrained communication bandwidth, and the need for robust multi-sensor integration. 
This work proposes a drone payload-based anomaly detection system that integrates heterogeneous sensing modalities with computer vision to enable intelligent monitoring. The system is built around the Raspberry Pi 4 Model B, which serves as the onboard processing unit responsible for sensor data acquisition, preprocessing, and communication. 
The payload incorporates multiple sensors, including the MPU6050 for inertial measurement, the BMP280 for pressure and altitude estimation, and the NEO-6M GPS Module for real-time positioning. In addition, visual data is captured using the Raspberry Pi Camera Module 3, which provides high-quality RGB image streams essential for computer vision tasks. To enable remote monitoring, the system utilizes an LTE communication module configured as a modem, with secure connectivity facilitated through Tail scale. This allows seamless communication between the drone and a ground station, where advanced processing such as YOLO-based object detection is performed. 
Unlike conventional systems that rely on continuous data streaming, the proposed architecture adopts a hybrid edge–ground processing model, where lightweight preprocessing is performed onboard, and computationally intensive tasks are offloaded to the ground station. This approach ensures efficient bandwidth utilization while maintaining real-time responsiveness. The system is designed to detect anomalies such as traffic congestion, crowd formation, fire or smoke presence, and abnormal drone motion. 

Hardware Requirements 

• Raspberry Pi 4 
• MPU6050 IMU Sensor
• BMP280 Pressure Sensor 
• GPS Module 
• Raspberry Pi Camera Module 3 
• LTE USB Modem / LTE HAT 
• Drone Frame and Flight Controller 
• Battery and Power Module 

Software Requirements 

• Raspberry Pi OS 
• Python • OpenCV • YOLOv8 1  • Tailscale VPN • PyQt / Tkinter • Flask / Socket Programming 
