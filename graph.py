import matplotlib.pyplot as plt
import numpy as np
import csv
import math

def plot_arc(ax, start, end, center, radius):
    # Calculate angles for arc
    start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
    end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
    
    # Ensure we draw the shorter arc
    if abs(end_angle - start_angle) > math.pi:
        if end_angle > start_angle:
            start_angle += 2*math.pi
        else:
            end_angle += 2*math.pi
            
    # Create points for arc
    theta = np.linspace(start_angle, end_angle, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, 'r-', linewidth=2)

# Create figure
fig, ax = plt.subplots(figsize=(15, 6))

# Read and plot segments
with open('big_windscreen_100.csv', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if parts[0] == 'LINE':
            x = [float(parts[1]), float(parts[3])]
            y = [float(parts[2]), float(parts[4])]
            ax.plot(x, y, 'b-', linewidth=2)
        elif parts[0] == 'ARC':
            start = [float(parts[1]), float(parts[2])]
            end = [float(parts[3]), float(parts[4])]
            center = [float(parts[5]), float(parts[6])]
            radius = float(parts[7])
            plot_arc(ax, start, end, center, radius)

# Set aspect ratio and labels
ax.set_aspect('equal')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_title('Car Windscreen Profile (Blue=Lines, Red=Arcs)')
ax.grid(True)

# Show plot
plt.show()