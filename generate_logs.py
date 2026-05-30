import matplotlib.pyplot as plt
import csv
from datetime import datetime

def generate_graph(csv_file, output_image):
    times = []
    values = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            times.append(datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S'))
            values.append(int(row['value']))

    plt.figure(figsize=(10, 5))
    plt.plot(times, values, marker='o')
    plt.title('Surveillance Data Over Time')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()

generate_graph('logs/data.csv', 'logs/data_plot.png')
