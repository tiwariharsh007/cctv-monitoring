import csv

def generate_report(data, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data["columns"])
        for row in data["rows"]:
            writer.writerow(row)
    return filename
