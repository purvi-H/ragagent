import os
import csv

# Path to the directory containing chart data files
# chart_data_directory = "/exports/eddie/scratch/s2024596/ragagent/dataset"

# Path to the directory containing chart data files
chart_data_directory = os.getcwd()
overall = []

# Read the output.txt file
with open("output.txt", "r") as output_file:
    lines = output_file.read().splitlines()

    # Iterate over each line in lines
    for i, line in enumerate(lines):
        split = line.split("|")
        type = split[0].strip()
        chart_data_filename = split[1].strip().replace('.txt', '.csv')
        if type == "Simple":
            # Get the file stored in split[1] from the data folder
            chart_data_path = os.path.join(chart_data_directory, "data", chart_data_filename)
            with open(chart_data_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                chart_data = list(reader)
        else:        
            chart_data_path = os.path.join(chart_data_directory, "multicolumn/data", chart_data_filename)
            with open(chart_data_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                chart_data = list(reader)

        # with open('/exports/eddie/scratch/s2024596/ragagent/dataset/charttitle.txt', 'r') as file:
        #     chart_title_lines = file.readlines()
        #     chart_title = chart_title_lines[i].strip()

        chart_title_path = os.path.join(chart_data_directory, 'charttitle.txt')
        with open(chart_title_path, 'r') as file:
            chart_title_lines = file.readlines()
            chart_title = chart_title_lines[i].strip()

        # with open('/exports/eddie/scratch/s2024596/ragagent/dataset/charttype.txt', 'r') as file:
            # chart_type_lines = file.readlines()
            # chart_type = chart_type_lines[i].strip()
        chart_type_path = os.path.join(chart_data_directory, 'charttype.txt')
        with open(chart_type_path, 'r') as file:
            chart_type_lines = file.readlines()
            chart_type = chart_type_lines[i].strip()

        # # Call the function with obtained chart_type, chart_data, and chart_title
        # response = llm_chain.run(chart_type, chart_data, chart_title)
        # print(response, "/n")
        overall.append((chart_type, chart_title, chart_data))

print(overall[5])