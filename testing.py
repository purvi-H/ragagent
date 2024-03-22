# import csv
# import json

# # Path to the CSV file
# csv_file_path = "/Users/purviharwani/Desktop/ragagent/dataset/data/0.csv"

# # Read CSV data and convert it to JSON

# with open(csv_file_path, newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     csv_data = [row for row in reader]

# # Convert CSV data to JSON
# json_data = json.dumps(csv_data, indent=1)

# # Print JSON data
# print(json_data)
# from typing import List

# chart_title = "smth"
# chart_data = "else"

# instruction = """
#             Generate 5 questions that can be answered only from this data about {chart_title} : {chart_data}
#             Generate the questions based on the following schema:
#             """
# new = instruction.format(chart_title=chart_title, chart_data=chart_data)
# prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
# print(type(new.to_string()))