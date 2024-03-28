import matplotlib.pyplot as plt

# Format of results file:
# - (separator)
# query_type, query_id, time_taken
# doc_id, score
# doc_id, score
# doc_id, score
# - (separator)

def analyse_file(file_name):
    is_ranking = False
    is_header = True
    results = {}
    with open(file_name, 'r') as file:
        lines = file.readlines()
        lines = [line.replace('\n','') for line in lines]
        current_header = ''
        for line in lines:
            if line == '-':
                if is_ranking:
                    is_ranking = False
                    is_header = True
                elif is_header:
                    is_header = False
                    is_ranking = True
            elif is_header:
                header = line.split(',')
                current_header = f'{header[1]}_{header[0]}'
                results[current_header] = {'time': header[2] , 'rankings': []}
            elif is_ranking:
                ranking = line.split(',')
                results[current_header]['rankings'].append({'doc_id': ranking[0], 'score': round(float(ranking[1]),4)})
    return results

def create_query_speed_chart(results, title):
    query_types = []
    times = []
    for query_type, data in results.items():
        query_types.append(query_type)
        times.append(float(data['time']))  # Converting time to float

    # Set Y-axis limits based on the maximum and minimum values of time
    min_time = min(times)
    max_time = max(times)

    # Adding some buffer space to the Y-axis range
    y_buffer = 0.1 * (max_time - min_time)
    y_min = min_time - y_buffer
    y_max = max_time + y_buffer

    # Creating the bar graph with adjusted Y-axis limits
    plt.figure(figsize=(10, 6))
    plt.bar(query_types, times, color='skyblue')
    plt.xlabel('Query')
    plt.ylabel('Time')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')  # Rotating x-axis labels for better readability
    plt.ylim(y_min, y_max)  # Setting Y-axis limits
    plt.tight_layout()

    # Displaying the bar graph
    plt.show()

def compare_query_speeds(file_name_1, file_name_2, query_type):
    results_1 = analyse_file(file_name_1)
    results_2 = analyse_file(file_name_2)

    times_1 = {}
    times_2 = {}

    # Extract query types and times for the given query type from both files
    for query, data in results_1.items():
        if query_type in query:
            times_1[query] = float(data['time'])

    for query, data in results_2.items():
        if query_type in query:
            times_2[query] = float(data['time'])

    query_types = list(times_1.keys())
    # Remove type from query types
    
    times_file_1 = list(times_1.values())
    times_file_2 = [times_2.get(query, 0) for query in query_types]

    # Create a bar graph to compare query speeds between the two files for the given query type
    plt.figure(figsize=(10, 6))
    plt.bar(query_types, times_file_1, color='skyblue', label='Without Stopwords')
    plt.bar(query_types, times_file_2, color='salmon', label='With Stopwords', alpha=0.5)

    plt.xlabel(f'{query_type} Query Type')
    plt.ylabel('Time')
    plt.title(f'{query_type} Query Speeds Comparison')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.show()

'''
results = analyse_file('results/results1.txt')
cosine_results = {}
tfidf_results = {}

# Loop through results, separate by cosine and tfidf
for query_type, data in results.items():
    query_array = query_type.split('_')
    if query_array[1] == 'Cosine-Similarity':
        cosine_results[query_array[0]] = data
    elif query_array[1] == 'TF-IDF':
        tfidf_results[query_array[0]] = data
'''

compare_query_speeds('results/original.txt', 'results/results2.txt',"Cosine-Similarity")