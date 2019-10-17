import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(line_count=50):
    filepath = "a3.txt"
    full_path = os.path.join('./', filepath)
    data = []
    with open(full_path) as file:
        counter = 0
        for line in file:
            if(counter >= line_count):
                return data
            else:
                # Get the empty values off
                data.append(list(filter(lambda x: len(x) > 0, line.strip().split(" "))))
            counter += 1
    return data

# Return n * 2 matrix with the value
def transform_to_vectors(raw_data):
    if raw_data == None:
        raise ValueError("No data!")
        return
    transformed = np.array(list(map(lambda x: np.array(x).reshape((2, 1)), raw_data)))
    result = transformed.reshape(len(raw_data), 2).astype(np.int)
    return result

def normalize(data):
    def normalize_value(value, min, max):
        return (value - min) / (max - min)
    data_t = data.transpose()

    max_x = np.amax(data_t[0])
    min_x = np.amin(data_t[0])
    max_y = np.amax(data_t[1])
    min_y = np.amin(data_t[1])

    normalized_x = list(map(lambda x: normalize_value(x, min_x, max_x), data_t[0]))
    normalized_y = list(map(lambda y: normalize_value(y, min_y, max_y), data_t[1]))

    result = np.array([normalized_x, normalized_y]).astype(np.float).transpose()
    return result


def calculate_means(data, old_means, k):
    def classify_value(point):
        # print(point)
        x = point[0]
        y = point[1]
        distances = []
        for mean in old_means:
            d = np.sqrt(np.power(x - mean[0], 2) + np.power(y - mean[1], 2))
            distances.append(d)
        index_min = np.argmin(np.array(distances))
        return index_min

    def get_new_mean(new_means, classification_counters):
        result = []
        for sample in enumerate(new_means):
            count = classification_counters[sample[0]]
            if count != 0:
                average = sample[1] / classification_counters[sample[0]]
            else:
                average = sample[1]
            result.append(average)
        return np.array(result)


    classified = []
    new_means = np.zeros((k, 2))
    classification_counters = np.zeros((k, 1))
    for sample in data:
        classification = classify_value(sample)
        classification_counters[classification] += 1
        new_means[classification][0] += sample[0]
        new_means[classification][1] += sample[1]

        classified.append(np.array([sample[0], sample[1], classify_value(sample)]))
    classified = np.array(classified).reshape((len(data), len(data[0]) + 1))
    new_means = get_new_mean(new_means, classification_counters)

    return (new_means, classified)

def main_loop(normal_data, max_counter=100, k=3):
    old_means = np.random.uniform(size=(k, 2))
    difference = 1
    counter = 0
    while difference != 0.0 and counter < max_counter:
        (means, data) = calculate_means(normal_data, old_means, k=k)
        # print(means)
        difference = np.linalg.norm(old_means - means)
        counter += 1
        old_means = means
        # print(counter)
        print("Counter: {}, difference: {}".format(counter, (difference)))

    return (data, means)

def create_plot_data(classifications, k):

    plot_data = [[] for _ in range(k)]
    for sample in classifications:
        [x, y, r] = sample
        plot_data[int(r)].append([x, y])

    result = list(map(lambda f: np.array(f), plot_data))
    return result

def create_styles(k, separation):
    styles = []
    for i in range(0, k):
        possible = np.random.rand(3, )
        match = True
        while(match):
            match = False
            for style in styles:
                d = np.linalg.norm(possible - style)
                print(d)
                if np.linalg.norm(possible - style) < separation:
                    possible = np.random.rand(3, )
                    match = True
                    continue
        styles.append(possible)

    return styles



def main():
    raw = load_data(line_count=7500)
    data = transform_to_vectors(raw)
    normal_data = normalize(data)
    k = 50

    # plt.scatter(normal_data.transpose()[0], normal_data.transpose()[1])
    # plt.show()
    classified, means = main_loop(normal_data, max_counter=30, k=k)
    # print(classified)
    plot_data = create_plot_data(classified, k)
    styles = create_styles(k, 0.2)
    for cluster_index in range(0, k):
        cluster = plot_data[cluster_index]
        styles.append(np.random.rand(3,))
        if len(cluster) > 0:
            xs = cluster.transpose()[0]
            ys = cluster.transpose()[1]
            plt.scatter(xs, ys, c=styles[cluster_index], alpha=0.4, marker='.')
            plt.scatter(means[cluster_index][0], means[cluster_index][1], c=styles[cluster_index], marker='s')

    plt.show()

if __name__ == '__main__':
    main()
