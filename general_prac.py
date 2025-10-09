# def alpha(x_list, y_list):
#     leng = min(len(x_list), len(y_list))
#     x_mean = sum(x_list) / leng
#     y_mean = sum(y_list) / leng
#     print(f"x_mean: {x_mean}, y_mean: {y_mean}")
#     nom = 0
#     denom = 0
#     for i in range(leng):
#         x = x_list[i]
#         y = y_list[i]
#         # nom = sigma(xi - x_mean)(yi - y_mean)
#         nom += 1.0 * (x - x_mean) * (y - y_mean)
#         # denom = sigma(xi - x_mean)^2
#         denom += 1.0 * (x - x_mean) ** 2
#     return nom / denom

# def beta(x_list, y_list, a):
#     leng = min(len(x_list), len(y_list))
#     x_mean = sum(x_list) / leng
#     y_mean = sum(y_list) / leng
#     return y_mean - a * x_mean

# def predict(x, a, b):
#     return a * x + b

# if __name__ == "__main__":
#     x_list = [30, 28, 32, 25, 25, 28, 22, 35, 40, 24]
#     y_list = [2.5, 3.0, 2.7, 4.0, 4.2, 4.1, 5.0, 3.0, 4.0, 4.5]
#     a = alpha(x_list, y_list)
#     b = beta(x_list, y_list, a)
#     print(f"y = {a} * x + {b}")
#     x = 52
#     print(f'the cars produced is {x} Thousands')
#     y = predict(x, a, b)
#     print(f'the predicted average cost for a car is {y} millions')

# given 2 lists, alpha_list for amplitudes and f_list for frequencies, plot the signal in frequency domain
# if __name__ == "__main__":
#     import numpy as np
#     import matplotlib.pyplot as plt
#     # a1=1, f1=3; a2=2, f2=6; a3=4, f3=12; a4=3, f4=15; a5=6, f5=24; a6=4, f6=30; a7=1, f7=33;
#     alpha_list = [1, 2, 4, 3, 6, 4, 1]
#     f_list = [3, 6, 12, 15, 24, 30, 33]
#     threshold = 18
#     def filter(alpha_list, f_list, threshold):
#         filtered_alpha = []
#         filtered_f = []
#         for alpha, f in zip(alpha_list, f_list):
#             if f <= threshold:
#                 filtered_alpha.append(alpha)
#                 filtered_f.append(f)
#             else:
#                 filtered_alpha.append(0)
#                 filtered_f.append(f)
#         return filtered_alpha, filtered_f
#     def bandpass(alpha_list, f_list, low, high, threshold):
#         filtered_alpha = []
#         filtered_f = []
#         for alpha, f in zip(alpha_list, f_list):
#             if low <= f <= high:
#                 filtered_alpha.append(alpha)
#                 filtered_f.append(f)
#             elif f < low:
#                 filtered_alpha.append(0.5 * alpha)
#                 filtered_f.append(f)
#             elif high < f < threshold:
#                 filtered_alpha.append(0.5 * alpha)
#                 filtered_f.append(f)
#             else:
#                 filtered_alpha.append(0)
#                 filtered_f.append(f)
#         return filtered_alpha, filtered_f
#     # alpha_list, f_list = filter(alpha_list, f_list, threshold=threshold)
#     threshold = 15
#     alpha_list, f_list = bandpass(alpha_list, f_list, low=6, high=9, threshold=threshold)
#     # t = np.linspace(0, 1, max(f_list) + 1)  # time vector
#     # signal = np.zeros_like(t)
#     # for alpha, f in zip(alpha_list, f_list):
#     #     signal += alpha * np.sin(2 * np.pi * f * t)

#     plt.figure(figsize=(10, 4))
#     # plt.plot(t, signal)
#     plt.title('Signal in Frequency Domain')
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Amplitude')
#     # set the x_axis range as [0, max(f_list) + 5]
#     # label all freq in f_list on x axis
#     plt.xticks(f_list+[threshold])
#     plt.yticks(range(0, max(alpha_list) + 2))
#     plt.stem(f_list, alpha_list, use_line_collection=True)
    

#     plt.savefig('frequency_spectrum.png')

# KNN algorithm
# import numpy as np

# def cosine_distance(a, b):
#     a = np.array(a)
#     b = np.array(b)
#     dot_product = np.dot(a, b)
#     norm_a = np.linalg.norm(a)
#     norm_b = np.linalg.norm(b)
#     if norm_a == 0 or norm_b == 0:
#         return 1.0
#     return 1 - dot_product / (norm_a * norm_b)

# def Manhattan_distance(a, b):
#     a = np.array(a)
#     b = np.array(b)
#     return np.sum(np.abs(a - b))

# def euclidean_distance(a, b):
#     a = np.array(a)
#     b = np.array(b)
#     return np.linalg.norm(a - b)

# def KNN(K, points, distance_metric, max_iters=10):
#     # first step select K random points as initial centroids
#     centroids = [points[i][0] for i in range(K)]
#     for iter in range(max_iters):
#         clusters = [[] for _ in range(K)]
#         for point, label in points:
#             distances = [distance_metric(point, centroid) for centroid in centroids]
#             min_index = np.argmin(distances)
#             clusters[min_index].append((point, label))
#         new_centroids = []
#         for cluster in clusters:
#             if len(cluster) == 0:
#                 new_centroids.append(np.random.rand(len(points[0][0])).tolist())
#                 continue
#             cluster_points = np.array([p[0] for p in cluster])
#             new_centroid = np.mean(cluster_points, axis=0).tolist()
#             new_centroids.append(new_centroid)
#         if euclidean_distance(np.array(centroids), np.array(new_centroids)) < 1e-2:
#             break
#         centroids = new_centroids
#     return clusters, centroids



# def predict(point, centroids, distance_metric, label_dict = {0: '0', 1: '1', 2: '2'}):
#     distances = [distance_metric(point, centroid) for centroid in centroids]
#     min_index = np.argmin(distances)
#     return label_dict.get(min_index, 'Unknown')
    


# if __name__ == "__main__":
#     points = []
#     points.append(([1.0, 2.0, 3.0], 'A'))
#     points.append(([0.5, 1.8, 2.7], 'B'))
#     points.append(([1.2, 2.2, 3.5], 'B'))
#     points.append(([4.6, 5.6, 3.7], 'A'))
#     points.append(([2.4, 4.6, 3.6], 'A'))
#     points.append(([3.5, 2, 4.1], 'B'))
#     points.append(([3.6, 4.6, 7.1], 'A'))
#     points.append(([6.2, 4.1, 1.3], 'B'))
#     points.append(([8.4, 3.5, 1.8], 'A'))
#     points.append(([5.8, 3.4, 2.7], 'B'))
#     centroids = []
#     tmp_dict = {}
#     label_dict = {}
#     for point, label in points:
#         if label not in tmp_dict:
#             tmp_dict[label] = []
#         tmp_dict[label].append(point)
#     for label, pts in tmp_dict.items():
#         centroid = np.mean(np.array(pts), axis=0).tolist()
#         label_dict[len(centroids)] = label
#         centroids.append((centroid))
#     print('For K = 2, predict point [3.5, 4.0, 6.0] using cosine distance: ', predict([3.5, 4.0, 6.0], centroids, cosine_distance, label_dict))

#     print('For K = 2, predict point [3.5, 4.0, 6.0] using Manhattan distance: ', predict([3.5, 4.0, 6.0], centroids, Manhattan_distance, label_dict))


#     x = ([3.5, 4.0, 6.0], 'I')
#     K = 3
#     points.append(x)
    

#     clusters, centroids = KNN(K, points, distance_metric=cosine_distance, max_iters=100)
#     print(
#         'Using cosine distance metric to cluster the points into 3 clusters:'
#     )
#     for i, cluster in enumerate(clusters):
#         print(f"Cluster {i}:")
#         for point, label in cluster:
#             print(f"  Point: {point}, Label: {label}")
#         print(f" Centroid: {centroids[i]}")
#     print('predicting point [3.5, 4.0, 6.0]: ', predict([3.5, 4.0, 6.0], centroids, cosine_distance))
#     clusters, centroids = KNN(K, points, distance_metric=Manhattan_distance, max_iters=100)
#     print(
#         'Using Manhattan distance metric to cluster the points into 3 clusters:'
#     )
#     for i, cluster in enumerate(clusters):
#         print(f"Cluster {i}:")
#         for point, label in cluster:
#             print(f"  Point: {point}, Label: {label}")
#         print(f" Centroid: {centroids[i]}")
#     print('predicting point [3.5, 4.0, 6.0]: ', predict([3.5, 4.0, 6.0], centroids, Manhattan_distance))
# import math
# import numpy as np
# ratio_list = [2.0/5.0, 3.0/5.0]
# # ratio_list = [0.5, 0.5]
# result = 0
# for ratio in ratio_list:
#     result += -1.0 * ratio * math.log2(ratio)
# print(result)
# ans = 5.0/9.0 * result
# print(ans)


import numpy as np

def euclidean_distance(a, b, print_flag = True):
    leng = min(len(a), len(b))
    if print_flag:
        print(f"Distance between point {a} and centroid {b}:")
        formular_str = "sqrt( "
        for i in range(leng):
            formular_str += f"({a[i]} - {b[i]})^2"
            if i != leng - 1:
                formular_str += " + "
        formular_str += " )"
    result = np.linalg.norm(np.array(a) - np.array(b))
    if print_flag:
        formular_str += f" = {result}"
        print(formular_str)
    return result

def KNN(K, points, distance_metric, max_iters=1):
    # first step select K random points as initial centroids
    centroids = [points[i][0] for i in range(K)]
    print(f'Take the first {K} points as initial centroids: {centroids}')
    for iter in range(max_iters):
        clusters = [[] for _ in range(K)]
        for point, label in points:
            print(f"calculate distances from point {point} to 2 centroids:")
            distances = [distance_metric(point, centroid) for centroid in centroids]
            min_index = np.argmin(distances)
            print(f"The point is assigned to the cluster {min_index + 1} with centroid {centroids[min_index]}\n")
            clusters[min_index].append((point, label))
        new_centroids = []
        print(f'After the first iteration, recalculate the centroids:')
        for idx, cluster in enumerate(clusters):
            if len(cluster) == 0:
                new_centroids.append(np.random.rand(len(points[0][0])).tolist())
                continue
            cluster_points = np.array([p[0] for p in cluster])
            print(f" Cluster {idx + 1} points: {cluster_points.tolist()}")
            print(f" New centroid of cluster {idx + 1} is the mean of these points:")
            tmp_cluster_points = cluster_points.tolist()
            formular_str = f"1/{len(tmp_cluster_points)} * ( "
            for i, p in enumerate(tmp_cluster_points):
                formular_str += f"{p}"
                if i != len(tmp_cluster_points) - 1:
                    formular_str += " + "
            formular_str += " )"
            
            new_centroid = np.mean(cluster_points, axis=0).tolist()
            formular_str += f" = {new_centroid}"
            print(formular_str)
            new_centroids.append(new_centroid)
        if euclidean_distance(np.array(centroids), np.array(new_centroids), print_flag=False) < 1e-2:
            break
        centroids = new_centroids
    return clusters, centroids


if __name__ == "__main__":
    points = []
    points.append(([12.0, 45.0, 78.0], ''))
    points.append(([14.0, 40.0, 81.0], 'B'))
    points.append(([11.0, 43.0, 77.0], 'B'))
    points.append(([14.0, 47.0, 79.0], 'A'))
    points.append(([15.0, 41.0, 83.0], 'A'))
    points.append(([11.0, 44.0, 75.0], 'B'))


    K = 2

    clusters, centroids = KNN(K, points, distance_metric=euclidean_distance, max_iters=1)
    print("So the answer is:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}:")
        print(f" Centroid: {centroids[i]}")
        for point, label in cluster:
            print(f"  Point: {point}")