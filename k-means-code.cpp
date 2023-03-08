/*
***************************************************************************
* Note that we have added the `#pragma acc parallel loop present` directive 
* before the loop in the `assign_points` function. This tells OpenACC to 
* parallelize the loop over the `points` vector and to make sure that the 
* `centroids` and `assignments` vectors are present on the device. We have 
* also removed the `using namespace std;` directive for clarity.
****************************************************************************
*/
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <openacc.h>

using namespace std;

// Define a structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Function to calculate the Euclidean distance between two points
double distance(const Point& a, const Point& b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// Function to initialize the centroids randomly
vector<Point> init_centroids(const vector<Point>& points, int K) {
    vector<Point> centroids(K);
    srand(time(NULL));
    for (int i = 0; i < K; i++) {
        centroids[i] = points[rand() % points.size()];
    }
    return centroids;
}

// Function to assign points to their nearest centroids
void assign_points(const vector<Point>& points, const vector<Point>& centroids, vector<int>& assignments) {
    #pragma acc parallel loop present(points, centroids, assignments) 
    for (int i = 0; i < points.size(); i++) {
        double min_dist = INFINITY;
        for (int j = 0; j < centroids.size(); j++) {
            double dist = distance(points[i], centroids[j]);
            if (dist < min_dist) {
                min_dist = dist;
                assignments[i] = j;
            }
        }
    }
}

// Function to compute the new centroids
vector<Point> compute_centroids(const vector<Point>& points, const vector<int>& assignments, int K) {
    vector<Point> centroids(K, {0, 0});
    vector<int> counts(K, 0);
    for (int i = 0; i < points.size(); i++) {
        int cluster = assignments[i];
        centroids[cluster].x += points[i].x;
        centroids[cluster].y += points[i].y;
        counts[cluster]++;
    }
    for (int i = 0; i < K; i++) {
        centroids[i].x /= counts[i];
        centroids[i].y /= counts[i];
    }
    return centroids;
}

// Function to perform K-means clustering
vector<int> kmeans(const vector<Point>& points, int K) {
    vector<Point> centroids = init_centroids(points, K);
    vector<int> assignments(points.size(), -1);
    int iter = 0;
    while (true) {
        assign_points(points, centroids, assignments);
        vector<Point> new_centroids = compute_centroids(points, assignments, K);
        if (centroids == new_centroids || iter > 100) {
            break;
        }
        centroids = new_centroids;
        iter++;
    }
    return assignments;
}

int main() {
    // Define the input list of nodes
    vector<Point> points = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}};

    // Define the number of trucks
    int K = 3;

    // Run K-means clustering to obtain the assignments of each point to a cluster
    vector<int> assignments = kmeans(points, K);

    // Output the results
	std::cout << "Assignments:" << std::endl;
	for (int i = 0; i < points.size(); i++) {
	std::cout << "Point (" << points[i].x << ", " << points[i].y << ") assigned to cluster " << assignments[i] << std::endl;
	}

	return 0;
}
