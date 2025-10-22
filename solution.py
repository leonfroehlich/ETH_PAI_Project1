import os
import typing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import norm
from scipy.optimize import bisect


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


class Model(object):
    def __init__(self):
        self.rng = np.random.default_rng(seed=0)
        self.max_train_points = 5000  # Set to None to use all training points

        kernel = Matern(length_scale=0.1, length_scale_bounds=(1e-2, 1e2), nu=1.5)* ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

        self.gpr = GaussianProcessRegressor(
            kernel = kernel,
            random_state = 0,
            normalize_y = True,
            copy_X_train = False,
        )

    def d_asym_cost_loss(self, x, mu, sigma):
        """
        Derivative of the asymmetric cost loss function.
        :param x: point at which to evaluate the derivative
        :param mu: GP posterior mean
        :param sigma: GP posterior stddev
        """
        z = (x - mu) / sigma # normalized 
        # This is derived from the asymetric cost function. Must find root of following to minimize cost:
        return COST_W_UNDERPREDICT*z*norm.sf(z) + COST_W_NORMAL*z*norm.cdf(z) + (COST_W_NORMAL - COST_W_UNDERPREDICT)*norm.pdf(z)

    def subsample(self, X, y):
        n = X.shape[0]
        if self.max_train_points is None or n <= self.max_train_points:
            self._trained_on_subset_idx = np.arange(n, dtype=int)
            return X, y, self._trained_on_subset_idx

        # Cluster on spatial coords to keep geographic coverage, don't consider area flag
        z = X[:, 0:2]

        k = self.max_train_points
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=2048)
        labels = kmeans.fit_predict(z)

        # Pick the training point closest to each centroid
        centers = kmeans.cluster_centers_
        # For efficiency, map each cluster to nearest member
        chosen = np.empty(k, dtype=int)
        # Build index lists per cluster
        for c in range(k):
            idx = np.flatnonzero(labels == c)
            if idx.size == 0:
                # choose random point if no members assigned to this cluster
                chosen[c] = self.rng.integers(0, n)
                continue
            # nearest by Euclidean distance in z-space
            pts = z[idx]
            dif = pts - centers[c]
            i_local = np.argmin(np.sum(dif * dif, axis=1))
            chosen[c] = idx[i_local]

        chosen = np.unique(chosen)
        self._trained_on_subset_idx = chosen

        return X[chosen], y[chosen], chosen


    def predict_pollution_concentration(self, test_coordinates: np.ndarray, test_area_flags: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_coordinates: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_area_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        gp_mean, gp_std = self.gpr.predict(test_coordinates, return_std=True)

        predictions = gp_mean
        # Adjust predictions in residential area using asymmetric cost function
        for i in range(test_coordinates.shape[0]):
            if test_area_flags[i]:
                # Find root of asymmetric cost derivative
                predictions[i] = bisect(self.d_asym_cost_loss, gp_mean[i] - 10*gp_std[i], gp_mean[i] + 10*gp_std[i], args=(gp_mean[i], gp_std[i]))

        return predictions, gp_mean, gp_std


    def fit_model_on_training_data(self, train_targets: np.ndarray, train_coordinates: np.ndarray, train_area_flags: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_coordinates: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_targets: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        :param train_area_flags: Binary variable denoting whether the 2D training point is in the residential area (1) or not (0)
        """
        
        # Subsample
        print('Subsampling training data...')
        train_coordinates, train_targets, _ = self.subsample(train_coordinates, train_targets)
        print(f'Fitting on {train_coordinates.shape[0]} training points.')

        self.gpr.fit(train_coordinates, train_targets)


def calculate_cost(ground_truth: np.ndarray, predictions: np.ndarray, area_flags: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param area_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case: underprediction
    mask = (predictions < ground_truth) & [bool(area_flag) for area_flag in area_flags]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def check_within_circle(coordinate, circle_parameters):
    """
    Checks if a coordinate is inside a circle.
    :param coordinate: 2D coordinate
    :param circle_parameters: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coordinate[0] - circle_parameters[0])**2 + (coordinate[1] - circle_parameters[1])**2 < circle_parameters[2]**2


def identify_city_area_flags(grid_coordinates):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param grid_coordinates: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])
    
    area_flags = np.zeros((grid_coordinates.shape[0],))

    for i,coordinate in enumerate(grid_coordinates):
        area_flags[i] = any([check_within_circle(coordinate, circ) for circ in circles])

    return area_flags

def plot_data(y, coordinates, area_flags):
    """
    Plots the training data.
    :param y: Pollution concentrations
    :param coordinates: 2D coordinates
    :param area_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    """
    plt.figure()
    plt.title('Training data visualization')
    # Separate into two groups based on area_flags
    in_area_coords, in_area_y = coordinates[area_flags], y[area_flags]
    out_area_coords, out_area_y = coordinates[~area_flags], y[~area_flags]

    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])

    scatter = plt.scatter(in_area_coords[:, 0], in_area_coords[:, 1], c=in_area_y, cmap=cm.viridis, marker='o', label='In City Area', edgecolor='k')
    scatter = plt.scatter(out_area_coords[:, 0], out_area_coords[:, 1], c=out_area_y, cmap=cm.viridis, marker='s', label='Out of City Area')
    # Plot city centers
    # for circle in circles:
    #     circle_patch = plt.Circle((circle[0], circle[1]), circle[2], color='red')
    #     plt.gca().add_patch(circle_patch)

    plt.colorbar(scatter, label='Pollution Concentration')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()


def perform_extended_model_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_grid = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    grid_area_flags = identify_city_area_flags(visualization_grid)
    
    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict_pollution_concentration(visualization_grid, grid_area_flags)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def get_city_area_data(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_coordinates = np.zeros((train_x.shape[0], 2), dtype=float)
    train_area_flags = np.zeros((train_x.shape[0],), dtype=bool)
    test_coordinates = np.zeros((test_x.shape[0], 2), dtype=float)
    test_area_flags = np.zeros((test_x.shape[0],), dtype=bool)

    train_coordinates[:] = train_x[:, 0:2]
    train_area_flags[:] = train_x[:, 2].astype(bool, copy=False)
    test_coordinates[:] = test_x[:, 0:2]
    test_area_flags[:] = test_x[:, 2].astype(bool, copy=False)

    assert train_coordinates.shape[0] == train_area_flags.shape[0] and test_coordinates.shape[0] == test_area_flags.shape[0]
    assert train_coordinates.shape[1] == 2 and test_coordinates.shape[1] == 2
    assert train_area_flags.ndim == 1 and test_area_flags.ndim == 1

    return train_coordinates, train_area_flags, test_coordinates, test_area_flags


def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_coordinates, train_area_flags, test_coordinates, test_area_flags = get_city_area_data(train_x, test_x)
    
    # Fit the model
    print('Training model')
    model = Model()
    model.fit_model_on_training_data(train_y, train_coordinates, train_area_flags)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.predict_pollution_concentration(test_coordinates, test_area_flags)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_model_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
