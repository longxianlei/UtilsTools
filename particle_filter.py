import numpy as np
from numpy.random import uniform, randn
import matplotlib.pyplot as plt
from scipy import stats


def create_uniform_particles(x_range, y_range, heading_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(heading_range[0], heading_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles


def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles


def predict(particles, u, std, dt=1.):
    """
    Moving according to the control input u=[heading change, velocity].
    With noise Q (std heading change, std velocity change)
    :param particles:
    :param u:u[heading, velocity].
    :param std:
    :param dt: delta time.
    :return:
    """
    N = len(particles)

    # predict heading with normal distribution.
    particles[:, 2] = u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # update x, y axis.
    dist = u[1] * dt + (randn(N) * std[1])
    particles[:, 0] += dist * np.cos(particles[:, 2])
    particles[:, 1] += dist * np.sin(particles[:, 2])


def update(particles, weights, z, R, landmarks):
    """
    Update the weights according to the measurements.
    P(x/z)=P(z/x)*P(x). P(x) prior of each particles' position, P(z/x) likelihood, how well the particles matched the
    measurements. So, using the pdf to measure the matched confidence.
    :param particles:
    :param weights:
    :param z:
    :param R:
    :param landmarks:
    :return:
    """
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= stats.norm(distance, R).pdf(z[i])
    weights += 1e-300
    weights /= sum(weights)


def estimate(particles, weights):
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean, var


# If we don't have new measurements, then, resampling.
def neff(weights):
    return 1. / np.sum(np.square(weights))


def systematic_resampling(weights):
    N = len(weights)
    # Partition the weights into N subdivisions with a constant value/offset.
    position = (np.arange(N) + np.random.random()) / N
    indexes = np.zeros(N, 'i')  # int
    cumulative = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if position[i] < cumulative[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def resample_from_indexes(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill(1.0 / len(weights))


def run_particle_filter(N, iters=18, sensor_std_err=0.1, do_plot=True, plot_particles=False,
                        xlim=(0, 20), ylim=(0, 20), initial_x=None):
    landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
    num_landmarks = len(landmarks)
    plt.figure()

    # create particles and weights.
    if initial_x is not None:
        particles = create_gaussian_particles(mean=initial_x, std=(5, 5, np.pi / 4), N=N)
    else:
        particles = create_uniform_particles(xlim, ylim, (0, 2 * np.pi), N)

    weights = np.ones(N) / N
    if plot_particles:
        alpha = 0.20
        if N > 5000:
            alpha *= np.sqrt(5000) / np.sqrt(N)
        plt.scatter(particles[:, 0], particles[:, 1],
                    alpha=alpha, color='g')

    xs = []
    robot_pos = np.array([0.0, 0.0])
    for x in range(iters):
        robot_pos += (1., 1.)

        # distance from the robot to the each landmarks.
        # this could be the measurement of the sensors to the robot.
        zs = (np.linalg.norm(landmarks - robot_pos, axis=1) + (np.random.randn(num_landmarks) * sensor_std_err))

        # move diagonally forward to (x+1, y+1), so, heading is pi/4, radius=1.414
        # predict the particles position according to the given direction and move.
        predict(particles, u=(0.78539, 1.414), std=(0.5, 0.5))
        # weights.fill(1.0/len(weights))
        # incorporate measurements, update the weights.
        update(particles, weights, zs, R=sensor_std_err, landmarks=landmarks)

        # resample if too few effective particles.
        if neff(weights) < N / 2:
            indexs = systematic_resampling(weights)
            resample_from_indexes(particles, weights, indexs)
            assert np.allclose(weights, 1 / N)
        mu, var = estimate(particles, weights)
        xs.append(mu)

        if plot_particles:
            plt.scatter(particles[:, 0], particles[:, 1], color='k', marker=',', s=1)
        p1 = plt.scatter(robot_pos[0], robot_pos[1], color='k', marker='+', s=180, lw=3)
        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')

    xs = np.array(xs)
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    print('final posisiton error, variance: \n\t', mu - np.array([iters, iters]), var)
    plt.show()


if __name__ == '__main__':
    from numpy.random import seed

    # particles2 = np.array([0.1, 0.2, 0.2, 0.03, 0.1, 0.37])
    # weights2 = np.array([0.1, 0.2, 0.2, 0.03, 0.1, 0.37])
    # indexes2 = systematic_resampling(weights2)
    # print(indexes2)
    # resample_from_indexes(particles2, weights2, indexes2)
    #
    # particles1 = np.array([0.1, 0.2, 0.2, 0.03, 0.1, 0.37])
    # weights1 = np.array([0.1, 0.2, 0.2, 0.03, 0.1, 0.37])

    seed(4)
    run_particle_filter(N=5000, plot_particles=False)
