"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, DotProduct, WhiteKernel, Sum, ConstantKernel as C
import matplotlib.pyplot as plt

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
DELTA = 0.999


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        self.sigma_f = 0.15
        self.sigma_v = 0.0001

        self.gp_f = GaussianProcessRegressor(kernel=ConstantKernel(constant_value=0.5) * RBF(length_scale=0.5), alpha=self.sigma_f**2)
        self.gp_v = GaussianProcessRegressor(kernel=DotProduct() + Matern(nu=2.5), alpha=self.sigma_v**2)
        
        self.X = np.array([]).reshape(-1, 1)
        self.f = np.empty((0, 1))
        self.v = np.empty((0, 1))

        self.lamb = 1000
        self.beta = 0.3
        # not needed for ucb:
        # self.xi = 0.02

        self.iteration = 0

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        return np.atleast_2d(self.optimize_acquisition_function())

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        
        mean_f, std_f = self.gp_f.predict(x, return_std=True)
        mean_v, std_v = self.gp_v.predict(x, return_std=True)
        
        # # UCB acquisition function
        # update beta according to the current iteration
        self.beta = np.sqrt(2 * np.log(self.iteration / DELTA))

        # UCB acquisition function
        af_value = mean_f + self.beta * std_f

        # Thompson sampling acquisition function
        # af_value = np.random.normal(mean_f, std_f)

        # Expected improvement acquisition function
        # Best observed value so far
        # f_best = np.max(self.gp_f.y_train_)
        # Z = (mean_f - f_best - self.xi) / std_f
        # af_value = (mean_f - f_best - self.xi) * norm.cdf(Z) + std_f * norm.pdf(Z)

        if mean_v + 3*std_v >= SAFETY_THRESHOLD:
            af_value = af_value - self.lamb * (mean_v + 3*std_v - SAFETY_THRESHOLD)

        return af_value

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        self.X = np.vstack([self.X, x])
        self.f = np.append(self.f, f)
        self.v = np.append(self.v, v)

        self.gp_f.fit(self.X, self.f)
        self.gp_v.fit(self.X, self.v)

        self.iteration += 1

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        idx_valid = np.where(self.v < SAFETY_THRESHOLD)
        x_valid = self.X[idx_valid]
        f_valid = self.f[idx_valid]
        x_opt = x_valid[np.argmax(f_valid)]
        return x_opt

    def plot(self, plot_recommendation: bool = True):
        """Plot objective, constraint posterior and acquisition function for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])

def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)

def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    unsafe_evals = 0
    # Check for valid shape
    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        if cost_val > SAFETY_THRESHOLD:
            unsafe_evals += 1

        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals {unsafe_evals}\n')
    


if __name__ == "__main__":
    main()
