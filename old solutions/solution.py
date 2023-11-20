"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, DotProduct, WhiteKernel, Sum, ConstantKernel as C


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
DELTA = 0.999


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.sigma_f = 0.15
        self.sigma_v = 0.0001

        #self.f_kernel = Matern(nu=2.5, length_scale=1.0)
        #self.v_kernel = DotProduct(sigma_0=0.0)+Matern(nu=2.5, length_scale=1.0)
        #self.f_kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.5)
        #self.v_kernel = DotProduct(sigma_0=0.0) + RBF(length_scale=1.0) + WhiteKernel(noise_level=np.sqrt(2.0))

        # self.f_kernel = C() * Matern(length_scale=1.0, nu=2.5)

        # self.v_kernel = Sum(C() * RBF(length_scale=5.0), C() * Matern(length_scale=1.0, nu=2.5))
        # self.v_kernel.k1.constant_value = np.sqrt(2)  # RBF variance
        # self.v_kernel.k2.constant_value = 4.0  # Prior mean

        self.gp_f = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=self.sigma_f**2)
        self.gp_v = GaussianProcessRegressor(kernel=DotProduct() + Matern(nu=2.5), alpha=self.sigma_v**2)
        #self.gp_v.mean = 4.0

        self.X = np.empty((0, 1))
        self.f = np.empty((0, 1))
        self.v = np.empty((0, 1))

        # might me tuned
        self.lamb = 10000000.0
        self.beta = 0.01

        self.iteration = 0

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        recommendation = np.atleast_2d(self.optimize_acquisition_function())
        return recommendation

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
        x = np.atleast_2d(x).transpose()
        # TODO: Implement the acquisition function you want to optimize.
        mean_f, std_f = self.gp_f.predict(x, return_std=True)
        # mean_v, std_v = self.gp_v.predict(x, return_std=True)
        # mean = mean_f - self.lamb * max(0.0, mean_v)
        # std = np.sqrt(std_f**2 + self.lamb**2 * std_v**2)
        # # UCB acquisition function
        # af_value = mean + self.beta * std
        # self.beta = np.sqrt(2 * np.log(self.iteration / DELTA))

        mean = mean_f + self.beta * std_f
        af_value = mean - self.lamb * np.maximum(self.gp_v.predict(x), SAFETY_THRESHOLD)

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
        # TODO: Add the observed data {x, f, v} to your model.
        self.X = np.append(self.X, x)
        self.f = np.append(self.f, f)
        self.v = np.append(self.v, v)

        x_fit = np.atleast_2d(self.X).transpose()

        self.gp_f.fit(x_fit, self.f)
        self.gp_v.fit(x_fit, self.v)

        self.iteration += 1

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        idx_valid = np.where(self.v < SAFETY_THRESHOLD)
        x_valid = self.X[idx_valid]
        f_valid = self.f[idx_valid]
        x_opt = x_valid[np.argmax(f_valid)]
        return x_opt

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


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
