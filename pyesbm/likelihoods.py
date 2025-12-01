"""
likelihoods classes
"""

class BaseLikelihood:
    def __init__(
        self, bipartite=False, prior_a=1, prior_b=1, eps=1e-10, degree_correction=0
    ):
        self.bipartite = bipartite
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.eps = eps
        self.degree_correction = degree_correction

    def compute_likelihood(self, Y, clusters_1, clusters_2):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update_steps(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _type_check(self):
        if not isinstance(self.bipartite, bool):
            raise TypeError(
                f"bipartite must be a boolean. You provided {type(self.bipartite)}"
            )
        if not isinstance(self.prior_a, (int, float)):
            raise TypeError(
                f"prior_a must be int or float. You provided {type(self.prior_a)}"
            )
        if not isinstance(self.prior_b, (int, float)):
            raise TypeError(
                f"prior_b must be int or float. You provided {type(self.prior_b)}"
            )
        if not isinstance(self.eps, float):
            raise TypeError(f"eps must be float. You provided {type(self.eps)}")
        if not isinstance(self.degree_correction, int):
            raise TypeError(
                f"degree_correction must be int. You provided {type(self.degree_correction)}"
            )


class BetaBernoulli(BaseLikelihood):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()

        args = {k: v for k, v in locals().items() if k != "self"}
        self._type_check(**args)

        self.alpha = alpha
        self.beta = beta

    def _type_check(self, **kwargs):
        alpha = kwargs.get("alpha")
        beta = kwargs.get("beta")
        if not isinstance(alpha, (int, float)):
            raise TypeError(
                f"alpha must be int or float. You provided {type(alpha)}"
            )
        if alpha <= 0:
            raise ValueError(f"alpha must be positive. You provided {alpha}")
        if not isinstance(beta, (int, float)):
            raise TypeError(
                f"beta must be int or float. You provided {type(beta)}"
            )
        if beta <= 0:
            raise ValueError(f"beta must be positive. You provided {beta}")

    def compute_likelihood(self, Y, clusters_1, clusters_2):
        # Implement the Bernoulli likelihood computation
        pass

    def update_logits(self):
        # Implement the steps to update the Bernoulli likelihood
        pass

class PoissonGamma:
    def __init__(self, shape=1.0, rate=1.0):
        super().__init__()

        args = {k: v for k, v in locals().items() if k != "self"}
        self._type_check(**args)

        self.shape = shape
        self.rate = rate

    def _type_check(self, **kwargs):
        shape = kwargs.get("shape")
        rate = kwargs.get("rate")
        if not isinstance(shape, (int, float)):
            raise TypeError(
                f"shape must be int or float. You provided {type(shape)}"
            )
        if shape <= 0:
            raise ValueError(f"shape must be positive. You provided {shape}")
        if not isinstance(rate, (int, float)):
            raise TypeError(
                f"rate must be int or float. You provided {type(rate)}"
            )
        if rate <= 0:
            raise ValueError(f"rate must be positive. You provided {rate}")

    def compute_likelihood(self, Y, clusters_1, clusters_2):
        # Implement the Poisson likelihood computation
        pass

    def update_logits(self):
        # Implement the steps to update the Poisson likelihood
        pass
