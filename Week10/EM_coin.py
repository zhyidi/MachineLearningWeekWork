import numpy as np


class EM_coin:
    def getParameter(self, pai, p, q, observations):
        pai_t = 0
        p_t = 0
        q_t = 0
        length = len(observations)
        u = [0 for _ in range(length)]
        while pai_t != pai or p_t != p or q_t != q:
            pai_t = pai
            p_t = p
            q_t = q
            for i in range(length):
                u[i] = (
                    pai * p ** observations[i] * (1 - p) ** (1 - observations[i])
                ) / (
                    pai * p ** observations[i] * (1 - p) ** (1 - observations[i])
                    + (1 - pai)
                    * q ** observations[i]
                    * (1 - q) ** (1 - observations[i])
                )
            pai = sum(u) / length
            p = sum(np.multiply(u, observations[:])) / sum(u)
            q = sum(np.multiply([-i + 1 for i in u], observations[:])) / sum(
                [-i + 1 for i in u]
            )
        result = [pai, p, q]
        return result


if __name__ == "__main__":
    emcoin = EM_coin()
   #  print(emcoin.getParameter(0.5, 0.5, 0.5, [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]))
   #  print(emcoin.getParameter(0.4, 0.6, 0.7, [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]))
    print(emcoin.getParameter(0.46, 0.55, 0.67, [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]))
