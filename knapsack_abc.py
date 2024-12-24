import numpy as np

def obj(X, W, P, C):
    """
    Objective function for the 0-1 knapsack problem.
    
    Parameters:
        X (array): Binary solution vector.
        W (array): Weights of items.
        P (array): Profits of items.
        C (float): Capacity constraint.

    Returns:
        float: Penalized objective value.
    """
    f = -np.sum(P * X)  # Maximize profit (minimize negative profit)
    g = np.sum(W * X) - C  # Constraint violation
    r = 10**3  # Penalty coefficient
    phi = f + r * max(0, g)**2  # Penalized objective
    return phi

def optimize(w, p, c, n=10, d=5, T=100):
    """
    Artificial Bee Colony algorithm for the 0-1 knapsack problem.

    Parameters:
        w (array): Weights of items.
        p (array): Profits of items.
        c (float): Capacity constraint.
        n (int): Number of solutions (default: 10).
        d (int): Problem dimension (default: 5).
        T (int): Number of iterations (default: 100).

    Returns:
        tuple: Best solution and its value.
    """
    np.random.seed(0)
    fi = np.random.choice([0, 1], size=(n, d))
    fii = np.random.choice([0, 1], size=(n, d))
    x = np.random.choice([0, 1], size=(n, d))
    fit = np.empty(n)

    # Initialize fitness values
    for i in range(n):
        fit[i] = obj(x[i], w, p, c)

    zeros = np.zeros(n)
    lim = 10
    f = np.zeros(T)

    enIyiCozum = x[np.argmin(fit)].copy()
    enIyiKar = min(fit)
    uygunluk = np.empty(n)
    v = np.empty((n, d))

    for t in range(T):  
        for i in range(n):
            k = np.random.randint(0, n)
            while i == k:
                k = np.random.randint(0, n)

            # Generate new candidate solution
            kısıt1 = np.logical_or(x[i], x[k]).astype(int)
            kısıt2 = np.logical_and(fi[i], kısıt1).astype(int)
            v[i] = np.logical_and(fii[i], kısıt2).astype(int)

            fit_val = obj(v[i], w, p, c)
            if fit_val < fit[i]:
                fit[i] = fit_val
                x[i] = v[i].copy()
                zeros[i] = 0
            else:
                zeros[i] += 1

            if fit_val < enIyiKar:
                enIyiKar = fit_val
                enIyiCozum = v[i].copy()

        # Calculate fitness probabilities
        for i in range(n):
            if fit[i] < 0:
                uygunluk[i] = 1 + abs(fit[i])
            else:
                uygunluk[i] = 1 / (1 + fit[i])

        p_vals = uygunluk / np.sum(uygunluk)

        # Generate new solutions based on probabilities
        for ii in range(n):
            i = np.random.choice(n, size=1, replace=True, p=p_vals)[0]
            k = np.random.randint(0, n)
            while i == k:
                k = np.random.randint(0, n)

            v = x.copy()
            v[k] = 1 - v[k]

            a_val = obj(v[i], w, p, c)
            if fit_val < fit[i]:
                fit[i] = fit_val
                x[i] = v[i].copy()
                zeros[i] = 0
            else:
                zeros[i] += 1

            if a_val < enIyiKar:
                enIyiKar = a_val
                enIyiCozum = v[i].copy()

        # Reinitialize stagnant solutions
        for i in range(n):
            if zeros[i] > lim:
                x[i] = np.random.choice([0, 1], size=d)
                fit[i] = obj(x[i], w, p, c)
                zeros[i] = 0  

        f[t] = enIyiKar

    return enIyiCozum, enIyiKar

if __name__ == "__main__":
    ornekler = [
        {"W": np.array([21, 33, 5, 7, 1]), "P": np.array([10, 20, 30, 40, 50]), "C": 10},
        {"W": np.array([15, 25, 35, 45, 5]), "P": np.array([5, 10, 15, 20, 25]), "C": 20},
        {"W": np.array([10, 20, 30, 40, 50]), "P": np.array([1, 2, 3, 4, 5]), "C": 15},
        {"W": np.array([5, 10, 15, 20, 25]), "P": np.array([50, 40, 30, 20, 10]), "C": 25},
        {"W": np.array([12, 24, 36, 48, 60]), "P": np.array([3, 6, 9, 12, 15]), "C": 30}
    ]

    data = np.empty([30, 5])
    solution = np.empty([5, 5])

    for i, ornek in enumerate(ornekler):
        W = ornek["W"]
        P = ornek["P"]
        C = ornek["C"]
        for r in range(30):
            cozum, deger = optimize(W, P, C)
            data[r, i] = -deger

        print(f"\nOrnek {i + 1}:")
        print("Ağırlıklar (W):", W)
        print("Değerler (P):", P)
        print("Kapasite (C):", C)
        print("En iyi Çözüm:", cozum)
        print("En Yüksek Değer:", -deger)

        solution[i, :] = cozum

    std_value = np.std(data, axis=0)
    max_value = np.max(data, axis=0)
    min_value = np.min(data, axis=0)
    mean_value = np.mean(data, axis=0)

    print("-------------------------------------------")
    print(f"En Yüksek Değer (Ortalama): {mean_value}")    
    print(f"Standart Sapma: {std_value}")
    print(f"Max: {max_value}")
    print(f"Min: {min_value}")
    print(f"Çözüm değerleri : {solution}")
