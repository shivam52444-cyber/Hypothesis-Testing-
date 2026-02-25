import numpy as np
def decision_from_posterior(
    delta_samples: np.ndarray,
    c_fp: float,        # cost of false positive (ship but actually bad)
    c_fn: float,        # cost of false negative (don’t ship but actually good)
    c_continue: float,  # cost of continuing (delay, opportunity cost)
    mde: float,         # minimum detectable effect (business threshold)
):
    """
    delta_samples: posterior samples of (B - A)
    c_fp: cost if we ship but true delta < 0
    c_fn: cost if we don't ship but true delta > mde
    c_continue: fixed cost of waiting / continuing experiment
    mde: minimum effect worth shipping
    """

    delta = np.asarray(delta_samples)

    # Probabilities under posterior
    p_bad = np.mean(delta < 0.0)          # ship but actually worse
    p_good_big = np.mean(delta > mde)      # meaningful improvement
    p_good_small = np.mean((delta > 0) & (delta <= mde))  # positive but not big enough

    # Expected loss if we SHIP:
    # - If delta < 0: false positive cost
    # - If delta >= 0: assume no loss (or you can add opportunity cost if small)
    loss_ship = p_bad * c_fp

    # Expected loss if we DON'T SHIP:
    # - If delta > mde: false negative (we missed a real improvement)
    loss_no_ship = p_good_big * c_fn

    # Expected loss if we CONTINUE:
    # - Fixed cost of waiting
    loss_continue = c_continue

    losses = {
        "ship": loss_ship,
        "no_ship": loss_no_ship,
        "continue": loss_continue
    }

    decision = min(losses, key=losses.get)

    return {
        "decision": decision,
        "losses": losses,
        "posterior_probs": {
            "P(delta < 0)": p_bad,
            "P(delta > mde)": p_good_big,
            "P(0 < delta <= mde)": p_good_small
        }
    }