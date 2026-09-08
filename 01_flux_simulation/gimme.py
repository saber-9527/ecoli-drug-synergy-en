"""
Python implementation of GIMME compatible with COBRApy.

Usage:
    from cobra.flux_analysis import gimme
    sol = gimme(model, reaction_scores, threshold, fraction_of_optimum)
"""

import cobra


def gimme(
    model,
    reaction_scores,
    threshold=0.5,
    fraction_of_optimum=0.1,
    objective=None
):
    """
    Run GIMME.

    reaction_scores: dict {reaction_id: numeric score}
    threshold: cutoff for "active" reactions
    fraction_of_optimum: minimal biomass fraction
    objective: biomass reaction id or None (use model.objective)
    """

    # Make a local copy
    m = model.copy()

    # ------------------------------------------------------------
    # STEP 1: Calculate baseline growth
    # ------------------------------------------------------------
    if objective is not None:
        m.objective = m.reactions.get_by_id(objective)

    baseline = m.slim_optimize()
    if baseline is None or baseline < 1e-9:
        raise RuntimeError("Baseline model infeasible.")

    biomass_rxn = list(m.objective.keys())[0]
    biomass_lb = baseline * fraction_of_optimum
    biomass_rxn.lower_bound = biomass_lb

    # ------------------------------------------------------------
    # STEP 2: Build penalty objective
    # minimize sum(p_r * |v_r|)
    # ------------------------------------------------------------
    penalties = {}
    for rxn in m.reactions:
        s = reaction_scores.get(rxn.id, None)
        if s is None:
            penalties[rxn.id] = 0.0
        else:
            penalties[rxn.id] = max(0.0, threshold - s)

    # Clear all previous objective coefficients
    for r in m.reactions:
        m.objective.set_linear_coefficients({r: 0})

    # Add penalties
    for rxn in m.reactions:
        m.objective.set_linear_coefficients({rxn: penalties[rxn.id]})

    m.objective_direction = "min"

    # ------------------------------------------------------------
    # STEP 3: Solve
    # ------------------------------------------------------------
    sol = m.optimize()

    if sol is None or sol.status != "optimal":
        raise RuntimeError("GIMME optimization failed.")

    return sol
