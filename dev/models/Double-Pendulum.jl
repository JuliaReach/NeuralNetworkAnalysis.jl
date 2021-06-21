module DoublePendulum

using NeuralNetworkAnalysis
using NeuralNetworkAnalysis: SingleEntryVector, Specification

const falsification = true;

const m = 0.5
const L = 0.5
const c = 0.0
const g = 1.0
const gL = g/L
const mL = m*L^2;

@taylorize function double_pendulum!(dx, x, p, t)
    x₁, x₂, x₃, x₄, T₁, T₂ = x

    # auxiliary terms
    Δ12 = x₁ - x₂
    ★ = cos(Δ12)
    x3sin12 = x₃^2 * sin(Δ12)
    x4sin12 = x₄^2 * sin(Δ12) / 2
    gLsin1 = gL * sin(x₁)
    gLsin2 = gL * sin(x₂)
    T1_frac = (T₁ - c * x₃) / (2 * mL)
    T2_frac = (T₂ - c * x₄) / mL
    bignum = x3sin12 - ★ * (gLsin1 - x4sin12 + T1_frac) + gLsin2 + T2_frac
    denom = ★^2 / 2 - 1

    dx[1] = x₃
    dx[2] = x₄
    dx[3] = ★ * bignum / (2 * denom) - x4sin12 + gLsin1 + T1_frac
    dx[4] = - bignum / denom
end;

function DoublePendulum_model(use_less_robust_controller::Bool)
    net_lr = @modelpath("Double-Pendulum", "controller_double_pendulum_less_robust.nnet")
    net_mr = @modelpath("Double-Pendulum", "controller_double_pendulum_more_robust.nnet")
    controller = read_nnet(use_less_robust_controller ? net_lr : net_mr)

    X₀ = BallInf(fill(1.15, 4), 0.15)
    if falsification
        # choose a single point in the initial states (here: the top-most one)
        if use_less_robust_controller
            X₀ = Singleton(high(X₀))
        else
            X₀ = Singleton(low(X₀))
        end
    end
    U₀ = ZeroSet(2)
    vars_idx = Dict(:state_vars=>1:4, :control_vars=>5:6)
    ivp = @ivp(x' = double_pendulum!(x), dim: 6, x(0) ∈ X₀ × U₀)

    period = use_less_robust_controller ? 0.05 : 0.02  # control period

    prob = ControlledPlant(ivp, controller, vars_idx, period)

    # Safety specification: [x[1], x[2], x[3], x[4]] ∈ safe_states for all t
    if falsification
        if use_less_robust_controller
            k = 5
        else
            k = 7
        end
    else
        k = 20
    end
    T = k * period  # time horizon

    lb = use_less_robust_controller ? -1.0 : -0.5
    ub = use_less_robust_controller ? 1.7 : 1.5
    safe_states = HPolyhedron([HalfSpace(SingleEntryVector(1, 6, 1.0), ub),
                               HalfSpace(SingleEntryVector(1, 6, -1.0), -lb),
                               HalfSpace(SingleEntryVector(2, 6, 1.0), ub),
                               HalfSpace(SingleEntryVector(2, 6, -1.0), -lb),
                               HalfSpace(SingleEntryVector(3, 6, 1.0), ub),
                               HalfSpace(SingleEntryVector(3, 6, -1.0), -lb),
                               HalfSpace(SingleEntryVector(4, 6, 1.0), ub),
                               HalfSpace(SingleEntryVector(4, 6, -1.0), -lb)])
    predicate = X -> isdisjoint(overapproximate(X, Hyperrectangle),
                                safe_states)  # sufficient property for guaranteed violation
    predicate_sol = sol -> any(predicate(R) for F in sol for R in F);

    spec = Specification(T, predicate_sol, safe_states)

    return prob, spec
end;

import DifferentialEquations

function run(use_less_robust_controller::Bool)
    prob, spec = DoublePendulum_model(use_less_robust_controller)

    alg = TMJets20(abstol=1e-9, orderT=8, orderQ=1)
    alg_nn = Ai2()

    function benchmark(; silent::Bool=false)
        # We solve the controlled system:
        silent || println("flowpipe construction")
        res_sol = @timed solve(prob, T=spec.T, alg_nn=alg_nn, alg=alg)
        sol = res_sol.value
        silent || print_timed(res_sol)

        # Next we check the property for an overapproximated flowpipe:
        solz = overapproximate(sol, Zonotope)
        if spec.predicate(solz)
            silent || println("The property is violated.")
        else
            silent || println("The property may be satisfied.")
        end

        # We also compute some simulations:
        silent || println("simulation")
        trajectories = falsification ? 1 : 10
        res_sim = @timed simulate(prob, T=spec.T, trajectories=trajectories,
                                  include_vertices=!falsification)
        sim = res_sim.value
        silent || print_timed(res_sim)

        return solz, sim
    end;

    benchmark(silent=true)  # warm-up
    if use_less_robust_controller
        println("Running analysis with less robust controller")
    else
        println("Running analysis with more robust controller")
    end
    res = @timed benchmark()  # benchmark
    sol, sim = res.value
    println("total analysis time")
    print_timed(res)

    return sol, sim, prob, spec
end;

# pass flag to choose the controller
# `true`: use a less robust controller
# `false`: use a more robust controller
# the choice also influences settings like the period and the specification
res_true = run(true)
res_false = run(false);

using Plots
import DisplayAs

function plot_helper(fig, vars, sol, sim, prob, spec)
    safe_states = spec.ext
    if vars[1] == 0
        safe_states_projected = project(safe_states, [vars[2]])
        time = Interval(0, T)
        safe_states_projected = cartesian_product(time, safe_states_projected)
    else
        safe_states_projected = project(safe_states, vars)
    end
    plot!(fig, safe_states_projected, color=:lightgreen, lab="safe states")
    if !falsification && 0 ∉ vars
        plot!(fig, project(initial_state(prob), vars), lab="X₀")
    end
    plot!(fig, sol, vars=vars, color=:yellow, lab="")
    lab_sim = falsification ? "simulation" : ""
    plot_simulation!(fig, sim; vars=vars, color=:black, lab=lab_sim)
    fig = DisplayAs.Text(DisplayAs.PNG(fig))
end

vars=(1, 2)
fig = plot(xlab="θ₁", ylab="θ₂")
sol, sim, prob, spec = res_true
if falsification
    plot!(leg=:topleft)
end
xlims!(-0.5, 2.0)
fig = plot_helper(fig, vars, sol, sim, prob, spec)
# savefig("Double-Pendulum-less-robust-x$(vars[1])-x$(vars[2]).png")
fig

vars=(3, 4)
fig = plot(xlab="θ₁'", ylab="θ₂'")
sol, sim, prob, spec = res_true
if falsification
    plot!(leg=:topleft)
end
xlims!(-0.7, 1.7)
ylims!(-1.6, 1.5)
fig = plot_helper(fig, vars, sol, sim, prob, spec)
# savefig("Double-Pendulum-less-robust-x$(vars[1])-x$(vars[2]).png")
fig

vars=(1, 2)
fig = plot(xlab="θ₁", ylab="θ₂")
sol, sim, prob, spec = res_false
if falsification
    plot!(leg=:topleft)
end
fig = plot_helper(fig, vars, sol, sim, prob, spec)
# savefig("Double-Pendulum-more-robust-x$(vars[1])-x$(vars[2]).png")
fig

vars=(3, 4)
fig = plot(xlab="θ₁'", ylab="θ₂'")
sol, sim, prob, spec = res_false
if falsification
    plot!(leg=:topleft)
    ylims!(-1.0, 1.5)
else
    xlims!(-1.8, 1.5)
    ylims!(-1.6, 1.5)
end
fig = plot_helper(fig, vars, sol, sim, prob, spec)
# savefig("Double-Pendulum-more-robust-x$(vars[1])-x$(vars[2]).png")
fig

end
nothing

