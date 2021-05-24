# # Double Inverted Pendulum
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/Double-Pendulum.ipynb)
#
# This example consists of a double-link pendulum with equal point masses ``m``
# at the end of connected mass-less links of length ``L``. Both links are
# actuated with torques ``T_1`` and ``T_2`` and we assume viscous friction
# exists with a coefficient of ``c`.

# ## Model
#
# ```math
# \begin{aligned}
# 2\ddot \theta_1 + \ddot \theta_2 cos(\theta_2 - \theta_1) - \ddot \theta^2_2 sin(\theta_2 - \theta_1) - 2 \frac{g}{L}sin\theta_1 + \frac{c}{mL^2}\dot\theta_1 &= \frac{1}{mL^2}T_1 \\
# \ddot \theta_1 cos(\theta_2 - \theta_1) + \ddot \theta_2 + \ddot \theta^2_1 sin(\theta_2 - \theta_1) - \frac{g}{L}sin\theta_2 + \frac{c}{mL^2}\dot\theta_2 &= \frac{1}{mL^2}T_2
# \end{aligned}
# ```
# where ``θ_1`` and ``θ_2`` are the angles that links make with the
# upward vertical axis. The state is:
# ```math
# \begin{aligned}
# [\theta_1, \theta_2, \dot \theta_1, \dot \theta_2]
# \end{aligned}
# ```
# The angular velocity and acceleration of the links are denoted with ``θ_1'``,
# ``θ_2'``, ``θ_1''`` and ``θ_2''`` and ``g`` is the gravitational acceleration.

using NeuralNetworkAnalysis

# flag to choose the setting
# `false`: use a less robust controller
# `true`: use a more robust controller
# the choice also influences settings like the period and the specification
use_less_robust_controller = false;

# model constants
const m = 0.5;
const L = 0.5;
const c = 0.0;
const g = 1.0;
const gL = g/L;
const mL = 1/(m*L^2);

@taylorize function double_pendulum!(dx, x, p, t)
    x₁, x₂, x₃, x₄, T₁, T₂ = x

    ## auxiliary terms
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

net_lr = @modelpath("Double-Pendulum", "controller_double_pendulum_less_robust.nnet")
net_mr = @modelpath("Double-Pendulum", "controller_double_pendulum_more_robust.nnet")
controller = read_nnet(use_less_robust_controller ? net_lr : net_mr);

# ## Specification

X₀ = BallInf(fill(1.15, 4), 0.15);
U₀ = ZeroSet(2);
vars_idx = Dict(:state_vars=>1:4, :control_vars=>5:6);
ivp = @ivp(x' = double_pendulum!(x), dim: 6, x(0) ∈ X₀ × U₀);

period = use_less_robust_controller ? 0.05 : 0.02;  # control period
T = 3 * period;  # time horizon

prob = ControlledPlant(ivp, controller, vars_idx, period);

safe_states = use_less_robust_controller ?
    BallInf(fill(0.35, 4), 1.35) : BallInf(fill(0.5, 4), 1.0);
# TODO spec: [x[1], x[2], x[3], x[4]] ∈ safe_states for all t

# ## Results

alg = TMJets20(abstol=1e-8, orderT=4, orderQ=2);
alg_nn = SampledApprox(nsamples=100, include_vertices=false);

@time sol = solve(prob, T=T, alg_nn=alg_nn, alg=alg);

# We also compute some simulations:
using DifferentialEquations
@time sim = simulate(prob, T=T; trajectories=10, include_vertices=true);

# Finally we plot the results
using Plots
import DisplayAs

function plot_helper(fig, vars)
    plot!(fig, sol, vars=vars, lab="");
    plot_simulation!(fig, sim; vars=vars, color=:red, lab="");
    fig = DisplayAs.Text(DisplayAs.PNG(fig))
##    infix = use_less_robust_controller ? "less" : "more"  # TODO temporary helper
##    savefig("DoublePendulum-$infix-$(vars[1])-$(vars[2])")
end

vars=(1, 2);
fig = plot(xlab="x₁", ylab="x₂");
plot_helper(fig, vars)

#-

vars=(3, 4);
fig = plot(xlab="x₃", ylab="x₄");
plot_helper(fig, vars)
