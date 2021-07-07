module TORA

using NeuralNetworkAnalysis, MAT
using NeuralNetworkAnalysis: UniformAdditivePostprocessing, SingleEntryVector,
                             NoSplitter

const verification = false;

@taylorize function TORA!(dx, x, p, t)
    x₁, x₂, x₃, x₄, u = x

    aux = 0.1 * sin(x₃)
    dx[1] = x₂
    dx[2] = -x₁ + aux
    dx[3] = x₄
    dx[4] = u
    dx[5] = zero(u)
    return dx
end

path = @modelpath("Sherlock-Benchmark-9-TORA", "controllerTora.mat")
controller = read_nnet_mat(path, act_key="act_fcns");

X₀ = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
U = ZeroSet(1)

vars_idx = Dict(:state_vars=>1:4, :control_vars=>5)
ivp = @ivp(x' = TORA!(x), dim: 5, x(0) ∈ X₀ × U)

period = 1.0  # control period
control_postprocessing = UniformAdditivePostprocessing(-10.0)  # control postprocessing

prob = ControlledPlant(ivp, controller, vars_idx, period;
                       postprocessing=control_postprocessing)

# Safety specification
T = 20.0  # time horizon
T_warmup = 2 * period  # shorter time horizon for dry run
T_reach = verification ? T : T_warmup  # shorter time horizon if not verifying

safe_states = HPolyhedron([HalfSpace(SingleEntryVector(1, 5, 1.0), 2.0),
                           HalfSpace(SingleEntryVector(1, 5, -1.0), 2.0),
                           HalfSpace(SingleEntryVector(2, 5, 1.0), 2.0),
                           HalfSpace(SingleEntryVector(2, 5, -1.0), 2.0),
                           HalfSpace(SingleEntryVector(3, 5, 1.0), 2.0),
                           HalfSpace(SingleEntryVector(3, 5, -1.0), 2.0),
                           HalfSpace(SingleEntryVector(4, 5, 1.0), 2.0),
                           HalfSpace(SingleEntryVector(4, 5, -1.0), 2.0)])
predicate = X -> X ⊆ safe_states;

alg = TMJets(abstol=1e-10, orderT=8, orderQ=3)
alg_nn = Ai2()
if verification
    splitter = BoxSplitter([4, 4, 3, 5])
else
    splitter = NoSplitter()
end

function benchmark(; T=T, silent::Bool=false)
    # We solve the controlled system:
    silent || println("flowpipe construction")
    res_sol = @timed sol = solve(prob, T=T, alg_nn=alg_nn, alg=alg,
                                 splitter=splitter)
    sol = res_sol.value
    silent || print_timed(res_sol)

    # Next we check the property for an overapproximated flowpipe:
    silent || println("property checking")
    solz = overapproximate(sol, Zonotope)
    res_pred = @timed predicate(solz)
    silent || print_timed(res_pred)
    if res_pred.value
        silent || println("The property is satisfied.")
    else
        silent || println("The property may be violated.")
    end
    return solz
end

benchmark(T=T_warmup, silent=true)  # warm-up
res = @timed benchmark(T=T_reach)  # benchmark
sol = res.value
println("total analysis time")
print_timed(res);

import DifferentialEquations

println("simulation")
res = @timed simulate(prob, T=T; trajectories=10, include_vertices=true)
sim = res.value
print_timed(res);

using Plots
import DisplayAs

function plot_helper(fig, vars)
    if vars[1] == 0
        safe_states_projected = project(safe_states, [vars[2]])
        time = Interval(0, T)
        safe_states_projected = cartesian_product(time, safe_states_projected)
    else
        safe_states_projected = project(safe_states, vars)
    end
    plot!(fig, safe_states_projected, color=:lightgreen, lab="safe states")
    if !verification && 0 ∉ vars
        plot!(fig, project(X₀, vars), lab="X₀")
    end
    plot!(fig, sol, vars=vars, color=:yellow, lab="")
    plot_simulation!(fig, sim; vars=vars, color=:black, lab="")
    fig = DisplayAs.Text(DisplayAs.PNG(fig))
end

vars = (1, 2)
fig = plot(xlab="x₁", ylab="x₂")
fig = plot_helper(fig, vars)
# savefig("TORA-x1-x2.png")
fig

vars = (1, 3)
fig = plot(xlab="x₁", ylab="x₃")
plot_helper(fig, vars)

vars = (1, 4)
fig = plot(xlab="x₁", ylab="x₄")
plot_helper(fig, vars)

vars=(2, 3)
fig = plot(xlab="x₂", ylab="x₃")
plot_helper(fig, vars)

vars=(2, 4)
fig = plot(xlab="x₂", ylab="x₄")
plot_helper(fig, vars)

vars=(3, 4)
fig = plot(xlab="x₃", ylab="x₄")
fig = plot_helper(fig, vars)
# savefig("TORA-x3-x4.png")
fig

vars = (0, 1)
fig = plot(xlab="t", ylab="x₁")
plot_helper(fig, vars)

vars=(0, 2)
fig = plot(xlab="t", ylab="x₂")
plot_helper(fig, vars)

vars=(0, 3)
fig = plot(xlab="t", ylab="x₃")
plot_helper(fig, vars)

vars=(0, 4)
fig = plot(xlab="t", ylab="x₄")
plot_helper(fig, vars)

tdom = range(0, 20, length=length(controls(sim, 1)))
fig = plot(xlab="t", ylab="u")
[plot!(fig, tdom, [c[1] for c in controls(sim, i)], lab="") for i in 1:length(sim)]
fig = DisplayAs.Text(DisplayAs.PNG(fig))

end
nothing

