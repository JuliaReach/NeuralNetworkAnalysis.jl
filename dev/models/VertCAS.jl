module VertCAS

using NeuralNetworkAnalysis

const g = 32.2  # gravitational constant

# accelerations (middle)
const ACC_MIDDLE = Dict(:COC => 0.0, :DNC => -7g/24, :DND => 7g/24,
                        :DES1500 => -7g/24, :CL1500 => 7g/24, :SDES1500 => -g/3,
                        :SCL1500 => g/3, :SDES2500 => -g/3, :SCL2500 => g/3)

# continuous dynamics matrix (h, hdot0)
const Δτ = 1.0
const A = [1  -Δτ; 0  1.];

CONTROLLERS = Dict{Symbol, Any}()

CTRL_IDX = [:COC, :DNC, :DND, :DES1500,
            :CL1500, :SDES1500, :SCL1500,
            :SDES2500, :SCL2500]

path = @modelpath("VertCAS", "")
for i = 1:9
    file = joinpath(path, "VertCAS_noResp_pra0$(i)_v9_20HU_200.nnet")
    adv = CTRL_IDX[i]
    CONTROLLERS[adv] = read_nnet(file)
end


struct State{T, N}
    state::T # state for (h, hdot0) variables
    τ::N
    adv::Symbol
end


ADVISORIES = Dict{Symbol, Any}()

# for every advisory, return a function to check
# whether the current climbrate complies with the advisory or not
# if the advisory *does not* comply => it is changed according to the ACC dictionary
# otherwise, the new acceleration is zero
using Symbolics
@variables x

ADVISORIES[:COC] = EmptySet(1)
ADVISORIES[:DNC] = HalfSpace(x <= 0)
ADVISORIES[:DND] = HalfSpace(x >= 0)
ADVISORIES[:DES1500] = HalfSpace(x <= -1500)
ADVISORIES[:CL1500] = HalfSpace(x >= 1500)
ADVISORIES[:SDES1500] = HalfSpace(x <= -1500)
ADVISORIES[:SCL1500] = HalfSpace(x >= 1500)
ADVISORIES[:SDES2500] = HalfSpace(x <= -2500)
ADVISORIES[:SCL2500] = HalfSpace(x >= 2500)

# this function receives X = [h, hdot0, τ, adv′] and the
# *previous* advisory adv
function get_acceleration(X::State, adv; ACC=ACC_MIDDLE)

    # obtain projection on hdot
    hdot = _interval(X.state, 2)

    # transform units from ft/s to ft/min
    hdot = 60 * hdot

    # new advisory
    adv′ = X.adv

    # check whether the current state complies with the advisory
    comply = hdot ⊆ ADVISORIES[adv′]

    if adv == adv′ && comply
        return 0.0
    else
        return ACC[adv′]
    end
end

# scalar case; alg is ignored
function forward_adv(X::Singleton, τ, adv; alg=nothing)
    v = vcat(element(X), τ)
    u = forward(CONTROLLERS[adv], v)
    imax = argmax(u)
    return CTRL_IDX[imax]
end

# set-based case
function forward_adv(X::AbstractZonotope, τ, adv; alg=Ai2())
    Y = cartesian_product(X, Singleton([τ]))

    out = forward_network(alg, CONTROLLERS[adv], Y)

    imax = argmax(high(out))
    return CTRL_IDX[imax]
end

function VCAS!(out::Vector{State{T, N}}, KMAX; ACC=ACC_MIDDLE, alg_nn=Ai2()) where {T, N}

    # unpack initial state
    X0 = first(out)
    S = X0.state
    τ = X0.τ
    adv = X0.adv

    # get initial acceleration
    hddot = ACC[adv]

    for i in 1:KMAX
        # compute next state
        b = [-hddot*Δτ^2 / 2, hddot * Δτ]
        S′ = affine_map(A, S, b)
        τ′ = τ - 1
        adv′ = forward_adv(S′, τ′, adv, alg=alg_nn)

        # store new state
        X′ = State(S′, τ′, adv′)
        push!(out, X′)

        # get acceleration from network
        # this logic only works for ACC_MIDDLE
        hddot = get_acceleration(X′, adv; ACC=ACC)

        # update current state
        S = S′
        τ = τ′
        adv = adv′
    end
    return out
end;

bad_states = HalfSpace([1.0, 0.0], 100.) ∩ HalfSpace([-1.0, 0.0], 100.)

# property for guaranteed violation
predicate = X -> X ⊆ bad_states
predicate_sol = sol -> any(predicate(R) for F in sol for R in F);

const h0 = Interval(-133, -129)
const hdot0 = [-19.5,-22.5, -25.5, -28.5]
const τ0 = 25.0
const adv0 = :COC

function _random_states(k=1, include_vertices::Bool=false, rand_h0::Bool=true)
    N = Float64
    T = Singleton{N, Vector{N}}
    states = Vector{State{T, N}}()
    xs = sample(h0, k, include_vertices=include_vertices)
    for x in xs
        if rand_h0
            # use a random value for y
            y = hdot0[rand(1:4)]
            S0 = State(Singleton([x[1], y]), τ0, adv0)
            push!(states, S0)
            continue
        end
        # use all possible values for y
        for i in 1:4
            y = hdot0[i]
            S0 = State(Singleton([x[1], y]), τ0, adv0)
            push!(states, S0)
        end
    end
    return states
end

function _all_states()
    S0 = [convert(Zonotope, concretize(h0 × Singleton([hdot0[i]]))) for i in 1:4]
    return [State(S0i, τ0, adv0) for S0i in S0]
end

function simulate_VCAS(X0::State; KMAX=10)
    out = [X0]
    sizehint!(out, KMAX+1)

    VCAS!(out, KMAX, ACC=ACC_MIDDLE)
    return out
end

# project onto the h variable
function _project(X::Vector{State{T, N}}) where {T<:Singleton, N}
    return [Singleton([Xi.state.element[1], Xi.τ]) for Xi in X]
end

_interval(X::LazySet, i) = overapproximate(Projection(X, (i,)), Interval)

function _project(X::Vector{State{T, N}}) where {T<:Zonotope, N}
    Xint = [_interval(Xi.state, 1) × Singleton([Xi.τ]) for Xi in X]
end

function run(X0)
    ensemble = [simulate_VCAS(X0i) for X0i in X0]
    res = _project.(ensemble)
    return res
end

function check(sol)
    println("property checking")
    res_pred = @timed predicate_sol(sol)
    print_timed(res_pred)
    if res_pred.value
        println("The property is violated.")
    else
        println("The property may be satisfied.")
    end
end;

X0 = _random_states(10, true, false)  # randomly sampled points (incl. vertices)
println("$(length(X0)) simulations with central advisories")
res = @timed begin
    res1 = run(X0)
    check(res1)
end
println("total analysis time")
print_timed(res);

println("flowpipe construction (unsound) with central advisories")
res = @timed begin
    res2 = run(_all_states())
    check(res1)
end
println("total analysis time")
print_timed(res);

using Plots
import DisplayAs

function plot_helper()
    fig = plot(xlims=(-200, 200), ylims=(14, 26), xlab="h (vertical distance)",
               ylab="τ (time to reach horizontally)")
    plot!(fig, bad_states, alpha=0.2, c=:red, lab="unsafe states")
    return fig
end

fig = plot_helper()
for o in res1
    plot!(fig, o, alpha=1.0)
end
fig = DisplayAs.Text(DisplayAs.PNG(fig))
# savefig("VertCAS-rand.png")
fig

fig = plot_helper()
for (i, c) in [(1, :blue), (2, :orange), (3, :green), (4, :cyan)]
    [plot!(fig, o, lw=2.0, alpha=1., markershape=:none, seriestype=:shape, c=c) for o in res2[i]]
end
fig = DisplayAs.Text(DisplayAs.PNG(fig))
# savefig("VertCAS-sets.png")
fig

end
nothing

