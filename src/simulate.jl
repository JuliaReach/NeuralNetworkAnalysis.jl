"""
    simulate(cp::AbstractControlProblem, args...; kwargs...)

Simulate a controlled system for a family of random trajectories.

### Input

- `cp`           -- controlled problem
- `trajectories` -- (optional, default: `10`) number of simulated trajectories

### Output

An object of type [`EnsembleSimulationSolution`](@ref).

### Notes

This function uses the ensemble simulations feature from `DifferentialEquations.jl`.
"""
function simulate(cp::AbstractControlProblem, args...; kwargs...)
    if !isdefined(@__MODULE__, :DifferentialEquations)
        error("package 'DifferentialEquations' is required for simulation")
    end

    ivp = plant(cp)
    network = controller(cp)
    st_vars = state_vars(cp)
    inpt_vars = input_vars(cp)
    n = length(st_vars)
    X₀ = project(initial_state(cp), st_vars)
    if !isempty(inpt_vars)
        W₀ = project(initial_state(cp), inpt_vars)
    end
    ctrl_vars = control_vars(cp)
    τ = period(cp)
    time_span = _get_tspan(args...; kwargs...)
    trajectories = get(kwargs, :trajectories, 10)
    inplace = get(kwargs, :inplace, true)
    preprocessing = control_preprocessing(cp)
    postprocessing = control_postprocessing(cp)
    include_vertices = get(kwargs, :include_vertices, false)

    t = tstart(time_span)
    iterations = ceil(Int, diam(time_span) / τ)

    # sample initial states
    states = sample(X₀, trajectories; include_vertices=include_vertices)
    trajectories = length(states)

    # preallocate
    extended = Vector{Vector{Float64}}(undef, trajectories)
    simulations = Vector{EnsembleSolution}(undef, iterations)
    all_controls = Vector{Vector{Vector{Float64}}}(undef, iterations)
    all_inputs = Vector{Vector{Vector{Float64}}}(undef, iterations)

    @inbounds for i in 1:iterations
        # compute control inputs
        controls = Vector{Vector{Float64}}(undef, trajectories)
        for j in 1:trajectories
            x₀ = states[j]
            x₀ = apply(preprocessing, x₀)
            network_output = forward(network, x₀)
            controls[j] = apply(postprocessing, network_output)
        end
        all_controls[i] = controls

        # compute  inputs
        if !isempty(inpt_vars)
            inputs = sample(W₀, trajectories)
            all_inputs[i] = inputs
        else
            inputs = nothing
        end

        # extend system state with inputs
        for j in 1:trajectories
            if inputs == nothing
                extended[j] = vcat(states[j], controls[j])
            else
                extended[j] = vcat(states[j], inputs[j], controls[j])
            end
        end

        T = i < iterations ? (t + τ) : tend(time_span)

        # simulate system for the next period
        simulations[i] = _solve_ensemble(ivp, extended, (t, T);
                                         inplace=inplace)

        # project to state variables
        for j in 1:trajectories
            ode_solution = simulations[i][j]
            final_extended = ode_solution.u[end]
            states[j] = final_extended[st_vars]
        end

        # advance time
        t = T
    end

    return EnsembleSimulationSolution(simulations, all_controls, all_inputs)
end
