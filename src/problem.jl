# ======================
# Control postprocessing
# ======================

abstract type ControlPostprocessing end

struct NoPostprocessing <: ControlPostprocessing
end

apply(::NoPostprocessing, x) = x

struct UniformAdditivePostprocessing{N<:Number} <: ControlPostprocessing
    shift::N

    function UniformAdditivePostprocessing(shift::N) where {N<:Number}
        if shift == zero(N)
            @warn("zero additive control postprocessing is discouraged; " *
                  "use `NoPostprocessing` instead")
        end
        return new{N}(shift)
    end
end

function apply(postprocessing::UniformAdditivePostprocessing, x)
    return x .+ postprocessing.shift
end

function apply(postprocessing::UniformAdditivePostprocessing, X::LazySet)
    return translate(X, fill(postprocessing.shift, dim(X)))
end

function apply(postprocessing::UniformAdditivePostprocessing, X::UnionSetArray)
    return UnionSetArray([apply(postprocessing, Xi) for Xi in array(X)])
end

# =====================
# Control preprocessing
# =====================

abstract type ControlPreprocessing end

struct NoPreprocessing <: ControlPreprocessing
end

apply(::NoPreprocessing, x) = x

struct FunctionPreprocessing{F<:Function} <: ControlPreprocessing
    f::F

    function FunctionPreprocessing(f::F) where {F<:Function}
        return new{F}(f)
    end
end

function apply(funct::FunctionPreprocessing, x)
    return funct.f(x)
end

# ================
# Control problems
# ================


abstract type AbstractControlProblem end

"""
    ControlledPlant{ST, CT, XT, DT, PT, CPRT, CPST} <: AbstractControlProblem

Struct representing a closed-loop controlled system.

### Fields

- `ivp`            -- initial-value problem
- `controller`     -- controller
- `vars`           -- dictionary storing state variables, input variables and
                      control variables
- `period`         -- control period
- `postprocessing` -- postprocessing of the controller output
- `preprocessing`  -- preprocessing of the controller input

### Parameters

- `ST`:  type of system
- `CT`:  type of controller
- `XT`:  type of initial condition
- `DT`:  type of variables
- `PT`:  type of period
- `CPRT`: type of control preprocessing
- `CPST`: type of control postprocessing

### Notes

While typically the `controller` is a neural network, this struct does not
prescribe the type.
"""
struct ControlledPlant{ST, CT, XT, DT, PT, CPRT, CPST} <: AbstractControlProblem
    ivp::InitialValueProblem{ST, XT}
    controller::CT
    vars::Dict{Symbol, DT}
    period::PT
    preprocessing::CPRT
    postprocessing::CPST

    function ControlledPlant(ivp::InitialValueProblem{ST, XT},
                             controller::CT,
                             vars::Dict{Symbol, DT},
                             period::PT;
                             preprocessing::CPRT=NoPreprocessing(),
                             postprocessing::CPST=NoPostprocessing()) where {ST, CT, XT, DT, PT, CPRT, CPST}
        return new{ST, CT, XT, DT, PT, CPRT, CPST}(ivp, controller, vars, period, preprocessing, postprocessing)
    end
end

plant(cp::ControlledPlant) = cp.ivp
MathematicalSystems.initial_state(cp::ControlledPlant) = initial_state(cp.ivp)
system(cp::ControlledPlant) = cp.ivp.s
controller(cp::ControlledPlant) = cp.controller
period(cp::ControlledPlant) = cp.period
control_postprocessing(cp::ControlledPlant) = cp.postprocessing
control_preprocessing(cp::ControlledPlant) = cp.preprocessing

state_vars(cp::ControlledPlant) = get(cp.vars, :state_vars, Int[])
input_vars(cp::ControlledPlant) = get(cp.vars, :input_vars, Int[])
control_vars(cp::ControlledPlant) = get(cp.vars, :control_vars, Int[])

# =============
# Specification
# =============

@with_kw struct Specification{NT, PT, ET}
    T::NT
    predicate::PT
    ext::ET = nothing
end

predicate(spec::Specification) = spec.predicate
time_horizon(spec::Specification) = spec.T
ext(spec::Specification) = spec.ext
