using NeuralNetworkAnalysis

@taylorize function f!(dx, x, p, t)
    x₁, x₂, x₃, x₄, w, u = x

    dx[1] = x₃^3 - x₂ + w
    dx[2] = x₃
    dx[3] = u
    dx[4] = zero(x₄) # w
    dx[5] = zero(x₅) # u
    return dx
end

X₀ = Hyperrectangle(low=[0.35, 0.45, 0.25], high=[0.45, 0.55, 0.35])
W₀ = Interval(-0.01, 0.01)
U₀ = Interval(2.0, 2.0)
prob = @ivp(x' = f!(x), dim: 5, x(0) ∈ X₀ × W₀ × U₀);

##sol = solve(prob, T=2.0);

