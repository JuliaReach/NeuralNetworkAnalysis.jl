using NeuralNetworkAnalysis

@taylorize function cartpole!(du, u, p, t)
    local f, m, l, mt, g = 10, 0.1, 0.5, 1.1, 9.8
    sinθ = sin(u[3])
    cosθ = cos(u[3])
    aux = (f + m*l*u[4]^2*sinθ) / mt
    aux2 = l*(4/3 - m*cosθ^2/mt)
    aux3 = (g*sinθ- cosθ) * aux / aux2
    aux4 = cosθ/mt
    aux5 = m*l*aux3

    du[1] = u[2]
    du[2] = aux - aux5 * aux4
    du[3] = u[4]
    du[4] = aux3 * aux4

    return du
end

X₀ = Hyperrectangle(low=[-0.05, -0.05, -0.05, -0.05], high=[0.05, 0.05, 0.05, 0.05])

prob = @ivp(x' = cartpole!(x), dim: 4, x(0) ∈ X₀)

# sol = solve(prob, T=1.0, alg=TMJets(maxsteps=20_000, abstol=1e-10));

using MAT
path = joinpath(@modelpath("Cart-Pole", "CartPole_Controller.mat"))
controller = read_nnet_mat(path, key="controller")

