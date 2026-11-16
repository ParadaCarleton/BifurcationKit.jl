using Test, BifurcationKit, LinearAlgebra
const BK = BifurcationKit
####################################################################################################
# PointWeightedDot: the weights of PALC's dot product are set at each accepted point
let
    F(x, p) = p.λ .* x .+ x .^ 3 ./ 3 .+ 0.01 .* (1:length(x))
    prob = BifurcationProblem(F, zeros(3), (λ = -1.5,), (@optic _.λ))
    options = ContinuationPar(ds = 0.01, dsmax = 0.1, p_min = -3., p_max = 1., max_steps = 40, detect_bifurcation = 0, detect_fold = false, newton_options = NewtonPar(tol = 1e-11))
    θ = 0.5

    # constant weights 1 / n give the default dot product
    constant!(weights, z) = fill!(weights, 1 / length(weights))
    for tangent in (Secant(), Bordered())
        plain = continuation(prob, PALC(; tangent, θ), options)
        weighted = continuation(prob, PALC(; tangent, θ, dotθ = BK.DotTheta(BK.PointWeightedDot(constant!, zeros(3)))), options)
        @test weighted.branch.param ≈ plain.branch.param rtol = 1e-10
    end

    # weights at the point: each step's arc length constraint holds in the weights of the point it starts from
    at_point!(weights, z) = (weights .= 1 .+ z.u .^ 2)
    for (tangent, orientation_check) in ((Secant(), false), (Bordered(), false), (Bordered(), true))
        dotθ = BK.DotTheta(BK.PointWeightedDot(at_point!, zeros(3)))
        iter = ContIterable(prob, PALC(; tangent, θ, dotθ, orientation_check), options)
        next = iterate(iter)
        start_weights = copy(dotθ.dot.weights)
        steps = 0
        while ~isnothing(next)
            state = next[1]
            @test dotθ.dot.weights ≈ 1 .+ state.z.u .^ 2
            @test dotθ(state.τ, θ) ≈ 1
            z, τ, ds, weights = BK._copy(state.z), BK._copy(state.τ), state.ds, copy(dotθ.dot.weights)
            next = iterate(iter, state)
            if ~isnothing(next) && BK.converged(next[1])
                znew = next[1].z
                @test θ * dot(znew.u - z.u, Diagonal(weights), τ.u) + (1 - θ) * (znew.p - z.p) * τ.p ≈ ds rtol = 1e-6
                steps += 1
            end
        end
        @test steps > 10
        # the weights moved along the branch
        @test norm(dotθ.dot.weights - start_weights) > 0.1
    end
end
