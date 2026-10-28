using Test, BifurcationKit, LinearAlgebra
const BK = BifurcationKit
####################################################################################################
# orientation_dot: the dot product whose sign orients a Bordered tangent without the determinant
let
    # The fold λ = x² at x = 0, approached from x = -1 with λ decreasing.
    F(x, p) = x .^ 2 .- p.λ
    prob = BifurcationProblem(F, [-1.0], (λ = 1.0,), (@optic _.λ))
    options = ContinuationPar(ds = -0.05, dsmax = 0.1, p_min = -1., p_max = 2., max_steps = 80, save_sol_every_step = 1, detect_bifurcation = 0, detect_fold = false, newton_options = NewtonPar(tol = 1e-11))
    # Weights that all but drop x, so past the fold the sign of λ's component decides the dot product.
    tiny!(weights, z) = fill!(weights, 1e-6)
    shrunk() = BK.DotTheta(BK.PointWeightedDot(tiny!, [1e-6]))

    @test PALC(; dotθ = shrunk()).orientation_dot isa BK.DotTheta{<:BK.PointWeightedDot}
    plain = continuation(prob, PALC(; tangent = Bordered(), θ = 0.5, dotθ = shrunk(), orientation_dot = BK.DotTheta()), options)
    xs = [only(s.x) for s in plain.sol]
    @test maximum(xs) > 0.5
    @test all(diff(xs) .> 0)

    # Oriented by the shrunk dot product, the tangent turns back at the fold.
    shrunk_oriented = continuation(prob, PALC(; tangent = Bordered(), θ = 0.5, dotθ = shrunk()), options)
    @test maximum(only(s.x) for s in shrunk_oriented.sol) < 0.5
end
