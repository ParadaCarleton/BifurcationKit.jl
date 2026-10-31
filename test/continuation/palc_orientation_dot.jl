using Test, BifurcationKit, LinearAlgebra
const BK = BifurcationKit

"The fold λ = x²."
F(x, p) = x .^ 2 .- p.λ

"A product with the arguments of a `DotTheta`; only its sign orients."
plain_product(u1, u2, p1, p2, θ) = dot(u1, u2) + p1 * p2

let
    prob = BifurcationProblem(F, [-1.0], (λ = 1.0,), (@optic _.λ))
    options = ContinuationPar(ds = -0.05, dsmax = 0.1, p_min = -1., p_max = 2., max_steps = 80, detect_bifurcation = 0, newton_options = NewtonPar(tol = 1e-11))

    @test PALC(; tangent = Bordered(), θ = 0.5, orientation_dot = plain_product).orientation_dot === plain_product
    @test PALC().orientation_dot isa BK.DotTheta
    default = continuation(prob, PALC(; tangent = Bordered(), θ = 0.5), options)
    explicit = continuation(prob, PALC(; tangent = Bordered(), θ = 0.5, orientation_dot = BK.DotTheta()), options)
    @test length(default.sol) == length(explicit.sol)
    called = continuation(prob, PALC(; tangent = Bordered(), θ = 0.5, orientation_dot = plain_product), options)
    @test length(called.sol) > 2
end
