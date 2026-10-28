using Test, BifurcationKit, LinearAlgebra
const BK = BifurcationKit
####################################################################################################
# orientation_dot: the dot product whose sign orients a Bordered tangent without the determinant
let
    # λ = x² at x = -1: every tangent is ± (1, -2) in (x, λ).
    F(x, p) = x .^ 2 .- p.λ
    prob = BifurcationProblem(F, [-1.0], (λ = 1.0,), (@optic _.λ))
    options = ContinuationPar(ds = 0.01, max_steps = 5, detect_bifurcation = 0, detect_fold = false)
    # Weights that all but drop x, so the sign of λ's component decides the dot product.
    tiny!(weights, z) = fill!(weights, 1e-6)
    shrunk() = BK.DotTheta(BK.PointWeightedDot(tiny!, [1e-6]))
    @test PALC(; dotθ = shrunk()).orientation_dot isa BK.DotTheta{<:BK.PointWeightedDot}

    # From the old tangent (1, 0.4), the plain product with (1, -2) is 0.5 - 0.4 > 0 and the shrunk one is negative.
    function new_tangent(algorithm)
        iter = ContIterable(prob, algorithm, options)
        state, _ = iterate(iter)
        state.τ.u .= [1.0]
        state.τ.p = 0.4
        return BK.bordered_tangent(state, iter, state.z, BK.getdot(iter); orientation_dot = BK.get_orientation_dot(iter))
    end
    plain = new_tangent(PALC(; tangent = Bordered(), θ = 0.5, dotθ = shrunk(), orientation_dot = BK.DotTheta()))
    @test only(plain.u) > 0 && plain.p < 0
    @test plain.p ≈ -2 * only(plain.u)

    # Any callable with a DotTheta's arguments orients; only its sign is read.
    plain_product(u1, u2, p1, p2, θ) = dot(u1, u2) - p1 * p2 / 4
    @test only(new_tangent(PALC(; tangent = Bordered(), θ = 0.5, dotθ = shrunk(), orientation_dot = plain_product)).u) > 0

    # Without orientation_dot the shrunk dotθ orients, and turns the tangent back.
    @test only(new_tangent(PALC(; tangent = Bordered(), θ = 0.5, dotθ = shrunk())).u) < 0
end
