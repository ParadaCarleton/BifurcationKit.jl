# The correctors of PALC and MoorePenrose and the Bordered tangent take `∂ₚF` from an `R01` passed by the user, by finite differences otherwise
using Test, BifurcationKit
const BK = BifurcationKit

# x³ - x - a = 0, with folds at a = ∓2 / (3√3)
F_cubic(x, par) = x .^ 3 .- x .- par.a
dFda_cubic(x, par) = fill(-one(eltype(x)), length(x))

# a `∂ₚF` which reports that it was called
struct R01Called <: Exception end
R01_called(x, par) = throw(R01Called())

# `F_cubic` in a buffer of the type of `x`, which a dual number `par.a` cannot be written to
function F_cubic_float(x, par)
    out = similar(x)
    out .= x .^ 3 .- x .- par.a
    return out
end

@testset "∂ₚF of the corrector from the jet" begin
    opts = ContinuationPar(p_min = -1.0, p_max = 1.0, ds = 0.01, dsmax = 0.05, max_steps = 200, detect_bifurcation = 0)
    for alg in (PALC(), PALC(tangent = Bordered()), MoorePenrose(), MoorePenrose(method = BK.pInv), MoorePenrose(method = BK.iterative))
        prob_called = BifurcationProblem(F_cubic, [-1.5], (a = -1.0,), (@optic _.a); R01 = R01_called)
        @test_throws R01Called continuation(prob_called, alg, opts)

        prob = BifurcationProblem(F_cubic, [-1.5], (a = -1.0,), (@optic _.a); R01 = dFda_cubic)
        br = continuation(prob, alg, opts)
        @test all(point -> abs(only(F_cubic(point.x, (a = point.p,)))) < 1.0e-8, br.sol)
        # the branch passes both folds
        @test any(diff(br.branch.param) .< 0)
    end

    # the default jet (`R01 = AutoDiff()`) keeps finite differences in `p`
    for alg in (PALC(), PALC(tangent = Bordered()), MoorePenrose())
        prob = BifurcationProblem(F_cubic_float, [-1.5], (a = -1.0,), (@optic _.a))
        br = continuation(prob, alg, opts)
        @test any(diff(br.branch.param) .< 0)
    end
end
