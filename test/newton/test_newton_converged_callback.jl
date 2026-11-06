# using Revise
using Test
using BifurcationKit, LinearAlgebra
const BK = BifurcationKit

# convergence test whose tolerance grows with the size of the iterate
struct ScaledTolerance{T}
    tol::T
end
(check::ScaledTolerance)(state; k...) = state.residual < check.tol * maximum(abs, state.x)

# deprecated form of the same test, as a callback verdict
struct ScaledToleranceVerdict{T}
    tol::T
end
function (check::ScaledToleranceVerdict)(state; k...)
    if state.residual < check.tol * maximum(abs, state.x)
        return NewtonConverged()
    end
    return true
end

# convergence test which passes from Newton step `step` on
struct FromStep
    step::Int
end
(check::FromStep)(state; k...) = state.step >= check.step

let
    F(x, p) = @. x^3 / 3 + x - p.r
    J(x, p) = diagm(0 => 1 .+ x .^ 2)
    par = (r = 0.5,)
    prob = BifurcationProblem(F, fill(2.0, 3), par, (@optic _.r); J)
    opts = NewtonPar(tol = 1e-12)
    @test opts.convergence_test === BK.convergence_test_default

    strict = BK.solve(prob, Newton(), opts)
    @test BK.converged(strict)
    @test last(strict.residuals) < 1e-12

    # `tol = Inf` leaves convergence to the test alone
    loosened = BK.solve(prob, Newton(), NewtonPar(tol = Inf, convergence_test = ScaledTolerance(1e-3)))
    @test BK.converged(loosened)
    @test loosened.itnewton < strict.itnewton
    @test 1e-12 < last(loosened.residuals) < 1e-3 * maximum(abs, loosened.u)

    # both tests must pass: a test still failing under `tol` keeps the iterations going
    both = BK.solve(prob, Newton(), NewtonPar(tol = 1e-3, convergence_test = FromStep(strict.itnewton)))
    @test BK.converged(both)
    @test both.itnewton == strict.itnewton
    # ... and a residual test still failing keeps them going past the convergence test
    @test BK.solve(prob, Newton(), NewtonPar(tol = 1e-12, convergence_test = FromStep(1))).itnewton == strict.itnewton
    # a test which never passes never converges
    @test ~BK.converged(BK.solve(prob, Newton(), NewtonPar(tol = Inf, max_iterations = 5, convergence_test = FromStep(6))))

    # a NaN residual stops the iterations, unconverged, whatever the tests
    nan = BK.solve(BifurcationProblem((x, p) -> x .* NaN, [1.0], (r = 0.5,), (@optic _.r)), Newton(), NewtonPar(tol = Inf))
    @test ~BK.converged(nan) && nan.itnewton == 0

    # `false` at the last call of the callback refuses a converged point
    @test ~BK.converged(BK.solve(prob, Newton(), opts; callback = (state; k...) -> state.step == 0))
    @test BK._converged(true, true) && ~BK._converged(false, true) && ~BK._converged(true, false)

    # a first residual equal to `tol` stops the iterations, so it converges
    tie = BK.solve(BifurcationProblem((x, p) -> x .- p.r, [0.0], (r = 0.5,), (@optic _.r)), Newton(), NewtonPar(tol = 0.5))
    @test BK.converged(tie) && tie.itnewton == 0

    # NewtonPar{T, L, E} and ContinuationPar{T, S, E} keep working without the type of the test
    @test NewtonPar{Float64, DefaultLS, DefaultEig}(tol = 1e-6).convergence_test === BK.convergence_test_default
    @test ContinuationPar{Float64, DefaultLS, DefaultEig}(newton_options = NewtonPar{Float64, DefaultLS, DefaultEig}(convergence_test = FromStep(2))).newton_options.convergence_test === FromStep(2)

    # the correctors converge at the convergence test
    for alg in (PALC(), MoorePenrose(), Natural())
        scaled = ContinuationPar(p_min = 0.0, p_max = 1.0, ds = 0.05, max_steps = 200, detect_bifurcation = 0, newton_options = NewtonPar(tol = Inf, max_iterations = 10, convergence_test = ScaledTolerance(1e-10)))
        floored = continuation(prob, alg, scaled)
        @test maximum(floored.param) >= 1.0
        # Natural records its predictor's parameter at the bound, so only points inside the box are checked
        @test all(point -> norm(F(point.x, (r = point.p,)), Inf) < 1e-10 * maximum(abs, point.x), filter(point -> point.p <= 1.0, floored.sol))
    end
end

# deprecated: a callback returning `NewtonConverged()` still stops, converged, whatever `tol`
let
    F(x, p) = @. x^3 / 3 + x - p.r
    J(x, p) = diagm(0 => 1 .+ x .^ 2)
    prob = BifurcationProblem(F, fill(2.0, 3), (r = 0.5,), (@optic _.r); J)
    opts = NewtonPar(tol = 1e-12)
    sol = @test_deprecated BK.solve(prob, Newton(), opts; callback = ScaledToleranceVerdict(1e-3))
    @test BK.converged(sol)
    @test 1e-12 < last(sol.residuals) < 1e-3 * maximum(abs, sol.u)
    @test BK._converged(false, NewtonConverged())
    @test BK._verdict_and(NewtonConverged(), true) === NewtonConverged()
    @test BK._verdict_and(NewtonConverged(), false) === false
end

# low precision and `Multiple` pass the convergence test through
let
    F(x, p) = @. x^3 / 3 + x - p.r
    J(x, p) = diagm(0 => 1 .+ x .^ 2)
    prob = BifurcationProblem(F, fill(2f0, 3), (r = 0.5f0,), (@optic _.r); J)
    opts = NewtonPar(tol = Inf32, max_iterations = 10, convergence_test = ScaledTolerance(1f-3))
    sol = BK.solve(prob, Newton(), opts)
    @test BK.converged(sol)
    @test eltype(sol.u) == Float32

    unreachable = ContinuationPar(p_min = 0f0, p_max = 1f0, ds = 0.05f0, max_steps = 200, detect_bifurcation = 0, newton_options = opts)
    floored = continuation(prob, Multiple(PALC(θ = 0.5f0), copy(prob.u0), 0.01f0, 3; dsfact = 1.5f0), unreachable)
    @test maximum(floored.param) >= 1
end
