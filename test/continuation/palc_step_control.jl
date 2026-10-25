using Test, BifurcationKit, LinearAlgebra
const BK = BifurcationKit
####################################################################################################
# the pure rules of CorrectorQuality
let
    tol = 1e-8
    control = CorrectorQuality(max_distance = 0.4)   # Program 1 of Allgower and Georg: dmax = 0.4, ctmax = 0.6, eta = 0.1
    @test (control.max_distance, control.max_contraction, control.max_growth, control.η) == (0.4, 0.6, 1.1, 0.1)
    # the distance test is off unless a threshold is given, the contraction test is on
    @test CorrectorQuality().max_distance === nothing
    @test CorrectorQuality().max_contraction == 0.6

    quality = BK.StepQuality(control)
    # no Newton step: the largest growth
    @test BK.growth_factor(control, quality) ≈ 1.1

    # first step: rejected from max_distance on
    @test ~BK.rejects_step(control, quality, 0.39, tol)
    @test BK.rejects_step(control, quality, 0.4, tol)
    # a short first step still allows the largest growth, a long one slows it down by sqrt(d / max_distance) * max_growth
    @test BK.growth_factor(control, BK.observe(control, quality, 0.1, tol)) ≈ 1.1
    @test BK.growth_factor(control, BK.observe(control, quality, 0.3, tol)) ≈ inv(1.1 * sqrt(0.3 / 0.4))

    # later steps: rejected when d_k / (d_{k-1} + tol η) exceeds max_contraction
    after_first = BK.observe(control, quality, 0.3, tol)
    @test ~BK.rejects_step(control, after_first, 0.18, tol)
    @test BK.rejects_step(control, after_first, 0.19, tol)
    contraction = 0.1 / (0.3 + tol * 0.1)
    @test BK.contraction_rate(control, after_first, 0.1, tol) ≈ contraction
    after_second = BK.observe(control, after_first, 0.1, tol)
    # the deceleration is the largest one asked for by any step
    @test after_second.deceleration ≈ max(1.1 * sqrt(0.3 / 0.4), 1.1 * sqrt(contraction / 0.6))
    @test after_second.steps == 2

    # a test that is off neither rejects a step nor slows the next one
    off = CorrectorQuality(max_distance = nothing, max_contraction = nothing)
    @test ~BK.rejects_step(off, BK.StepQuality(off), 1e6, tol)
    @test ~BK.rejects_step(off, BK.observe(off, BK.StepQuality(off), 1e6, tol), 1e12, tol)
    @test BK.growth_factor(off, BK.observe(off, BK.observe(off, BK.StepQuality(off), 1e6, tol), 1e12, tol)) ≈ 1.1
    # the growth is capped by max_growth when no test asked for more deceleration
    @test BK.growth_capped(control, quality)
    @test BK.growth_capped(control, BK.observe(control, quality, 0.1, tol))
    @test ~BK.growth_capped(control, BK.observe(control, quality, 0.3, tol))
    # max_growth = nothing leaves ds to the Newton iteration count: no factor, tests still reject
    unfactored = CorrectorQuality(max_distance = 0.4, max_growth = nothing)
    @test BK.growth_factor(unfactored, BK.observe(unfactored, BK.StepQuality(unfactored), 0.3, tol)) === nothing
    @test BK.rejects_step(unfactored, BK.StepQuality(unfactored), 0.4, tol)
    # the distance test is off alone
    no_distance = CorrectorQuality()
    @test ~BK.rejects_step(no_distance, BK.StepQuality(no_distance), 1e6, tol)
    @test BK.rejects_step(no_distance, BK.observe(no_distance, BK.StepQuality(no_distance), 1.0, tol), 0.7, tol)

    # thresholds follow the float type of the continuation
    @test BK.CorrectorQuality{Float32}(control) isa CorrectorQuality{Float32}
    @test BK.CorrectorQuality{Float32}(no_distance).max_distance === nothing
    @test CorrectorQuality(max_distance = 0.4f0, max_contraction = 0.6f0, max_growth = 1.1f0, η = 0.1f0) isa CorrectorQuality{Float32}
    @test CorrectorQuality(max_contraction = 0.6f0, max_growth = 1.1f0, η = 0.1f0) isa CorrectorQuality{Float32}
    @test_throws AssertionError CorrectorQuality(max_growth = 1.0)
    @test_throws AssertionError CorrectorQuality(max_distance = 0.0)
    @test_throws AssertionError CorrectorQuality(max_contraction = -1.0)
end
####################################################################################################
# continuation with the step control: u^3 - u - (p - 1) = 0 has two folds, p ≈ 0.615 and p ≈ 1.385
let
    F(u, p) = @. u^3 - u - (p.λ - 1)
    make_problem(T) = BifurcationProblem(F, T[-1.3247179], (λ = zero(T),), (@optic _.λ))
    options(T; ds = 0.05, dsmax = 0.5) = ContinuationPar(ds = T(ds), dsmin = T(1e-6), dsmax = T(dsmax), a = T(0.5), p_min = zero(T), p_max = T(2), η = T(150), tol_stability = T(1e-10), dsmin_bisection = T(1e-16), tol_bisection_eigenvalue = T(1e-16), tol_param_bisection_event = T(1e-16), max_steps = 200, detect_fold = true, detect_bifurcation = 0, newton_options = NewtonPar(tol = sqrt(eps(T))))

    # off by default
    @test PALC().step_control === nothing

    for tangent in (Secant(), Bordered())
        br = continuation(make_problem(Float64), PALC(; tangent, step_control = CorrectorQuality()), options(Float64))
        # it follows the branch through both folds
        @test count(sp -> sp.type == :fold, br.specialpoint) == 2
        @test all(point -> norm(F(point.x, (λ = point.p,)), Inf) < 1e-7, br.sol)
        # ds grows by at most max_growth per step (the recorded ds is the one of the next step), whatever the rejections
        ds = abs.(br.branch.ds)
        # (the last step may end on the boundary p_max, where the Natural corrector is used)
        @test all(ds[2:(end - 1)] .<= 1.1 .* ds[begin:(end - 2)] .* (1 + 1e-12))
        # and it does grow when the corrector is good
        @test maximum(ds) > 2 * first(ds)
    end

    # without it ds follows the Newton iteration count, which grows it faster than max_growth
    legacy = continuation(make_problem(Float64), PALC(step_control = nothing), options(Float64))
    @test count(sp -> sp.type == :fold, legacy.specialpoint) == 2
    # the branch passes the sheet between the folds, |u| < 1/√3 (λ = u^3 - u + 1): the long corrector that would jump from
    # u ≈ -0.92 to the upper sheet has an iterate outside [p_min, p_max], which refuses the step, and a shorter step follows the curve
    @test any(point -> abs(point.x[1]) < 1 / sqrt(3), legacy.sol)
    legacy_ds = abs.(legacy.branch.ds)
    @test any(legacy_ds[2:(end - 1)] .> 1.1 .* legacy_ds[begin:(end - 2)])

    # with max_growth = nothing ds follows the Newton iteration count again, the tests still refuse steps
    unfactored = continuation(make_problem(Float64), PALC(step_control = CorrectorQuality(; max_distance = 0.4, max_growth = nothing)), options(Float64))
    @test count(sp -> sp.type == :fold, unfactored.specialpoint) == 2
    unfactored_ds = abs.(unfactored.branch.ds)
    @test any(unfactored_ds[2:(end - 1)] .> 1.1 .* unfactored_ds[begin:(end - 2)])

    # a short distance to the curve limits the step, the other test off
    only_distance(max_distance) = CorrectorQuality(; max_distance, max_contraction = nothing)
    loose = continuation(make_problem(Float64), PALC(step_control = only_distance(nothing)), options(Float64))
    tight = continuation(make_problem(Float64), PALC(step_control = only_distance(1e-3)), options(Float64))
    @test maximum(abs, tight.branch.ds) < maximum(abs, loose.branch.ds)
    @test length(tight.branch) > length(loose.branch)
    @test all(point -> norm(F(point.x, (λ = point.p,)), Inf) < 1e-7, tight.sol)

    # the continuation keeps the float type of the problem
    for T in (Float32, Float64)
        control = CorrectorQuality(; max_distance = T(0.4), max_contraction = T(0.6), max_growth = T(1.1), η = T(0.1))
        br = continuation(make_problem(T), PALC(θ = T(0.5), step_control = control), options(T))
        @test eltype(br.branch.ds) == T
        @test all(point -> point.p isa T, br.sol)
        @test count(sp -> sp.type == :fold, br.specialpoint) == 2
    end

    # the step control wraps the Newton callback of continuation, whose refusals still stop the corrector
    wall(state; fromNewton = true, kwargs...) = fromNewton || state.p <= 1
    walled = continuation(make_problem(Float64), PALC(step_control = CorrectorQuality()), options(Float64); callback_newton = wall)
    @test all(point -> point.p <= 1, walled.sol)
    @test maximum(point -> point.p, walled.sol) > 0.9
end
