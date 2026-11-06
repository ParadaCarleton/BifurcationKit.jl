module NonlinearSolveExt
    # Correctors of continuation by NonlinearSolve.jl, see `NonlinearSolveCorrector`.
    using BifurcationKit
    using NonlinearSolve: NonlinearSolveBase, GeneralizedFirstOrderAlgorithm, LinearSolve, SciMLBase
    using BorderedLinearSolve: BorderedLinearSolve, BorderedLinearProblem
    using RecursiveArrayTools: ArrayPartition
    using Accessors: @set
    import LinearAlgebra as LA
    const BK = BifurcationKit
    const SciMLOperators = SciMLBase.SciMLOperators

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    function check_matrix(J)
        if !(J isa AbstractMatrix)
            throw(ArgumentError("a dense linear solver needs a Jacobian which is an `AbstractMatrix`, got $(typeof(J)). Use a matrix-free linear solver such as `KrylovJL_GMRES`."))
        end
        return J
    end

    """
    The Jacobian `J` (and `∂ₚF` for PALC) at the iterate `point`, built once per iterate however many products a matrix-free linear solver takes. It is mutable because it is overwritten at each new iterate of a solve; a memo that is not `valid` holds the types of a first iterate and no iterate.
    """
    mutable struct JacobianMemo{Tu, TJ, Tdp}
        point::Tu
        J::TJ
        dFdp::Tdp
        valid::Bool
    end

    JacobianMemo(point, J, dFdp) = JacobianMemo(point, J, dFdp, true)

    "A memo of the types of `memo` that holds no iterate yet, so that no Jacobian is built before NonlinearSolve asks for one."
    stale_like(memo::JacobianMemo{Tu, TJ, Tdp}) where {Tu, TJ, Tdp} = JacobianMemo{Tu, TJ, Tdp}(memo.point, memo.J, memo.dFdp, false)

    remembers(memo::JacobianMemo, point) = memo.valid && memo.point == point

    function remember!(memo::JacobianMemo, point, J, dFdp)
        memo.point, memo.J, memo.dFdp, memo.valid = BK._copy(point), J, dFdp, true
        return memo
    end

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Solves of `F(x, p0) = 0`. The parameter of the NonlinearProblem is a `FixedContext`.
    struct FixedContext{Tprob, Tpar, Tmemo}
        prob::Tprob
        par::Tpar
        memo::Tmemo
    end

    # without a template context, the memo holds the Jacobian at the starting point `x0`; with one, only its types
    fixed_memo(::Nothing, prob, par, x0) = JacobianMemo(BK._copy(x0), BK.jacobian(prob, x0, par), nothing)
    fixed_memo(template::FixedContext, ::Any, ::Any, ::Any) = stale_like(template.memo)
    fixed_context(prob, par, x0, template) = FixedContext(prob, par, fixed_memo(template, prob, par, x0))

    function iterate_jacobian!(ctx::FixedContext, x)
        if !remembers(ctx.memo, x)
            remember!(ctx.memo, x, BK.jacobian(ctx.prob, x, ctx.par), nothing)
        end
        return ctx.memo
    end

    fixed_residual(x, ctx::FixedContext) = BK.residual(ctx.prob, x, ctx.par)
    fixed_jacobian(x, ctx::FixedContext) = check_matrix(iterate_jacobian!(ctx, x).J)
    # the product with the Jacobian at the iterate `x`: an iterative linear solver gets it as NonlinearSolve's operator, bound to the iterate at every step
    fixed_jvp(v, x, ctx::FixedContext) = BK.apply(iterate_jacobian!(ctx, x).J, v)

    BK.corrector_jacobian(ctx::FixedContext, x) = iterate_jacobian!(ctx, x).J
    BK.corrector_matrix(::FixedContext, x, state_matrix::AbstractMatrix) = state_matrix

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Solves of the bordered system of PALC. The unknown is `ArrayPartition(x, [p])`: the state `x` (any `AbstractArray`) and the parameter `p` stay separate.
    palc_state(w::ArrayPartition) = w.x[1]
    palc_parameter(w::ArrayPartition) = only(w.x[2])

    # Norm of the residual `[F; N]` of the bordered system, `N` being the arclength constraint
    struct BorderedNorm{Tn}
        normN::Tn
    end
    (bn::BorderedNorm)(r::ArrayPartition) = max(bn.normN(palc_state(r)), abs(palc_parameter(r)))

    """
    The Jacobian `[J ∂ₚF; row' c]` of the bordered system of PALC for a Jacobian `J` which is a matrix, as an operator on `ArrayPartition(x, [p])`, the way NonlinearSolve.jl takes a Jacobian which is not a matrix. Its linear systems are solved by BorderedLinearSolve.jl (`BorderedSolve`), which only solves with `J` and never forms the bordered matrix.

    - `J`: the Jacobian `∂ₓF` of the problem, an `AbstractMatrix`,
    - `dFdp`: `∂ₚF`,
    - `row`: the row of the arclength constraint acting on `x`, `θ A τu` for the weighted dot product `A`,
    - `c`: the coefficient of `p` in the arclength constraint, `(1 - θ) τp`.

    It is mutable because NonlinearSolve keeps a copy of the Jacobian (`jac_prototype`), into which it copies (`copyto!`) a new one.
    """
    mutable struct BorderedJacobian{Tt, TJ, Tdp, Trow, Tc} <: SciMLOperators.AbstractSciMLOperator{Tt}
        J::TJ
        dFdp::Tdp
        row::Trow
        c::Tc
    end

    function BorderedJacobian(J, dFdp, row, c)
        Tt = promote_type(eltype(dFdp), eltype(row), typeof(c))
        return BorderedJacobian{Tt, typeof(J), typeof(dFdp), typeof(row), typeof(c)}(J, dFdp, row, c)
    end

    function Base.copyto!(dst::BorderedJacobian, src::BorderedJacobian)
        dst.J, dst.dFdp, dst.row, dst.c = src.J, src.dFdp, src.row, src.c
        return dst
    end
    Base.copy(A::BorderedJacobian) = BorderedJacobian(A.J, A.dFdp, A.row, A.c)
    Base.similar(A::BorderedJacobian) = copy(A)
    Base.size(A::BorderedJacobian) = (length(A.dFdp) + 1, length(A.dFdp) + 1)
    Base.size(A::BorderedJacobian, i::Integer) = size(A)[i]
    Base.eltype(::BorderedJacobian{Tt}) where {Tt} = Tt
    # the products with the transpose of a trust region's gradient
    Base.adjoint(A::BorderedJacobian) = BorderedJacobian(adjoint(A.J), A.row, A.dFdp, conj(A.c))
    Base.transpose(A::BorderedJacobian) = BorderedJacobian(transpose(A.J), conj(A.row), conj(A.dFdp), A.c)
    SciMLOperators.islinear(::BorderedJacobian) = true
    SciMLOperators.has_adjoint(::BorderedJacobian) = true

    function LA.mul!(y::ArrayPartition, A::BorderedJacobian, v::ArrayPartition)
        vx, vp = palc_state(v), palc_parameter(v)
        palc_state(y) .= A.J * vx .+ A.dFdp .* vp
        y.x[2] .= LA.dot(A.row, vx) + A.c * vp
        return y
    end

    function LA.mul!(y::ArrayPartition, A::BorderedJacobian, v::ArrayPartition, α::Number, β::Number)
        y .= α .* (A * v) .+ β .* y
        return y
    end

    Base.:*(A::BorderedJacobian, v::ArrayPartition) = LA.mul!(similar(v), A, v)

    bordered_problem(A::BorderedJacobian, b::ArrayPartition) = BorderedLinearProblem(A.J, A.dFdp, A.row, A.c, palc_state(b), palc_parameter(b))

    """
    The linear solver of NonlinearSolve for a `BorderedJacobian` (`LinearSolve.LinearSolveFunction(BorderedSolve(bls))`): it solves the bordered system with the algorithm `bls` of BorderedLinearSolve.jl.
    """
    struct BorderedSolve{Tbls}
        bls::Tbls
    end

    # The cache of `bls` lives in the LinearSolve cache, so the factorization of `J` (or the preconditioner of an iterative solver) is kept while NonlinearSolve reuses the Jacobian.
    function LinearSolve.init_cacheval(alg::LinearSolve.LinearSolveFunction{<:BorderedSolve}, A::BorderedJacobian, b::ArrayPartition, args...)
        return SciMLBase.init(bordered_problem(A, b), alg.solve_func.bls)
    end

    """
    The cache of `bls` for `problem`. A `BorderingBLS` cache keeps the factorization of `J` while NonlinearSolve reuses the Jacobian (`isfresh = false`) and refactors when NonlinearSolve evaluated a new one (`isfresh`) or `J` is another object, so a `J` rewritten in place is not read through a stale factorization; the cache of any other algorithm (`DirectBLS`) holds the problem alone and is built again.
    """
    function refreshed_cache(cache::BorderedLinearSolve.BorderedCache, problem, bls, isfresh)
        if isfresh || problem.J !== cache.prob.J
            # assigning `A` marks the LinearSolve cache for refactorization and keeps its workspace
            cache.inner.A = BorderedLinearSolve._flatten_operator(problem.J, problem.f)
        end
        cache.prob = problem
        return cache
    end
    refreshed_cache(::Any, problem, bls, isfresh) = SciMLBase.init(problem, bls)

    function (bs::BorderedSolve)(A::BorderedJacobian, b::ArrayPartition, u, p, isfresh, Pl, Pr, cache; kwargs...)
        # the arclength row of the current corrector (`p.p`, its `PALCCache`): a Jacobian reused across correctors carries over `J` and `∂ₚF` only
        ctx = p.p
        problem = BorderedLinearProblem(A.J, A.dFdp, ctx.row, ctx.c, palc_state(b), palc_parameter(b))
        sol = SciMLBase.solve!(refreshed_cache(cache, problem, bs.bls, isfresh))
        palc_state(u) .= sol.v
        u.x[2] .= sol.σ
        return u
    end

    with_linsolve(descent::NonlinearSolveBase.NewtonDescent, linsolve) = @set descent.linsolve = linsolve
    with_linsolve(descent::NonlinearSolveBase.Dogleg, linsolve) = @set descent.newton_descent = with_linsolve(descent.newton_descent, linsolve)

    # `alg` with the linear solver of its Newton step replaced by BorderedLinearSolve's `bls`, which solves with the `linsolve` of `alg` by default
    BK._palc_algorithm(alg::GeneralizedFirstOrderAlgorithm, bls) = palc_algorithm(alg, alg.descent, bls)
    # an algorithm which can still be the corrector of Natural and `solve`
    BK._palc_algorithm(alg, bls) = nothing

    # the descents whose step solves a linear system with the Jacobian, here the bordered one of PALC
    function palc_algorithm(alg, descent::Union{NonlinearSolveBase.NewtonDescent, NonlinearSolveBase.Dogleg}, bls)
        # BEC+1 (one correction sweep) keeps the bordered solve accurate at a fold, where `J` is singular
        bls = something(bls, BorderedLinearSolve.BorderingBLS(; inner = NonlinearSolveBase.get_linear_solver(descent), k = 1))
        return @set alg.descent = with_linsolve(descent, LinearSolve.LinearSolveFunction(BorderedSolve(bls)))
    end
    palc_algorithm(alg, descent, bls) = nothing

    function check_palc_algorithm(corrector::NonlinearSolveCorrector)
        if isnothing(corrector.palc_alg)
            throw(ArgumentError("`NonlinearSolveCorrector` solves the bordered system of PALC with BorderedLinearSolve.jl in the Newton step of a first-order algorithm (`NewtonDescent` or `Dogleg` descent, as in `NewtonRaphson` or `TrustRegion`), got $(nameof(typeof(corrector.alg)))."))
        end
        return corrector.palc_alg
    end

    """
    The data of one PALC corrector, the NonlinearSolve parameter of its bordered problem: the solution `z0`, the tangent `τ0` and the step `ds` of the step, the dot product `dotθ` and weight `θ` of the arclength constraint. It holds no continuation state, so its type depends on the problem alone.
    """
    struct PALCContext{Tprob, Tpar, Tz0, Tτ0, Tds, Tdot, Tθ, Trow, Tc, Tmemo}
        prob::Tprob
        par::Tpar
        z0::Tz0
        τ0::Tτ0
        ds::Tds
        dotθ::Tdot
        θ::Tθ
        "`θ A τu` for the weighted dot product `A` of `dotθ`, the row of the arclength constraint acting on `x`"
        row::Trow
        "`(1 - θ) τp`, the coefficient of `p` in the arclength constraint"
        c::Tc
        "the `JacobianMemo` of the last iterate"
        memo::Tmemo
    end

    # `∂ₚF` from the jet of the problem (`R01` of `BifurcationProblem`, automatic differentiation by default); problems without a jet (minimally augmented, periodic orbits) by finite differences
    palc_dFdp(prob::BK.AbstractAllJetBifProblem, x, par) = BK.R01(prob, x, par)
    palc_dFdp(prob, x, par) = BK.R01(BK.FiniteDifferences(), prob, x, par)

    # without a template context, the memo holds the Jacobian at the predictor `w0`; with one, only its types
    function palc_memo(::Nothing, prob, par, w0)
        x, p = palc_state(w0), palc_parameter(w0)
        par_p = BK.set(par, BK.getlens(prob), p)
        return JacobianMemo(BK._copy(w0), BK.jacobian(prob, x, par_p), palc_dFdp(prob, x, par_p))
    end
    palc_memo(template::PALCContext, ::Any, ::Any, ::Any) = stale_like(template.memo)

    function palc_context(prob, z0, τ0, ds, dotθ, θ, w0::ArrayPartition, template)
        par = BK.getparams(prob)
        row = BK._copy(τ0.u) .* θ
        applyξu! = BK._get_apply_dot(dotθ)
        if !isnothing(applyξu!)
            applyξu!(row)
        end
        return PALCContext(prob, par, z0, τ0, ds, dotθ, θ, row, (one(θ) - θ) * τ0.p, palc_memo(template, prob, par, w0))
    end

    function palc_residual(w::ArrayPartition, ctx::PALCContext)
        x, p = palc_state(w), palc_parameter(w)
        Fx = BK.residual(ctx.prob, x, BK.set(ctx.par, BK.getlens(ctx.prob), p))
        Nx = BK.arc_length_eq(ctx.dotθ, x, ctx.z0.u, p - ctx.z0.p, ctx.τ0.u, ctx.τ0.p, ctx.θ, ctx.ds)
        return ArrayPartition(Fx, [Nx])
    end

    function iterate_jacobian!(ctx::PALCContext, w::ArrayPartition)
        if !remembers(ctx.memo, w)
            x, p = palc_state(w), palc_parameter(w)
            par = BK.set(ctx.par, BK.getlens(ctx.prob), p)
            remember!(ctx.memo, w, BK.jacobian(ctx.prob, x, par), palc_dFdp(ctx.prob, x, par))
        end
        return ctx.memo
    end

    function palc_jacobian(w::ArrayPartition, ctx::PALCContext)
        (; J, dFdp) = iterate_jacobian!(ctx, w)
        return BorderedJacobian(check_matrix(J), dFdp, ctx.row, ctx.c)
    end

    # A matrix-free `J` (a function `dx -> J dx`) gives NonlinearSolve the product with `[J ∂ₚF; row' c]` at the iterate `w`, which NonlinearSolve binds to the iterate at every step:
    # each step is a Newton step, and `jacobian_reuse` keeps only the preconditioner, built by the `precs` of the iterative linear solver for the whole bordered system.
    # The iterative linear solver solves the whole bordered system: BorderedLinearSolve keeps its inner solver's cache for one operator, so each iterate would rebuild the preconditioner.
    function palc_jvp(v::ArrayPartition, w::ArrayPartition, ctx::PALCContext)
        (; J, dFdp) = iterate_jacobian!(ctx, w)
        vx, vp = palc_state(v), palc_parameter(v)
        return ArrayPartition(BK.apply(J, vx) .+ dFdp .* vp, [LA.dot(ctx.row, vx) + ctx.c * vp])
    end

    BK.corrector_jacobian(ctx::PALCContext, w::ArrayPartition) = iterate_jacobian!(ctx, w).J

    function BK.corrector_matrix(ctx::PALCContext, w::ArrayPartition, state_matrix::AbstractMatrix)
        dFdp = iterate_jacobian!(ctx, w).dFdp
        return vcat(hcat(state_matrix, vec(dFdp)), hcat(LA.adjoint(vec(ctx.row)), ctx.c))
    end

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "Whether the callback verdict `verdict` or a residual `res` within `tol` ends Newton's method, as in `newton_palc`."
    ends_newton(verdict, res, tol) = !BK._continues(verdict) || res <= tol

    "The call of the step loop for the bordered problem of PALC: the callback of `newton_palc` at an iterate."
    struct PALCIterate{Tcallback, Tkw, Tnorm, Tz0, Tzp, Tds, Tcp, Topt, Tres}
        callback::Tcallback
        kwargs::Tkw
        norm::Tnorm
        z0::Tz0
        z_pred::Tzp
        ds::Tds
        contparams::Tcp
        options::Topt
        residuals::Tres
    end

    function (it::PALCIterate)(w::ArrayPartition, residual::ArrayPartition, step::Int, tol)::Bool
        x, p = BK._copy(palc_state(w)), palc_parameter(w)
        res = it.norm(residual)
        push!(it.residuals, res)
        # `itlinear` marks a Newton step for `PALCStepCheck`, which runs the quality tests of `CorrectorQuality` on it; NonlinearSolve does not report the iterations of its linear solver
        verdict = it.callback((; x, res_f = palc_state(residual), residual = res, step, itlinear = 0, it.contparams, it.z0, it.z_pred, it.ds, p, it.residuals, it.options); fromNewton = false, it.kwargs...)
        return ends_newton(verdict, res, tol)
    end

    "The call of the step loop for a Newton solve at a fixed parameter: the callback of `newton` at an iterate."
    struct FixedIterate{Tcallback, Tkw, Tnorm, Tx0, Topt, Tres}
        callback::Tcallback
        kwargs::Tkw
        norm::Tnorm
        x0::Tx0
        options::Topt
        residuals::Tres
    end

    function (it::FixedIterate)(x, fx, step::Int, tol)::Bool
        res = it.norm(fx)
        push!(it.residuals, res)
        verdict = it.callback((; x = BK._copy(x), fx, J = nothing, residual = res, step, it.options, it.x0, it.residuals); fromNewton = true, it.kwargs...)
        return ends_newton(verdict, res, tol)
    end

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    A NonlinearSolve cache with the types of the unknown and the context it was built for (`proto`, whose memo gives the types of the memos of later contexts), the options (`key`) and the Jacobian kind (`matrix`). A solve with other types, tolerance, number of iterations, bounds or norm builds a temporary cache instead.
    """
    struct SolverCache{Tcache, Tkey, Tu, Tctx}
        cache::Tcache
        key::Tkey
        proto::Tctx
        matrix::Bool
    end

    "The caches of a continuation, in its state: the bordered problem of PALC and the problem at fixed parameter, each `nothing` when the continuation has no such solve."
    struct NonlinearSolveCaches{Tpalc, Tfixed}
        palc::Tpalc
        fixed::Tfixed
    end

    solver_key(options::NewtonPar, u0, bounds, norm) = (; tol = options.tol, max_iterations = options.max_iterations, length = length(u0), bounds, norm)

    template_context(::Nothing) = nothing
    template_context(slot::SolverCache) = slot.proto
    palc_slot(::Nothing) = nothing
    palc_slot(solver::NonlinearSolveCaches) = solver.palc
    fixed_slot(::Nothing) = nothing
    fixed_slot(solver::NonlinearSolveCaches) = solver.fixed

    fits(::Nothing, u0, ctx, key, matrix::Bool) = false
    function fits(slot::SolverCache{Tcache, Tkey, Tu, Tctx}, u0, ctx, key, matrix::Bool) where {Tcache, Tkey, Tu, Tctx}
        return u0 isa Tu && ctx isa Tctx && slot.matrix == matrix && slot.key == key
    end

    # `p ∈ [p_min, p_max]` as the box of the parameter of PALC's unknown, for an algorithm that handles bounds
    function palc_bounds(alg, w0::ArrayPartition, p_min, p_max)
        if !SciMLBase.allowsbounds(alg)
            return nothing
        end
        x = palc_state(w0)
        lb = ArrayPartition(fill(oftype(p_min, -Inf), size(x)), [p_min])
        ub = ArrayPartition(fill(oftype(p_max, Inf), size(x)), [p_max])
        return (; lb, ub)
    end

    # `matrix`: the Jacobian is the operator `jacobian(u0, ctx)`, given as `jac_prototype`; otherwise NonlinearSolve takes `jacobian` for a dense linear solver and builds its own operator from `jvp` for an iterative one
    function build_solver_cache(corrector::NonlinearSolveCorrector, alg, residual, jacobian, jvp, u0, ctx, options::NewtonPar, bounds, norm, matrix::Bool)
        if matrix
            f = SciMLBase.NonlinearFunction{false}(residual; jac = jacobian, jac_prototype = jacobian(u0, ctx))
        else
            f = SciMLBase.NonlinearFunction{false}(residual; jac = jacobian, jvp)
        end
        problem = SciMLBase.NonlinearProblem(f, u0, ctx; something(bounds, (;))...)
        cache = SciMLBase.init(problem, alg;
            abstol = options.tol,
            reltol = zero(options.tol),
            maxiters = options.max_iterations,
            termination_condition = NonlinearSolveBase.AbsNormTerminationMode(norm),
            corrector.solve_kwargs...)
        key = solver_key(options, u0, bounds, norm)
        return SolverCache{typeof(cache), typeof(key), typeof(u0), typeof(ctx)}(cache, key, ctx, matrix)
    end

    """
    Whether `ctx` starts at the point where the last solve of `cache` ended: the step was accepted, so its Jacobian is one step old.
    A rejected or failed step ended elsewhere (the retry starts at the older point with a smaller step), and a fixed-parameter solve has no previous step.
    """
    continues_accepted_step(cache, ctx::PALCContext) = begin
        endpoint = NonlinearSolveBase.get_u(cache)
        palc_state(endpoint) == ctx.z0.u && palc_parameter(endpoint) == ctx.z0.p
    end
    continues_accepted_step(cache, ctx) = false

    "Solve from `u0` in the context `ctx` with the cache of `slot`, started again from `u0` (it may check the initial iterate)."
    function run_solver!(slot::SolverCache, corrector::NonlinearSolveCorrector, u0, ctx, call, tol)
        # only passed when asked, so a NonlinearSolve without the keyword works with the default
        if corrector.reuse_jacobian && continues_accepted_step(slot.cache, ctx)
            SciMLBase.reinit!(slot.cache, u0; p = ctx, reuse_jacobian = true)
        else
            SciMLBase.reinit!(slot.cache, u0; p = ctx)
        end
        return solve_steps!(slot.cache, call, tol)
    end

    # the cache of `slot` when it fits, a temporary cache otherwise; two branches so that each is type stable
    function solve_cache(slot, corrector::NonlinearSolveCorrector, alg, residual, jacobian, jvp, u0, ctx, options::NewtonPar, call, bounds, norm, matrix::Bool)
        if fits(slot, u0, ctx, solver_key(options, u0, bounds, norm), matrix)
            return run_solver!(slot, corrector, u0, ctx, call, options.tol)
        end
        temporary = build_solver_cache(corrector, alg, residual, jacobian, jvp, u0, ctx, options, bounds, norm, matrix)
        return run_solver!(temporary, corrector, u0, ctx, call, options.tol)
    end

    """
    Step `cache` until it terminates or `call` ends Newton's method, showing `call` each iterate once, then finish as `solve!` does.
    """
    function solve_steps!(cache, call, tol)
        step = 0
        while NonlinearSolveBase.not_terminated(cache)
            SciMLBase.step!(cache)
            step += 1
            if call(NonlinearSolveBase.get_u(cache), NonlinearSolveBase.get_fu(cache), step, tol)
                cache.retcode = SciMLBase.ReturnCode.Success
                cache.force_stop = true
            end
        end
        # a terminated cache only gets its return code, residual and solution assembled
        return SciMLBase.solve!(cache)
    end

    # PALC borders a matrix Jacobian with BorderedLinearSolve; a matrix-free one goes to the algorithm as is
    function palc_setup(corrector::NonlinearSolveCorrector, ctx::PALCContext)
        if ctx.memo.J isa AbstractMatrix
            return (; alg = check_palc_algorithm(corrector), jvp = nothing, matrix = true)
        end
        return (; alg = corrector.alg, jvp = palc_jvp, matrix = false)
    end

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # The caches are built at the first point of the branch, with the types of the solves of the continuation.
    function BK._solver_caches(corrector::NonlinearSolveCorrector, it::BK.AbstractContinuationIterable, z0, palc)
        x0 = BK._copy(z0.u)
        if !(x0 isa AbstractArray)
            return nothing
        end
        prob, options = it.prob, BK.getcontparams(it).newton_options
        fixed_ctx = fixed_context(prob, BK.getparams(prob), x0, nothing)
        fixed = build_solver_cache(corrector, corrector.alg, fixed_residual, fixed_jacobian, fixed_jvp, x0, fixed_ctx, options, nothing, it.normC, false)
        return NonlinearSolveCaches(palc_solver_cache(corrector, it, z0, x0, palc), fixed)
    end

    palc_solver_cache(::NonlinearSolveCorrector, ::Any, ::Any, ::Any, ::Nothing) = nothing
    function palc_solver_cache(corrector::NonlinearSolveCorrector, it, z0, x0, palc::Tuple)
        dotθ, θ = palc
        contparams = BK.getcontparams(it)
        w0 = ArrayPartition(BK._copy(x0), [z0.p])
        # the solution stands in for the tangent: a nonzero arclength row keeps a preconditioner built from this context nonsingular
        ctx = palc_context(it.prob, z0, BK._copy(z0), contparams.ds, dotθ, θ, w0, nothing)
        (; alg, jvp, matrix) = palc_setup(corrector, ctx)
        bounds = palc_bounds(alg, w0, contparams.p_min, contparams.p_max)
        return build_solver_cache(corrector, alg, palc_residual, palc_jacobian, jvp, w0, ctx, contparams.newton_options, bounds, BorderedNorm(it.normC), matrix)
    end

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    function BK._newton_nonlinearsolve(corrector::NonlinearSolveCorrector,
                                       solver::Union{Nothing, NonlinearSolveCaches},
                                       prob::BK.AbstractBifurcationProblem,
                                       x0, params0, options::NewtonPar;
                                       normN = LA.norm,
                                       callback = BK.cb_default,
                                       kwargs...)
        (; tol) = options
        x0 = BK._copy(x0)
        ctx = fixed_context(prob, params0, x0, template_context(fixed_slot(solver)))
        fx = BK.residual(prob, x0, params0)
        res = normN(fx)
        compute = callback((; x = x0, fx, J = nothing, residual = res, step = 0, options, x0, residuals = [res]); fromNewton = true, kwargs...)
        if !BK._continues(compute) || res <= tol
            flag = BK._converged(res <= tol, callback((; x = x0, fx, residual = res, step = 0, options, x0, residuals = [res]); fromNewton = true, kwargs...))
            return NonLinearSolution(x0, prob, [res], flag, 0, 0)
        end
        residuals = [res]
        call = FixedIterate(callback, NamedTuple(kwargs), normN, x0, options, residuals)
        sol = solve_cache(fixed_slot(solver), corrector, corrector.alg, fixed_residual, fixed_jacobian, fixed_jvp, x0, ctx, options, call, nothing, normN, false)
        resend = normN(sol.resid)
        step = sol.stats.nsteps
        flag = BK._converged(SciMLBase.successful_retcode(sol) & (resend <= tol),
               callback((; x = sol.u, fx = sol.resid, residual = resend, step, options, x0, residuals); fromNewton = true, kwargs...))
        return NonLinearSolution(sol.u, prob, residuals, flag, step, sol.stats.nsolve)
    end

    function BK._newton_palc_nonlinearsolve(corrector::NonlinearSolveCorrector,
                                            solver::Union{Nothing, NonlinearSolveCaches},
                                            iter::BK.AbstractContinuationIterable,
                                            state::BK.AbstractContinuationState,
                                            dotθ = BK.getdot(iter);
                                            normN = LA.norm,
                                            callback = BK.cb_default,
                                            kwargs...)
        prob = iter.prob
        contparams = BK.getcontparams(iter)
        z0, z_pred = BK.getsolution(state), state.z_pred
        (; tol, linsolver) = contparams.newton_options
        (; p_min, p_max) = contparams
        x_pred = z_pred.u
        if !(x_pred isa AbstractArray)
            throw(ArgumentError("`NonlinearSolveCorrector` needs the state to be an `AbstractArray`, got $(typeof(x_pred))."))
        end

        w0 = ArrayPartition(BK._copy(x_pred), [z_pred.p])
        ctx = palc_context(prob, z0, state.τ, state.ds, dotθ, BK.getθ(iter), w0, template_context(palc_slot(solver)))
        bnorm = BorderedNorm(normN)
        residual0 = palc_residual(w0, ctx)
        res = bnorm(residual0)
        residuals = [res]
        options = (; linsolver)
        compute = callback((; x = x_pred, res_f = palc_state(residual0), residual = res, step = 0, contparams, z0, z_pred, ds = state.ds, p = z_pred.p, residuals, options); fromNewton = false, kwargs...)
        if !BK._continues(compute) || res <= tol
            x, p = x_pred, z_pred.p
            flag = BK._converged(res <= tol, callback((; x, res_f = palc_state(residual0), residual = res, step = 0, contparams, z0, z_pred, ds = state.ds, p, residuals, options, closing = true); fromNewton = false, kwargs...))
            return NonLinearSolution(BorderedArray(BK._copy(x), p), prob, residuals, flag, 0, 0)
        end
        (; alg, jvp, matrix) = palc_setup(corrector, ctx)
        call = PALCIterate(callback, NamedTuple(kwargs), bnorm, z0, z_pred, state.ds, contparams, options, residuals)
        sol = solve_cache(palc_slot(solver), corrector, alg, palc_residual, palc_jacobian, jvp, w0, ctx, contparams.newton_options, call, palc_bounds(alg, w0, p_min, p_max), bnorm, matrix)
        x, p = BK._copy(palc_state(sol.u)), palc_parameter(sol.u)
        resend = bnorm(sol.resid)
        step = sol.stats.nsteps
        flag = BK._converged(SciMLBase.successful_retcode(sol) & (resend <= tol),
               callback((; x, res_f = palc_state(sol.resid), residual = resend, step, contparams, z0, z_pred, ds = state.ds, p, residuals, options, closing = true); fromNewton = false, kwargs...))
        return NonLinearSolution(BorderedArray(x, p), prob, residuals, flag, step, sol.stats.nsolve)
    end
end
