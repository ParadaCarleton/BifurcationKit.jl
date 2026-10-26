# using Revise#, Plots
using Test
using BifurcationKit, SparseArrays, LinearAlgebra, ForwardDiff
const BK = BifurcationKit
const FD = ForwardDiff

function Laplacian1D(N, lx, bc = :none)
    hx = 2lx / N
    Δ = spdiagm(0 => -2ones(N), 1 => ones(N - 1), -1 => ones(N - 1))
    Δ[1, end] = 1
    Δ[end, 1] = 1
    D = spdiagm(1 => ones(N - 1), -1 => -ones(N - 1))
    D[1, end] = -1
    D[end, 1] = 1
    D = D / (2hx)
    Δ = Δ / hx^2
    return Δ, D
end

# add to f the nonlinearity
@views function NL!(f, u, p)
    (; r, μ, ν, c3, c5, γ) = p
    n = div(length(u), 2)
    u1 = u[1:n]
    u2 = u[n + 1:2n]

    ua = u1.^2 .+ u2.^2

    f1 = f[1:n]
    f2 = f[n + 1:2n]

    f1 .+= @. r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1 + γ
    f2 .+= @. r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2

    return f
end

function Fcgl!(f, u, p, t = 0)
    mul!(f, p.Δ, u)
    NL!(f, u, p)
end

Fcgl(u, p, t = 0) = Fcgl!(similar(u), u, p)

# remark: I checked this against finite differences
@views function Jcgl(u, p)
    (; r, μ, ν, c3, c5, Δ) = p

    n = div(length(u), 2)
    u1 = u[1:n]
    u2 = u[n + 1:2n]

    ua = u1.^2 .+ u2.^2

    f1u = zero(u1)
    f2u = zero(u1)
    f1v = zero(u1)
    f2v = zero(u1)

    @. f1u = r - 2 * u1 * (c3 * u1 - μ * u2) - c3 * ua - 4 * c5 * ua * u1^2 - c5 * ua^2
    @. f1v = -ν - 2 * u2 * (c3 * u1 - μ * u2) + μ * ua - 4 * c5 * ua * u1 * u2
    @. f2u = ν - 2 * u1 * (c3 * u2 + μ * u1) - μ * ua - 4 * c5 * ua * u1 * u2
    @. f2v = r - 2 * u2 * (c3 * u2 + μ * u1) - c3 * ua - 4 * c5 * ua * u2^2 - c5 * ua^2

    jacdiag = vcat(f1u, f2v)

    Δ + spdiagm(0 => jacdiag, n => f1v, -n => f2u)
end
using Random, Accessors
####################################################################################################
"""EXPERIMENT: one run of test_wave.jl's TW branch and line-198 fold start with the eigensolver `eigls`."""
function probe_run(label, eigls)
    n = 50
    Δ, D = Laplacian1D(n, -pi, :Periodic)
    par_cgl = (
        r = 0.0,
        μ = 0.5,
        ν = 1.0,
        c3 = -1.0,
        c5 = 1.0,
        Δ = blockdiag(Δ, Δ),
        Db = blockdiag(D, D),
        γ = 0.0,
        δ = 1.0,
        N = 2n,
    )
    prob = BifurcationKit.BifurcationProblem(
        Fcgl,
        zeros(2n),
        par_cgl,
        (@optic _.r);
        J = Jcgl,
    )
    opt_newton = NewtonPar(
        tol = 1.0e-9,
        verbose = false,
        eigsolver = eigls,
        max_iterations = 20,
    )
    opts_br = ContinuationPar(
        dsmin = 0.001,
        dsmax = 0.15,
        ds = 0.001,
        p_max = 2.5,
        detect_bifurcation = 3,
        nev = 9,
        plot_every_step = 50,
        newton_options = opt_newton,
        max_steps = 30,
        n_inversion = 8,
        max_bisection_steps = 20,
    )
    br = continuation(prob, PALC(), opts_br, verbosity = 0)
    hopf = br.specialpoint[2]
    λs = br.eig[hopf.idx].eigenvals
    v1 = geteigenvector(eigls, br.eig[hopf.idx][2], hopf.ind_ev)
    v1 ./= norm(v1)
    v2 = geteigenvector(eigls, br.eig[hopf.idx][2], hopf.ind_ev - 2)
    v2 ./= norm(v2)
    uold = real.(hopf.x .+ v1 .+ v2)
    tw_model = BK.TWModel(
        re_make(prob, params = (par_cgl..., r = hopf.param - 0.01)),
        par_cgl.Db,
        copy(uold),
    )
    wave0 = vcat(uold, -0.99)
    opt_cont_br = ContinuationPar(
        p_min = -1.0,
        p_max = 1.0,
        newton_options = NewtonPar(tol = 1.0e-8),
        max_steps = 30,
        detect_bifurcation = 2,
        ds = -0.001,
    )
    @reset opt_cont_br.newton_options.eigsolver = BK.DefaultGEig(B = diagm(0 => vcat(
        ones(2n),
        0,
    )))
    br_tw = continuation(
        (@set tw_model.jacobian = BK.FullLU()),
        wave0,
        PALC(),
        opt_cont_br;
        verbosity = 0,
    )
    fold = try
        br_fold = continuation(
            br_tw,
            1,
            (@optic _.c3);
            jacobian_ma = BK.MinAug(),
            usehessian = false,
            verbosity = 0,
        )
        "ok $(length(br_fold))"
    catch exception
        "FAILED $(first(sprint(showerror, exception), 60))"
    end
    # Fourier content of the guess's complex amplitude A = u1 + i u2: a travelling wave has one of the modes ±k only
    A = (uold[1:n] .- hopf.x[1:n]) .+ im .* (uold[(n + 1):2n] .- hopf.x[(n + 1):2n])
    X = collect(LinRange(-pi, pi, n))
    modes = [k => round(abs(sum(A .* exp.(-im * k .* X))) / n, digits = 3) for k in -3:3]
    println("PROBE-GUESS $label | |A_k| $modes")
    # per step: unstable count, the eigenvalue nearest 0, and the largest |λ|
    for (step, eiginfo) in pairs(br_tw.eig)
        λ = eiginfo.eigenvals
        nearest = λ[argmin(abs.(λ))]
        println(
            "PROBE-EIG $label | step $step r $(round(br_tw.branch.param[step], digits = 4)) n_unstable $(br_tw.branch.n_unstable[step]) n_imag $(br_tw.branch.n_imag[step]) | nearest 0: $(round(nearest, sigdigits = 3)) | max |λ| $(round(maximum(abs, λ), sigdigits = 3)) | count $(length(λ)) | top real $(round.(sort(real.(λ), rev = true)[1:min(4, end)], sigdigits = 3))",
        )
    end
    sp = br_tw.specialpoint
    println(
        "PROBE $label | hopf ind_ev $(hopf.ind_ev) imag signs $(sign.(imag.(λs[hopf.ind_ev - 3:hopf.ind_ev]))) |v1'v2| $(round(abs(dot(v1, v2)), digits = 3)) | ‖u‖ $(round(norm(uold), digits = 3)) | tw speed0 $(round(br_tw.branch.param[1], digits = 4)) | sp types $([pt.type for pt in sp]) params $([round(pt.param, digits = 4) for pt in sp]) | fold $fold",
    )
    return nothing
end
let
    for seed in 1:4
        probe_run("seed $seed", EigArpack(1.0, :LM; v0 = randn(MersenneTwister(seed), 100)))
    end
end
