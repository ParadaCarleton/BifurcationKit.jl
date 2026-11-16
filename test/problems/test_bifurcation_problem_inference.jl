using BifurcationKit, Test

residual(x, p) = x .- p
residual!(out, x, p) = (out .= x .- p)
out_of_place(x::Vector{Float32}, level::Float32) = BifurcationProblem(residual, x, level; inplace = false)
in_place(x::Vector{Float32}, level::Float32) = BifurcationProblem(residual!, x, level; inplace = true)

@testset "BifurcationProblem infers a concrete type from a literal inplace" begin
    for build in (out_of_place, in_place)
        inferred = only(Base.return_types(build, (Vector{Float32}, Float32)))
        @test isconcretetype(inferred)
    end
end

"A callable of four arguments, which fails the arity check `_isinplace` runs on a vector field."
four_arguments(x, p, a, b) = x

@testset "an explicit inplace skips the vector field's arity check" begin
    @test_throws ErrorException BifurcationProblem(four_arguments, [1.0], 0.0)
    @test BifurcationProblem(four_arguments, [1.0], 0.0; inplace = false) isa BifurcationProblem
end
