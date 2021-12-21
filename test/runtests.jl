#Encoding: UTF-8 -*-

using Test


##
## Run tests:
## Enter repo main directory, execute julia, and call `] test` (] enters Pkg mode)
## cd $PICFUN 
## julia --project=.
## `] test`
@testset "all" begin
    @testset "solvers" begin
        include("solvers.jl")
    end # testset solvers

    @testset "diffable" begin
        include("diffable.jl")
    end

end #testset all
# End of file runtests.jl
