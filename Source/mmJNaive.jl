import Pkg; 
Pkg.add("KernelAbstractions")
Pkg.add("CUDAKernels")
Pkg.add("CUDA")
Pkg.add("NVTX")
Pkg.add("General")
using KernelAbstractions, Test, CUDA, CUDAKernels
#, NVTX

if has_cuda_gpu()
    CUDA.allowscalar(false)
end

# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(a, b, c)
    i, j = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(c))
    for k = 1:size(a)[2]
        tmp_sum += a[i,k] * b[k, j]
    end

    c[i,j] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function matmul!(a, b, c)
    if size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    if isa(a, Array)
	#CUDA.@profile
        kernel! = matmul_kernel!(CPU(),4)
    else
	#CUDA.@profile
        kernel! = matmul_kernel!(CUDADevice(),256)
    end
    kernel!(a, b, c, ndrange=size(c)) 
end

dim = parse(Int, ARGS[1])
a = rand(dim, dim)

b = rand(dim, dim)
#b = inv(a)

c = zeros(dim, dim)

# beginning CPU tests, returns event
ev = matmul!(a,b,c)
wait(ev)

@test isapprox(c, a*b)

# beginning GPU tests
if has_cuda_gpu()
    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CuArray(c)

    ev = matmul!(d_a, d_b, d_c)
    wait(ev)

    @test isapprox(Array(d_c), a*b)
end

#print matrix c (result)
show(stdout, "text/plain", abs.(round.(c)))
