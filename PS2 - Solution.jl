###############################################################################
# PS 2 - Solutions
###############################################################################

using LinearAlgebra
using JuMP
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

using Plots

###############################################################################
# Problem 1: Iterative solver for nonlinear equations
###############################################################################
function fixed_point_solver(
    f::Function,
    x0::Real,
    α::Real;
    ϵ::Real=1e-6,
    maxiter::Int=1000
)
    """
    fixed_point_solver(f, x0, α; ϵ=1e-6, maxiter=1000)

    Solves f(x)=0 by considering g(x) = x + f(x) and iterating:
        x_{n+1} = (1 - α)*g(x_n) + α*x_n

    Returns (flag, x_sol, f_sol, diff, xhist, rhist).
    """
    g(x) = x + f(x)

    xhist = Float64[]
    rhist = Float64[]
    push!(xhist, x0)

    xcur = x0
    for iter in 1:maxiter
        xnext = (1 - α)*g(xcur) + α*xcur
        push!(xhist, xnext)
        push!(rhist, abs(xnext - xcur))


        if abs(xnext - xcur)/(1 + abs(xcur)) < ϵ

            return (0, xnext, f(xnext), abs(xnext - g(xnext)), xhist, rhist)
        end
        xcur = xnext
    end


    return (1, NaN, NaN, NaN, xhist, rhist)
end

###############################################################################
# Problem 2: Some linear algebra
###############################################################################
function exact_solution(α, β)
    """
    Returns the exact solution of A x = b (5x5 system).
    By manual derivation, x = [1,1,1,1,1].
    """
    return ones(5)
end

function solve_system(α, β)
    """
    Builds A, b, solves A x = b via backslash,
    returns (x_exact, x_numeric, resid, condA).
    """
    A = [
        1.0  -1.0   0.0    (α - β)   β;
        0.0   1.0  -1.0     0.0      0.0;
        0.0   0.0   1.0    -1.0      0.0;
        0.0   0.0   0.0     1.0     -1.0;
        0.0   0.0   0.0     0.0      1.0
    ]
    b = [α, 0.0, 0.0, 0.0, 1.0]

    x_exact = exact_solution(α, β)
    x_num   = A \ b

    resid   = norm(A*x_num - b) / norm(b)
    condA   = cond(A)

    return (x_exact, x_num, resid, condA)
end

###############################################################################
# Problem 3: Internal rate of return
###############################################################################
function NPV(r, C::Vector{<:Real})
    """
    Net Present Value: NPV(r, C) = Σ [C[t]/(1+r)^t].
    """
    s = 0.0
    for (t, val) in enumerate(C)
        # t-1 in the exponent, since enumerate starts from 1
        s += val / (1+r)^(t-1)
    end
    return s
end

function internal_rate(C::Vector{<:Real}; tol=1e-7, maxiter=200)
    """
    Finds r > -1 s.t. NPV(r,C)=0, via bracket+bisection.
    Returns NaN if no root or bracket fails.
    """
    if all(x -> x ≥ 0, C) || all(x -> x ≤ 0, C)
        @warn "No sign change => no IRR."
        return NaN
    end


    r_low = -0.9999
    r_high = 1e6

    f_low  = NPV(r_low,  C)
    f_high = NPV(r_high, C)

    if abs(f_low) < tol
        return r_low
    elseif abs(f_high) < tol
        return r_high
    elseif f_low*f_high > 0
        @warn "No sign change in bracket => no IRR in [-0.9999,1e6]."
        return NaN
    end

    for iter in 1:maxiter
        r_mid = 0.5*(r_low + r_high)
        f_mid = NPV(r_mid, C)

        if abs(f_mid) < tol
            return r_mid
        end

        if f_mid*f_low > 0
            r_low = r_mid
            f_low = f_mid
        else
            r_high = r_mid
            f_high = f_mid
        end
    end

    @warn "Bisection did not converge in $maxiter iterations."
    return NaN
end

###############################################################################
# Problem 4: CES production & cost minimization
###############################################################################
function f_ces(x1, x2, α, σ)
    """
    CES production function:
        f(x1,x2) = [ α*x1^((σ-1)/σ) + (1-α)*x2^((σ-1)/σ) ]^(σ/(σ-1))
    with σ=1 => Cobb-Douglas.
    """
    if σ == 1
        return x1^α * x2^(1 - α)
    else
        term1 = α * x1^((σ-1)/σ)
        term2 = (1 - α) * x2^((σ-1)/σ)
        return (term1 + term2)^(σ/(σ-1))
    end
end

function plot_ces_contours(α, σ; x1max=5.0, x2max=5.0, levels=10)
    """
    Creates a contour plot of f_ces over [0,x1max] x [0,x2max].
    Returns the Plots.jl plot object.
    """
    x1vals = range(0, x1max, length=200)
    x2vals = range(0, x2max, length=200)
    Z = [f_ces(x1, x2, α, σ) for x2 in x2vals, x1 in x1vals]

    contour(
        x1vals,
        x2vals,
        Z,
        title = "CES contour (α=$α, σ=$σ)",
        xlabel = "x1",
        ylabel = "x2",
        levels = levels
    )
end

function ces_cost_min(α, σ, w1, w2, y)
    """
    Minimize cost = w1*x1 + w2*x2 subject to f_ces(x1,x2,α,σ)=y, x1>=0, x2>=0.
    Returns (cost, x1_opt, x2_opt).

    Updates:
      - Ipopt logs are turned off.
      - We check solver status to avoid false 'did not converge' messages.
    """

    if y <= 0
        return (0.0, 0.0, 0.0)
    end

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0) 
    @variable(model, x1 >= 0)
    @variable(model, x2 >= 0)

    if σ == 1
       
        @constraint(model, x1^α * x2^(1 - α) == y)
    else
        expo = (σ - 1)/σ
        @constraint(model, (α*x1^expo + (1-α)*x2^expo)^(σ/(σ-1)) == y)
    end

    @objective(model, Min, w1*x1 + w2*x2)
    optimize!(model)

    status = termination_status(model)
    if status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        x1_opt = value(x1)
        x2_opt = value(x2)
        return (w1*x1_opt + w2*x2_opt, x1_opt, x2_opt)
    else
        return (NaN, NaN, NaN)
    end
end

# Problem 1 example:
 f(x) = (x + 1)^(1/3) - x
 flag, xsol, fval, diff, xhist, rhist = fixed_point_solver(f, 1.0, 0.0)
 println("Problem 1 => flag=$flag, xsol=$xsol, fval=$fval, diff=$diff")

# Problem 2 example:
 x_exact, x_num, resid, cA = solve_system(0.1, 10)
 println("Problem 2 => x_exact=$x_exact, x_num=$x_num, resid=$resid, cond(A)=$cA")

# Problem 3 example:
 Ctest = [-5, 0, 0, 2.5, 5]
 r_irr = internal_rate(Ctest)
 println("Problem 3 => IRR = $r_irr")

# Problem 4 example:
# 4(a) contour plot
 p = plot_ces_contours(0.5, 1.0, x1max=5, x2max=5, levels=15)
 display(p)

# 4(b) cost minimization
 cval, x1opt, x2opt = ces_cost_min(0.5, 0.25, 2.0, 1.0, 1.0)
 println("Problem 4 => cost=$cval, x1=$x1opt, x2=$x2opt")
