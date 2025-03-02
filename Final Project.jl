###############################################################################
#       Final Project Code with Outer Constructor for ModelParams
###############################################################################

using LinearAlgebra
using Statistics
using DataFrames
using Printf
using Plots
using Distributions

if !isdefined(Main, :ModelParams)
    mutable struct ModelParams
        β::Float64
        γ::Float64
        α::Float64
        δ::Float64
        A::Float64
        ρ::Float64
        σ::Float64
        z_grid::Vector{Float64}
        Pz::Matrix{Float64}
        ϕ::Float64
        λ::Float64
        τ::Float64
        a_grid::Vector{Float64}
        w::Float64
        r::Float64
        ybar::Float64

        function ModelParams(; β, γ, α, δ, A, ρ, σ, z_grid, Pz, ϕ, λ, τ,
                                 a_grid, w, r, ybar)
            return new(β, γ, α, δ, A, ρ, σ, z_grid, Pz, ϕ, λ, τ, a_grid, w, r, ybar)
        end
    end
end

function stationary_distribution(P::Matrix{Float64}; atol=1e-14, maxiter=1000)
    N = size(P,1)
    dist = fill(1.0 / N, (1, N))  # row vector
    for _ in 1:maxiter
        dist_new = dist * P
        if maximum(abs.(dist_new .- dist)) < atol
            return vec(dist_new)
        end
        dist = dist_new
    end
    return vec(dist)
end

function tauchen(N::Int, ρ::Float64, σ::Float64, m::Float64=3.0)
    std_z = sqrt(σ^2 / (1 - ρ^2))
    z_max = m * std_z
    z_min = -m * std_z
    z_grid = collect(range(z_min, z_max, length=N))
    step = (z_max - z_min) / (N - 1)

    P = zeros(N, N)
    for j in 1:N
        for k in 1:N
            if k == 1
                P[j,k] = cdf(Normal(ρ*z_grid[j], σ), z_grid[1] + step/2)
            elseif k == N
                P[j,k] = 1.0 - cdf(Normal(ρ*z_grid[j], σ), z_grid[N] - step/2)
            else
                P[j,k] = cdf(Normal(ρ*z_grid[j], σ), z_grid[k] + step/2) -
                         cdf(Normal(ρ*z_grid[j], σ), z_grid[k] - step/2)
            end
        end
    end

    π_stat = stationary_distribution(P)
    current_mean = dot(π_stat, exp.(z_grid))
    c = -log(current_mean)
    z_grid_new = z_grid .+ c

    return (z_grid_new, P)
end

function solve_household!(m::ModelParams; tol=1e-6, maxiter=1000)
    na = length(m.a_grid)
    nz = length(m.z_grid)

    function u(c; γ=m.γ)
        (c <= 0) && return -1e10
        if γ == 1.0
            return log(c)
        else
            return (c^(1 - γ) - 1) / (1 - γ)
        end
    end

    function net_labinc(logz::Float64)
        y_pre = m.w * exp(logz)
        return (1.0 - m.τ)*(y_pre / m.ybar)^(1.0 - m.λ)*m.ybar
    end

    β, r = m.β, m.r

    V = zeros(na, nz)
    Vn = copy(V)
    policy_a = zeros(na, nz)
    policy_c = zeros(na, nz)

    dist = 1.0
    iter = 0
    while dist > tol && iter < maxiter
        Vn .= 0.0
        for iz in 1:nz
            labinc = net_labinc(m.z_grid[iz])
            for ia in 1:na
                a_now = m.a_grid[ia]
                best_val = -1e15
                best_ap = m.a_grid[1]
                best_c = 0.0
                for k in 1:na
                    a_next = m.a_grid[k]
                    c_try = labinc + (1+r)*a_now - a_next
                    if c_try <= 0
                        continue
                    end
                    EV_next = 0.0
                    for znext in 1:nz
                        EV_next += m.Pz[iz,znext]*V[k,znext]
                    end
                    val = u(c_try) + β*EV_next
                    if val > best_val
                        best_val = val
                        best_ap = a_next
                        best_c = c_try
                    end
                end
                Vn[ia,iz] = best_val
                policy_a[ia,iz] = best_ap
                policy_c[ia,iz] = best_c
            end
        end
        dist = maximum(abs.(Vn .- V))
        V, Vn = Vn, V
        iter += 1
    end

    return (V, policy_a, policy_c)
end

function compute_stationary_dist(m::ModelParams, policy_a::Matrix{Float64};
                                 tol=1e-12, maxiter=10000)
    na = length(m.a_grid)
    nz = length(m.z_grid)
    distA = fill(1.0/(na*nz), na, nz)
    distAnew = similar(distA)

    function findindex_a(x)
        i = searchsortedfirst(m.a_grid, x)
        if i == 0
            return 1
        elseif i > na
            return na
        else
            if i==1 || i==na
                return i
            else
                ldist = abs(m.a_grid[i-1] - x)
                rdist = abs(m.a_grid[i] - x)
                return ldist < rdist ? (i-1) : i
            end
        end
    end

    for _ in 1:maxiter
        distAnew .= 0.0
        for iz in 1:nz
            for ia in 1:na
                ap = policy_a[ia,iz]
                jap = findindex_a(ap)
                for izp in 1:nz
                    distAnew[jap, izp] += distA[ia,iz]*m.Pz[iz,izp]
                end
            end
        end
        if maximum(abs.(distAnew .- distA)) < tol
            return distAnew
        end
        distA, distAnew = distAnew, distA
    end
    return distA
end

function gini(x::Vector{Float64}, w::Vector{Float64}=ones(length(x)))
    idx = (x .> 0) .& (w .> 0)
    x = x[idx]
    w = w[idx]
    order = sortperm(x)
    xsorted = x[order]
    wsorted = w[order]
    Sw = sum(wsorted)
    yw = xsorted .* wsorted
    Sy = sum(yw)
    cw = cumsum(wsorted)
    cy = cumsum(yw)
    L_prev = 0.0
    F_prev = 0.0
    G = 0.0
    for i in 1:length(xsorted)
        F_i = cw[i]/Sw
        L_i = cy[i]/Sy
        G += (F_i - F_prev)*(L_i + L_prev)
        F_prev = F_i
        L_prev = L_i
    end
    return 1 - G
end

function lorenz_curve(x::Vector{Float64}, w::Vector{Float64}=ones(length(x)))
    idx = (x .>= 0) .& (w .> 0)
    x = x[idx]
    w = w[idx]
    order = sortperm(x)
    xsorted = x[order]
    wsorted = w[order]
    Sw = sum(wsorted)
    yw = xsorted .* wsorted
    Sy = sum(yw)
    cw = cumsum(wsorted)
    cy = cumsum(yw)
    F = cw./Sw
    L = cy./Sy
    return (F, L)
end

function production(A, K, L, α)
    return A*(K^α)*(L^(1-α))
end

function equilibrium_beta(β_guess::Float64, m::ModelParams, target_K::Float64;
                          tol=1e-5, maxiter=100)
    β_low = 0.90
    β_high = 0.9999
    f_low = asset_demand(β_low, m) - target_K
    f_high = asset_demand(β_high, m) - target_K
    if f_low*f_high > 0
        @warn "Bisection range may not bracket root. f_low=$f_low, f_high=$f_high"
    end

    for _ in 1:maxiter
        β_mid = 0.5*(β_low + β_high)
        f_mid = asset_demand(β_mid, m) - target_K
        if abs(f_mid) < tol
            return β_mid
        end
        if f_mid*f_low > 0
            β_low = β_mid
            f_low = f_mid
        else
            β_high = β_mid
            f_high = f_mid
        end
    end
    return 0.5*(β_low + β_high)
end

function asset_demand(β_try::Float64, m::ModelParams)
    oldβ = m.β
    oldr = m.r
    oldw = m.w

    m.β = β_try
    m.r = 0.04
    m.w = 1.0

    V, polA, polC = solve_household!(m)
    distA = compute_stationary_dist(m, polA)

    Abar = 0.0
    na, nz = size(distA)
    for iz in 1:nz
        for ia in 1:na
            Abar += distA[ia,iz]*m.a_grid[ia]
        end
    end

    m.β = oldβ
    m.r = oldr
    m.w = oldw
    return Abar
end

function solve_economy_lambda0!(m::ModelParams; labor_share=0.64,
    r_target=0.04, IY_ratio=0.2, GY_ratio=0.2)
    m.α = 1.0 - labor_share
    α = m.α
    K_ = (α - 0.2)/(0.04*(1-α))
    A_ = 1.0 / ((1-α)*K_^α)
    δ_ = 0.2*A_*K_^(α-1)
    m.A = A_
    m.δ = δ_

    Y_ = production(A_, K_, 1.0, α)
    τ_ = 0.2*Y_
    m.τ = τ_

    β_sol = equilibrium_beta(0.94, m, K_)
    m.β = β_sol

    m.r = 0.04
    m.w = 1.0

    return (K_, Y_, β_sol, τ_)
end

function solve_economy_lambda!(m::ModelParams; λnew=0.15, GY_ratio=0.2, tol=1e-6)
    oldλ = m.λ
    m.λ = λnew

    function clear_capital_market(r_guess::Float64, τ_guess::Float64)
        α, A_, δ_ = m.α, m.A, m.δ
        K_ = ((r_guess + δ_)/(α*A_))^(1/(α-1))
        w_ = (1-α)*A_*(K_^α)

        oldr, oldw, oldτ = m.r, m.w, m.τ
        m.r = r_guess
        m.w = w_
        m.τ = τ_guess

        V, polA, polC = solve_household!(m)
        distA = compute_stationary_dist(m, polA)

        A_d = 0.0
        na, nz = size(distA)
        for iz in 1:nz
            for ia in 1:na
                A_d += distA[ia,iz]*m.a_grid[ia]
            end
        end

        y_sum = 0.0
        for iz in 1:nz
            zval = exp(m.z_grid[iz])
            for ia in 1:na
                y_sum += distA[ia,iz]*(w_*zval)
            end
        end

        m.r = oldr
        m.w = oldw
        m.τ = oldτ

        return (A_d - K_, K_, w_, y_sum)
    end

    function find_r_for_capital_market(τ_guess::Float64)
        rmin, rmax = 0.0001, 0.10
        for _ in 1:100
            rmid = 0.5*(rmin + rmax)
            fmid, _, _, _ = clear_capital_market(rmid, τ_guess)
            fmin, _, _, _ = clear_capital_market(rmin, τ_guess)
            if abs(fmid) < 1e-7
                return rmid
            end
            if fmid*fmin > 0
                rmin = rmid
            else
                rmax = rmid
            end
        end
        return 0.5*(rmin + rmax)
    end

    function gov_revenue_diff(τ_guess::Float64)
        r_star = find_r_for_capital_market(τ_guess)
        α, A_, δ_ = m.α, m.A, m.δ
        K_ = ((r_star + δ_)/(α*A_))^(1/(α-1))
        w_ = (1-α)*A_*(K_^α)

        oldr, oldw, oldτ = m.r, m.w, m.τ
        m.r = r_star
        m.w = w_
        m.τ = τ_guess

        V, polA, polC = solve_household!(m)
        distA = compute_stationary_dist(m, polA)

        y_sum = 0.0
        na, nz = size(distA)
        for iz in 1:nz
            zval = exp(m.z_grid[iz])
            for ia in 1:na
                y_sum += distA[ia,iz]*(w_*zval)
            end
        end

        λ_ = m.λ
        T_ = 0.0
        for iz in 1:nz
            zval = exp(m.z_grid[iz])
            for ia in 1:na
                prob_ = distA[ia,iz]
                y_pre = w_*zval
                post_tax = (1.0 - τ_guess)*(y_pre / y_sum)^(1.0 - λ_)*y_sum
                T_ += prob_*(y_pre - post_tax)
            end
        end

        Y_ = production(A_, K_, 1.0, α)
        m.r = oldr
        m.w = oldw
        m.τ = oldτ

        return T_ - GY_ratio*Y_
    end

    τ_min, τ_max = 0.0, 1.0
    f_min = gov_revenue_diff(τ_min)
    f_max = gov_revenue_diff(τ_max)
    if f_min*f_max > 0
        @warn "Revenue not bracketed by [0,1]. (f_min=$f_min, f_max=$f_max)"
    end

    τ_star = 0.0
    for _ in 1:100
        τ_mid = 0.5*(τ_min + τ_max)
        f_mid = gov_revenue_diff(τ_mid)
        if abs(f_mid) < tol
            τ_star = τ_mid
            break
        end
        f_min = gov_revenue_diff(τ_min)
        if f_mid*f_min > 0
            τ_min = τ_mid
        else
            τ_max = τ_mid
        end
        τ_star = τ_mid
    end

    r_star = find_r_for_capital_market(τ_star)
    α, A_, δ_ = m.α, m.A, m.δ
    K_ = ((r_star + δ_)/(α*A_))^(1/(α-1))
    w_ = (1-α)*A_*(K_^α)

    m.τ = τ_star
    m.r = r_star
    m.w = w_

    V, polA, polC = solve_household!(m)
    distA = compute_stationary_dist(m, polA)

    y_sum = 0.0
    na, nz = size(distA)
    for iz in 1:nz
        zval = exp(m.z_grid[iz])
        for ia in 1:na
            y_sum += distA[ia,iz]*(w_*zval)
        end
    end
    m.ybar = y_sum

    return (τ_star, r_star, w_, K_, distA, V, polA, polC)
end

function main()
    # 1) Setup
    ρ, σ = 0.9, 0.4
    (zraw, Pz) = tauchen(5, ρ, σ, 3.0)

    amin, amax = 0.0, 40.0
    nA = 200
    a_grid = collect(range(amin, amax, length=nA))

    # 2) Construct ModelParams with keyword arguments
    mparam = ModelParams(
        β=0.95,
        γ=2.0,
        α=0.36,
        δ=0.0,
        A=1.0,
        ρ=ρ,
        σ=σ,
        z_grid=zraw,
        Pz=Pz,
        ϕ=0.0,
        λ=0.0,
        τ=0.0,
        a_grid=a_grid,
        w=1.0,
        r=0.04,
        ybar=1.0
    )

    # 3) Solve λ=0 economy
    labor_share_US = 0.64
    K0, Y0, β0, τ0 = solve_economy_lambda0!(mparam; labor_share=labor_share_US)
    V0, polA0, polC0 = solve_household!(mparam)
    distA0 = compute_stationary_dist(mparam, polA0)

    # Gini λ=0
    atinc0 = Float64[]
    assets0 = Float64[]
    weights0 = Float64[]
    na, nz = size(distA0)
    for iz in 1:nz
        zval = exp(mparam.z_grid[iz])
        y_pre = mparam.w*zval
        posttax = (1.0 - mparam.τ)*(y_pre / mparam.ybar)^(1.0 - mparam.λ)*mparam.ybar
        for ia in 1:na
            prob_ = distA0[ia,iz]
            push!(atinc0, posttax)
            push!(assets0, mparam.a_grid[ia])
            push!(weights0, prob_)
        end
    end
    Gini_atinc0 = gini(atinc0, weights0)
    Gini_assets0 = gini(assets0, weights0)

    # 4) Solve λ=0.15 economy
    (τ_star, r_star, w_star, K1, distA1, V1, polA1, polC1) = solve_economy_lambda!(mparam; λnew=0.15)

    # Gini λ=0.15
    atinc1 = Float64[]
    assets1 = Float64[]
    weights1 = Float64[]
    na, nz = size(distA1)
    for iz in 1:nz
        zval = exp(mparam.z_grid[iz])
        y_pre = mparam.w*zval
        posttax = (1.0 - mparam.τ)*(y_pre / mparam.ybar)^(1.0 - mparam.λ)*mparam.ybar
        for ia in 1:na
            prob_ = distA1[ia,iz]
            push!(atinc1, posttax)
            push!(assets1, mparam.a_grid[ia])
            push!(weights1, prob_)
        end
    end
    Gini_atinc1 = gini(atinc1, weights1)
    Gini_assets1 = gini(assets1, weights1)

    # 5) Summary
    stats = DataFrame(
        λ = [0.0, 0.15],
        r = [0.04, mparam.r],
        w = [1.0, mparam.w],
        τ = [τ0, mparam.τ],
        K_over_Y = [
            K0 / production(mparam.A, K0, 1.0, mparam.α),
            K1 / production(mparam.A, K1, 1.0, mparam.α)
        ],
        Gini_aftertax_inc = [Gini_atinc0, Gini_atinc1],
        Gini_assets       = [Gini_assets0, Gini_assets1]
    )

    println("=== Final Comparison of the Two Economies ===")
    println(stats)

    # 6) Plots
    zlow = 1
    zmid = Int(cld(size(V0,2),2))
    zhigh = size(V0,2)

    p1 = plot(mparam.a_grid, V0[:,zlow], label="z-low",
        title="Value Function, λ=0 (z-low, mid, high)",
        xlabel="Assets", ylabel="V(a,z)")
    plot!(p1, mparam.a_grid, V0[:,zmid], label="z-mid")
    plot!(p1, mparam.a_grid, V0[:,zhigh], label="z-high")
    display(p1)

    p2 = plot(mparam.a_grid, V1[:,zlow], label="z-low",
        title="Value Function, λ=0.15 (z-low, mid, high)",
        xlabel="Assets", ylabel="V(a,z)")
    plot!(p2, mparam.a_grid, V1[:,zmid], label="z-mid")
    plot!(p2, mparam.a_grid, V1[:,zhigh], label="z-high")
    display(p2)

    p3 = plot(mparam.a_grid, polA0[:,zlow], label="z-low",
        title="Policy for a'(a) - λ=0", xlabel="Assets", ylabel="a'(a,z)")
    plot!(p3, mparam.a_grid, polA0[:,zmid], label="z-mid")
    plot!(p3, mparam.a_grid, polA0[:,zhigh], label="z-high")
    display(p3)

    p4 = plot(mparam.a_grid, polA1[:,zlow], label="z-low",
        title="Policy for a'(a) - λ=0.15", xlabel="Assets", ylabel="a'(a,z)")
    plot!(p4, mparam.a_grid, polA1[:,zmid], label="z-mid")
    plot!(p4, mparam.a_grid, polA1[:,zhigh], label="z-high")
    display(p4)

    distA_marg0 = sum(distA0, dims=2)[:]
    distA_marg1 = sum(distA1, dims=2)[:]

    p5 = plot(mparam.a_grid, distA_marg0, title="Asset Distribution, λ=0",
        xlabel="Assets", ylabel="Density")
    display(p5)

    p6 = plot(mparam.a_grid, distA_marg1, title="Asset Distribution, λ=0.15",
        xlabel="Assets", ylabel="Density")
    display(p6)

    F0_at, L0_at = lorenz_curve(atinc0, weights0)
    F1_at, L1_at = lorenz_curve(atinc1, weights1)

    p7 = plot(F0_at, L0_at, label="λ=0", title="Lorenz Curve (After-tax Inc)",
        xlabel="Cumulative population", ylabel="Cumulative Income Share")
    plot!(p7, [0,1], [0,1], label="45°")
    display(p7)

    p8 = plot(F1_at, L1_at, label="λ=0.15", title="Lorenz Curve (After-tax Inc)",
        xlabel="Cumulative population", ylabel="Cumulative Income Share")
    plot!(p8, [0,1], [0,1], label="45°")
    display(p8)

    F0_a, L0_a = lorenz_curve(assets0, weights0)
    F1_a, L1_a = lorenz_curve(assets1, weights1)

    p9 = plot(F0_a, L0_a, label="λ=0", title="Lorenz Curve (Assets)",
        xlabel="Cumulative population", ylabel="Cumulative Asset Share")
    plot!(p9, [0,1], [0,1], label="45°")
    display(p9)

    p10 = plot(F1_a, L1_a, label="λ=0.15", title="Lorenz Curve (Assets)",
        xlabel="Cumulative population", ylabel="Cumulative Asset Share")
    plot!(p10, [0,1], [0,1], label="45°")
    display(p10)

end

main()
