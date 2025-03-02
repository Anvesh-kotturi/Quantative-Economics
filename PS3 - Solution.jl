using Random
using LinearAlgebra
using Statistics
using Plots  

################################################################################
#                           1) BASIL’S ORCHID QUEST
################################################################################

# Basil parameters
const N_basil       = 50
const X_basil       = 50.0
const f_basil       = 0.5
const q_basil       = 0.15
const pmin_basil    = 10.0
const pmax_basil    = 100.0
const dp_basil      = 0.1
const prices_basil  = collect(pmin_basil:dp_basil:pmax_basil)
const nPrices_basil = length(prices_basil)

function solve_basil_problem()
    N = N_basil
    v       = Vector{Float64}(undef, N+1)
    approach = Vector{Int}(undef, N+1)
    buy    = Matrix{Int}(undef, N+1, nPrices_basil)

    v[N+1] = -N*f_basil
    approach[N+1] = 0
    for pidx in 1:nPrices_basil
        buy[N+1,pidx] = 0
    end

    uniform_prob = 1.0 / nPrices_basil

    for n in reverse(0:N-1)
        i = n + 1
        v_stop = -n * f_basil

        exp_offer = 0.0
        for pidx in 1:nPrices_basil
            price = prices_basil[pidx]
            buy_util = X_basil - price - (n+1)*f_basil
            exp_offer += max(buy_util, v[i+1]) * uniform_prob
        end
        v_approach = -f_basil + q_basil*exp_offer + (1-q_basil)*v[i+1]

        if v_stop >= v_approach
            v[i] = v_stop
            approach[i] = 0
            for pidx in 1:nPrices_basil
                buy[i,pidx] = 0
            end
        else
            v[i] = v_approach
            approach[i] = 1
            for pidx in 1:nPrices_basil
                price = prices_basil[pidx]
                buy_util = X_basil - price - (n+1)*f_basil
                buy[i,pidx] = (buy_util>v[i+1]) ? 1 : 0
            end
        end
    end

    return v, approach, buy
end

function compute_basil_stats(v, approach, buy)
    N = N_basil
    P_search = zeros(N+1)
    P_search[1] = 1.0

    total_prob_buy = 0.0
    total_price_buy = 0.0
    expected_approaches = 0.0

    for i in 1:N
        pn = P_search[i]
        if pn == 0
            continue
        end
        if approach[i] == 0
            # Basil stops
            continue
        end

        prob_offer = q_basil
        accept_prob = 0.0
        accept_price_sum = 0.0
        for pidx in 1:nPrices_basil
            if buy[i,pidx] == 1
                accept_prob += 1.0/nPrices_basil
                accept_price_sum += prices_basil[pidx]/nPrices_basil
            end
        end

        prob_buy_now = pn * prob_offer * accept_prob
        total_prob_buy += prob_buy_now
        total_price_buy += prob_buy_now * accept_price_sum

        prob_continue = pn * ((1.0 - q_basil) + prob_offer*(1.0 - accept_prob))
        if i+1 <= N+1
            P_search[i+1] += prob_continue
        end

        expected_approaches += pn
    end

    exp_price = total_prob_buy>0 ? (total_price_buy/total_prob_buy) : 0.0
    return total_prob_buy, exp_price, expected_approaches
end

function run_ideal_demo(approach, buy)
    println("\n=== Basil's Automated Demonstration: *Optimal* Policy ===")
    rng = MersenneTwister(0)
    n=0
    total_cost=0.0
    while n<N_basil
        if approach[n+1]==0
            println("At vendor $n, Basil stops. Utility = $(-n*f_basil). End demonstration.\n")
            return
        end
        total_cost += f_basil
        println("Approach vendor $(n+1). cost so far=$total_cost.")
        has_orchid = rand(rng)<q_basil
        if has_orchid
            p = rand(rng)*(pmax_basil - pmin_basil)+pmin_basil
            p = round(p, digits=1)
            p = clamp(p, pmin_basil, pmax_basil)
            pidx = Int(round((p - pmin_basil)/dp_basil +1))
            println("  Vendor $(n+1) has orchid at price=$p.")
            if buy[n+1,pidx]==1
                util = X_basil - p - total_cost
                println("  Basil BUYS. Utility=$util. End demonstration.\n")
                return
            else
                println("  Basil REJECTS. Continue.")
            end
        else
            println("  Vendor $(n+1) does NOT have the orchid.")
        end
        n+=1
    end
    println("Reached vendor $N_basil with no purchase. Utility=$(-N_basil*f_basil). End.\n")
end
function run_interactive_mode(approach, buy)
    println("\n=== BASIL INTERACTIVE MODE: *You* control Basil ===")
    println("Type 'A' to approach or 'H' to go home. If orchid is found, type 'B' to buy or 'R' to reject.\n")

    rng = MersenneTwister()
    n = 0
    total_cost = 0.0
    while n < N_basil
        println("\nAt vendor $n, cost so far=$total_cost.")
        print("Approach next vendor (A) or go home (H)? ")
        ans = readline()
        if ans == ""
            ans = "A"
        end
        ans = lowercase(ans)
        if ans[1] == 'h'
            println("You STOP. Utility=$(-total_cost). End.\n")
            return
        elseif ans[1] == 'a'
            total_cost += f_basil
            println("Approach vendor $(n+1). cost so far=$total_cost.")
            has_orchid = rand(rng) < q_basil
            if has_orchid
                p = rand(rng)*(pmax_basil - pmin_basil) + pmin_basil
                p = round(p, digits=1)
                p = clamp(p, pmin_basil, pmax_basil)
                println("  Vendor $(n+1) HAS orchid at price=$p.")
                print("Buy (B) or Reject (R)? ")
                ans2 = readline()
                if ans2 == ""
                    ans2 = "B"
                end
                ans2 = lowercase(ans2)
                if ans2[1] == 'b'
                    util = X_basil - p - total_cost
                    println("You BUY at $p. Utility=$util. End.\n")
                    return
                else
                    println("You REJECT. Continue.")
                end
            else
                println("  Vendor $(n+1) does NOT have the orchid.")
            end
            n += 1
        else
            println("Invalid choice. Type A or H.")
        end
    end
    println("Reached final vendor $N_basil with no purchase. Utility=$(-total_cost).\n")
end


################################################################################
# 2) JOB SEARCH WITH SEPARATIONS (Iteration Cap)
################################################################################

const wagegrid = 10:1:100
const nw       = length(wagegrid)
const β_js     = 0.95
const c_js     = 30.0

function solve_job_search_with_separations(p::Float64)
    VU = zeros(nw)
    VE = zeros(nw)

    function bellman_VE(wi, oldVU)
        w = wagegrid[wi]
        E_VU = mean(oldVU)
        return (w + β_js*p*E_VU) / (1.0 - β_js*(1.0-p))
    end

    function bellman_VU(wi, oldVU, oldVE)
        E_VU = mean(oldVU)
        return max(oldVE[wi], c_js + β_js*E_VU)
    end

    maxiter = 5000
    tol = 1e-6
    diff = 1.0
    iter = 0
    oldVU = copy(VU)
    oldVE = copy(VE)

    while diff>tol && iter<maxiter
        iter+=1
        for wi in 1:nw
            VE[wi] = bellman_VE(wi, oldVU)
        end
        for wi in 1:nw
            VU[wi] = bellman_VU(wi, oldVU, VE)
        end
        diff = maximum(abs.(VE .- oldVE)) + maximum(abs.(VU .- oldVU))
        oldVU .= VU
        oldVE .= VE
    end
    if iter>=maxiter
        @warn "Job search iteration hit maxiter=$maxiter (diff=$diff, tol=$tol)."
    end

    E_VU = mean(VU)
    reservation_index = findfirst(wi -> VE[wi]>= c_js+β_js*E_VU, 1:nw)
    reservation_wage = reservation_index==nothing ? Inf : wagegrid[reservation_index]

    accept_prob = 0.0
    if isfinite(reservation_wage)
        n_accept = count(w->w>=reservation_wage, wagegrid)
        accept_prob = n_accept / nw
    end
    exp_duration = accept_prob>0 ? 1.0/accept_prob : Inf
    return reservation_wage, accept_prob, exp_duration
end

function generate_job_search_plots()
    ps = 0:0.05:0.9
    wstars = Float64[]
    accs   = Float64[]
    durs   = Float64[]

    for pp in ps
        wstar, qacc, dur = solve_job_search_with_separations(pp)
        push!(wstars, wstar)
        push!(accs,   qacc)
        push!(durs,   dur)
    end

    p1 = plot(ps, wstars, xlabel="p", ylabel="Reservation Wage", title="Reservation Wage vs p")
    p2 = plot(ps, accs,   xlabel="p", ylabel="Acceptance Prob",  title="Acceptance Prob vs p")
    p3 = plot(ps, durs,   xlabel="p", ylabel="Unemp Duration",   title="Unemp Duration vs p")

    return p1, p2, p3
end


################################################################################
# 3) NEOCLASSICAL GROWTH MODEL (Setter syntax for axis labels)
################################################################################

const β_g  = 0.95
const α_g  = 0.3
const δ_g  = 0.05
const gamma_values_g = [0.5, 1.0, 2.0]

function k_star(β, α, δ)
    function eq(k)
        return α*k^(α-1) + (1-δ) - 1/β
    end
    lo, hi = 1e-6, 1e6
    for i in 1:300
        mid = 0.5*(lo+hi)
        if eq(mid)>0
            hi=mid
        else
            lo=mid
        end
    end
    return 0.5*(lo+hi)
end

function simulate_ngm(gamma; T=50, k0=nothing)
    kss = k_star(β_g, α_g, δ_g)
    if k0===nothing
        k0 = 0.5*kss
    end

    kpath = zeros(T+1)
    cpath = zeros(T)
    ipath = zeros(T)
    ypath = zeros(T)

    kpath[1] = k0
    css = kss^α_g - δ_g*kss
    for t in 1:T
        kt = kpath[t]
        yt = kt^α_g
        max_c = yt+(1-δ_g)*kt
        c_guess = 0.2*css + 0.8*max_c
        c_guess = min(c_guess, max_c)
        cpath[t] = c_guess
        ipath[t] = max_c - c_guess
        ypath[t] = yt
        kpath[t+1] = max_c - c_guess
    end
    return kpath, cpath, ipath, ypath, kss, css
end

function time_to_half_gap(gamma)
    T=300
    kpath, cpath, ipath, ypath, kss, css = simulate_ngm(gamma, T=T)
    halfgap = 0.5*(kss - kpath[1])
    for t in 1:T
        if (kss - kpath[t+1])<halfgap
            return t
        end
    end
    return T
end

function generate_growth_results()
    println("Gamma   #PeriodsToCloseHalfGap")
    for g in gamma_values_g
        t_hg = time_to_half_gap(g)
        println("$(g)         $(t_hg)")
    end

    T=30
    p_capital = plot()
    p_output  = plot()
    p_invy    = plot()
    p_cy      = plot()

    for g in gamma_values_g
        kpath, cpath, ipath, ypath, kss, css = simulate_ngm(g, T=T)
        plot!(p_capital, 0:T, kpath, label="γ=$g")
        plot!(p_output,  1:T, ypath, label="γ=$g")
        plot!(p_invy,    1:T, ipath ./ ypath, label="γ=$g")
        plot!(p_cy,      1:T, cpath ./ ypath, label="γ=$g")
    end

    xlabel!(p_capital, "t")
    ylabel!(p_capital, "k(t)")
    xlabel!(p_output, "t")
    ylabel!(p_output, "y(t)")
    xlabel!(p_invy, "t")
    ylabel!(p_invy, "i(t)/y(t)")
    xlabel!(p_cy, "t")
    ylabel!(p_cy, "c(t)/y(t)")

    p_final = plot(p_capital, p_output, p_invy, p_cy, layout=(2,2), legend=:bottomright)
    display(p_final)
end


################################################################################
# 4) MARKOV DYNAMICS (Column-vector iteration to avoid DimensionMismatch)
################################################################################

const Zvals = [:z1,:z2,:z3]
const Pmark = [0.5 0.3 0.2;
               0.2 0.7 0.1;
               0.3 0.3 0.4]

function sigma_markov(x,z)
    if z==:z1
        return 0
    elseif z==:z2
        return x
    else
        return x<=4 ? x+1 : 3
    end
end

function markov_dynamics_XZ()
    states = [(x,z) for x in 0:5 for z in Zvals]  
    nstates = length(states)
    M = zeros(nstates, nstates)

    for i in 1:nstates
        (xx, zz) = states[i]
        xnext = sigma_markov(xx, zz)
        zidx = findfirst(==(zz), Zvals)
        for znext_idx in 1:3
            j = findfirst(s->s==(xnext, Zvals[znext_idx]), states)

            M[i,j] = Pmark[zidx,znext_idx]
        end
    end

    dist = fill(1.0/nstates, nstates)  
    tol = 1e-12
    diff = 1.0

    while diff>tol
        newdist = M' * dist  
        dd = abs.(newdist .- dist)
        diff = maximum(dd)
        dist = newdist
    end

    margX = zeros(6)
    for i in 1:nstates
        (xx, zz) = states[i]
        margX[xx+1] += dist[i]
    end
    expX = sum(x*margX[x+1] for x in 0:5)
    return M, dist, margX, expX
end

function main()
    # PROBLEM 1) Basil's Orchid Quest
    
    println("\n--- PROBLEM 1) Basil’s Orchid Quest ---")
    v_basil, approach_basil, buy_basil = solve_basil_problem()
    prob_buy, exp_price, exp_n = compute_basil_stats(v_basil, approach_basil, buy_basil)
    println("Probability Basil buys = $prob_buy")
    println("Expected price (conditional on buy) = $exp_price")
    println("Expected # vendors approached = $exp_n")

    run_ideal_demo(approach_basil, buy_basil)
    run_interactive_mode(approach_basil, buy_basil)

    # PROBLEM 2) Job Search with Separations
    
    println("\n--- PROBLEM 2) Job Search with Separations ---")
    for pval in (0.1, 0.3, 0.5)
        wstar, qacc, dur = solve_job_search_with_separations(pval)
        println("p=$pval => reservation wage=$wstar, accept prob=$qacc, E[unemp dur]=$dur")
    end
     p1,p2,p3 = generate_job_search_plots()
     display(p1); display(p2); display(p3)

    # PROBLEM 3) Neoclassical Growth
    
    println("\n--- PROBLEM 3) Neoclassical Growth Model ---")
    println("Half-gap times for gamma in [0.5,1.0,2.0]:")
    generate_growth_results()
    
    # PROBLEM 4) Markov Dynamics
    
    println("\n--- PROBLEM 4) Markov Dynamics (X,Z) ---")
    M, dist, margX, expX = markov_dynamics_XZ()
    println("Transition matrix size: ", size(M))  # (18,18)
    println("Stationary dist (as column vector) => shape: ", size(dist))
    println(dist)
    println("Marginal distribution of X: ", margX)
    println("Expected X in steady state: ", expX)

end

main()
