using Plots

# Parameters
β = 0.95         # Discount factor
c = 0.5          # Unemployment benefit
p_values = 0.01:0.01:0.3  # Separation probability range (p)
wages = 1:1:20   # Wage grid (from 1 to 20)
π = 1/length(wages)  # Uniform wage distribution

# Function to calculate VE(w) and VU(w) using value iteration
function calculate_value_functions(p, β, c, wages, π)
    VE = zeros(length(wages))  # Value of being employed at each wage
    VU = zeros(length(wages))  # Value of being unemployed at each wage
    
    max_iterations = 2000
    tolerance = 1e-6

    for iter in 1:max_iterations
        VE_new = zeros(length(wages))
        VU_new = zeros(length(wages))

        # Iterate over wages to update the value functions
        for i in 1:length(wages)
            VE_new[i] = wages[i] + β * ((1 - p) * VE[i] + p * sum(π * VU))
            VU_new[i] = max(c + β * sum(π * VU), VE[i])
        end
        
        # Check for convergence
        if maximum(abs.(VE_new - VE)) < tolerance && maximum(abs.(VU_new - VU)) < tolerance
            break
        end
        
        VE = VE_new
        VU = VU_new
    end
    
    return VE, VU
end

# Function to calculate reservation wage, acceptance probability, and unemployment duration
function calculate_statistics(p, β, c, wages, π)
    VE, VU = calculate_value_functions(p, β, c, wages, π)
    
    # Find the reservation wage w* (where VE(w) >= VU(w))
    reservation_wage = findfirst(x -> VE[x] >= VU[x], 1:length(wages))

    # Calculate the acceptance probability q (probability that wage >= reservation_wage)
    q = sum(wages .>= wages[reservation_wage]) / length(wages)

    # Calculate expected duration of unemployment (1 / q, geometric distribution)
    expected_duration = 1 / q

    return wages[reservation_wage], q, expected_duration
end

# Store the results for all p values
reservation_wages = Float64[]
acceptance_probabilities = Float64[]
durations_of_unemployment = Float64[]

# Loop over p values
for p in p_values
    w_star, q, duration = calculate_statistics(p, β, c, wages, π)
    push!(reservation_wages, w_star)
    push!(acceptance_probabilities, q)
    push!(durations_of_unemployment, duration)
end

# Plot 1: Reservation wage as a function of p
plot(p_values, reservation_wages, label="Reservation Wage (w*)", xlabel="Separation Probability (p)", ylabel="Reservation Wage", linewidth=2, color=:blue)

# Plot 2: Acceptance probability as a function of p
plot(p_values, acceptance_probabilities, label="Acceptance Probability (q)", xlabel="Separation Probability (p)", ylabel="Acceptance Probability", linewidth=2, color=:red)

# Plot 3: Expected duration of unemployment as a function of p
plot(p_values, durations_of_unemployment, label="Duration of Unemployment", xlabel="Separation Probability (p)", ylabel="Expected Duration", linewidth=2, color=:green)




