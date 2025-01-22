# Constants and parameters
X = 50.0          # Utility from the orchid
c = 0.5           # Cost of approaching a vendor
q = 0.15          # Probability vendor has the orchid
pmin = 10.0       # Minimum price
pmax = 100.0      # Maximum price
N = 51            # Total number of vendors
price_grid = range(pmin, pmax, length=100) |> collect
n_prices = length(price_grid)

# Bellman solver (static computation for optimal decisions)
function Bellman_Equation(N, price_grid, q, c, X)
    v = zeros(Float64, N + 1)  # Value function
    σapproach = zeros(Int, N)  # Optimal approach decisions
    σbuy = zeros(Int, N, n_prices)  # Optimal buy decisions

    for n in N:-1:1
        vA = -Inf  # Value of approaching the vendor
        for (j, p) in enumerate(price_grid)
            vB = X - p  # Utility if Basil buys the orchid
            vA_temp = q * vB + (1 - q) * v[n + 1] - c  # Bellman equation
            if vA_temp > vA
                vA = vA_temp
                σbuy[n, j] = 1  # Optimal decision to buy
            else
                σbuy[n, j] = 0  # No buy action
            end
        end
        v[n] = max(0.0, vA)  # Value of continuing the search
        σapproach[n] = vA > 0.0 ? 1 : 0  # Should approach the vendor?
    end
    return v, σapproach, σbuy
end

# Answer problem questions (statistical calculation)
function Statistics(Bought, Final_Cost, total_vendors)
    Probability_of_Buying = Bought / total_vendors  # Probability of buying
    Price_Paid = Bought > 0 ? Final_Cost / Bought : 0.0  # Expected price
    return Probability_of_Buying, Price_Paid, total_vendors
end

# Interactive simulation with dynamic tracking
function interactive_simulation!(N, price_grid, q, c, X)
    println("\nWelcome to Basil's Orchid Quest!\n")
    # Initiate all variables required
    Cost_Incurred = 0.0
    No_of_Vendors_Approached = 0
    Found_Orchid = false
    Bought = 0
    Final_Cost = 0.0

    for n in 1:N
        No_of_Vendors_Approached += 1
        Cost_Incurred += c
        println("Vendor $n:")

        # Simulate whether the vendor has an orchid
        has_orchid = rand() < q  # To make each output unique in it's own way.

        if has_orchid
            # Simulate the price from the price grid for the orchid
            price_index = rand(1:n_prices)  # To make sure no prices appear twice in a single run
            price = price_grid[price_index]  # Prices Matrix
            println("\nThe vendor has the orchid! Price: \$$(round(price, digits=2))\n")

            # User decision
            action = ""
            while !(action in ["b", "t", "c"])
                print("\nYour action (b to BUY/t to TERMINATE/c to CONTINUE): ")
                action = readline()
            end

            if action == "b"
                Cost_Incurred += price
                Bought += 1
                Final_Cost += price
                println("\nBasil bought the orchid for \$$(round(price, digits=2)).")
                println("Total cost: \$$(round(Cost_Incurred, digits=2))")
                Found_Orchid = true
                break
            elseif action == "t"
                println("\nBasil terminated the search.")
                println("Total cost: \$$(round(Cost_Incurred, digits=2))")
                break
            else
                println("\nBasil decided to continue the search.")
            end
        else
            println("The vendor does not have the orchid. Basil continues the search.")
        end
    end

    if !Found_Orchid

        println("\n\nEND OF SEARCH")
        println("\nBasil approached all vendors but did not find the orchid.")  # Bad luck scenario
        println("Total cost incurred: \$$(round(Cost_Incurred, digits=2))")
    end

    # Computes the final statistics using the feedback through interactive stimulation function above
    Probability_of_Buying, Price_Paid, total_vendors = Statistics(Bought, Final_Cost, No_of_Vendors_Approached)

    println("\n\n After Basil's decisions:")
    println("\n1. Probability of Basil actually buying the orchid based on decisions: $(round(Probability_of_Buying * 100, digits=2))%") # This is based on the decisions made by Basil (User) through interactive stimulation function
    println("2. Final price paid for the pruchase of Orchid: \$$(round(Price_Paid, digits=2))") # Total cost incurred in the process based on decisions made.
    println("3. Number of vendors Basil has approached: $(round(total_vendors, digits=2))\n") # Total number of vendors appraoched before terminating or finding the Orchid
    ## In case of Bad Luck scenario, the number of vendors would be equal to the toal number of vendors.
end

# Main execution
println("\n Before Basil's decisions:")  
println("\n")
println("1. Probability Basil will buy the orchid: 100.0%")  # As Basil definitely wants to buy the Orchid, Initial state probability would be 100%
println("2. Expected price Basil will pay: \$$(round(pmin, digits=2))")  # Basil is willing to pay $10 (excluding other costs incurred in the process)
println("3. Expected number of vendors Basil will approach: $(N)\n")  # Total number of Vendor available Festival Sale

# Start interactive simulation, dynamically updating statistics
interactive_simulation!(N, price_grid, q, c, X)
