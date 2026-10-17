import numpy as np
import sys 
import matplotlib.pyplot as plt

# ----------------- ALL FUNCTIONS AT THE TOP --------------------------------

def divide_interval(N, h, R):                                       
    """ This function helps create our grid interval """
    interval = []                                                              # We are going to make our interval list 0 < x < R
    x = 0.0
    
    while len(interval) < (N + 1):                         
        if len(interval) == N:                                                 # If we are at the end of our list, make the last value R 
            interval.append(R)
            return interval
        else:                                                                  # Keep adding our values to the list rounded to 7 digits
            interval.append(round(x, 7))
            x += h

def centerline_equation_approachA(A_matrix, b_vector, q_triple_prime, h, k):
    """ Creates the centerline equation using Approach A (ghost-point) boundary
        condition on the first row of the system (index 0)
        Equation: (2*T_0) - (2*T_1) = (q''' * h^2) / K """
    
    RHS = (q_triple_prime * (h**2)) / k                                        # Calculate our constant on the RHS of the equation
    A_matrix[0, 0] = 2.0                                                       # Coefficient of T_0
    A_matrix[0, 1] = -2.0                                                      # Coefficient of T_1
    b_vector[0]    = RHS
    
    return A_matrix, b_vector

def centerline_equation_approachB(A_matrix, b_vector, q_triple_prime, h, k):
    """ Creates the centerline equation using Approach B boundary condition on 
        the first row of the system (index 0)
        Equation: T_0 - T_1 = 0
    """
    
    A_matrix[0, 0] = 1.0                                                       # Coefficient of T_0
    A_matrix[0, 1] = -1.0                                                      # Coefficient of T_1
    b_vector[0]    = 0
    
    return A_matrix, b_vector
    
def interior_equation(A_matrix, b_vector, q_triple_prime, h, k, N):
    """ Creates the interior equation
        Equation: -T_{i-1} + 2*T_i - T_{i+1} = (q''' * (h^2)) / k
    """
    
    RHS = (q_triple_prime * (h**2)) / k                                        # Calculate our constant on the RHS of the equation
    
    for i in range(1, N-1):                                                    # Loops row 1 starting at columnn 1 up to the second to last
        A_matrix[i, i-1]   = -1.0                                              # Coefficient of T_{i-1}
        A_matrix[i, i]     = 2.0                                               # Coefficient of T_i
        A_matrix[i, i+1]   = -1.0                                              # Coefficient of T_{i+1}
        b_vector[i]        = RHS                                
        
    return A_matrix, b_vector

def outer_surface_equation(A_matrix, b_vector, q_triple_prime, h, k, T_s, N):  
    """ Creates the outer surface equation
        Equation: -T_{N-2} + 2*T{N-1} = ((q''' * (h^2)) / k) + T_s
    """

    RHS = ((q_triple_prime * (h**2)) / k) + T_s                                # Calculate our constant on the RHS of the equation
    A_matrix[N-1, N-2] = -1.0                                                  # Coefficient of T_{N-2}
    A_matrix[N-1, N-1] = 2.0                                                   # Coefficient of T_{N-1}
    b_vector[N-1]      = RHS
    
    return A_matrix, b_vector
    
def residual_check(A_matrix, b_vector, T_vector):                              # This function is used for step 6 / step 7
    """ This function is to solve for the residual vector r = AT - b, and it 
        should return the max absolute residual value in the system. To multiply
        our matrix with a vector, we need to use the true matrix multiplication 
        symbol (also known as dot product) by using '@'. 
    """
    
    r = (A_matrix @ T_vector) - b_vector                                       # Calculates the residual vector (its going to be a lot of numbers)
    max_r = np.max(np.abs(r))                                                  # This helps find the absolute maximum value out of the list of numbers for the largest error from zero
    
    return max_r


def step_6_Nlist(N):
   """ Will generate a list of 5 N Values, doubling each time it loops 
   """
   
   N_list = []                                                                
   while len(N_list) < 5:
        N_list.append(N)
        N = N * 2 
    
   return N_list

def simulation_plotting(A_matrix, b_vector, q_triple_prime, N_list, N, R, T_s):
    """ Will loop through our N vector list, build our matrices, solve them, and
        then print. Even though this function is named after step 6, both step 7
        and step 8 reside in this code as well. 
    """
   
    T_0_analytical = T_s + ((q_triple_prime * (R**2)) / (2 * k)) 
    
    print(f"\nOur Analystical Centerline Temperatue (T_0) is: {T_0_analytical}\n")
   
# --------------------- STEP 7: PLOT DATA ------------------------------------
    grid_points_plot = []
    T_0_A_plots = []
    T_0_B_plots = [] 
# ----------------------------------------------------------------------------
   
    for current_N in N_list:                                                   # We got N_list from the function step_6_Nlist
        current_h = R / current_N
        grid_points = current_N + 1
        
        print(f"For N = {current_N}, our current gid spacing is h = {current_h}")
        print(f"For N = {current_N}, our number of grid points is: {grid_points}:")
        
        grid_points_plot.append(grid_points)                                   # Part 7
        
# ---------------- Build and Solve Approach A System --------------------------
    
        A_A = np.zeros((current_N, current_N))
        b_A = np.zeros(current_N)
        A_A, b_A = centerline_equation_approachA(A_A, b_A, q_triple_prime, current_h, k)
        A_A, b_A = interior_equation(A_A, b_A, q_triple_prime, current_h, k, current_N)
        A_A, b_A = outer_surface_equation(A_A, b_A, q_triple_prime, current_h, k, T_s, current_N)
       
        T_solved_A = np.linalg.solve(A_A, b_A)
        T_0_A = T_solved_A[0]                                                  # The first value of the vecotr is the centerline T_0
        difference_A = abs(T_0_A - T_0_analytical)
        
        T_0_A_plots.append(T_0_A)                                              # Part 7 
        
# -------------------- Build and Solve Approach B System ----------------------
       
        A_B = np.zeros((current_N, current_N))
        b_B = np.zeros(current_N)
        A_B, b_B = centerline_equation_approachB(A_B, b_B, q_triple_prime, current_h, k)
        A_B, b_B = interior_equation(A_B, b_B, q_triple_prime, current_h, k, current_N)
        A_B, b_B = outer_surface_equation(A_B, b_B, q_triple_prime, current_h, k, T_s, current_N)
       
        T_solved_B = np.linalg.solve(A_B, b_B)
        T_0_B = T_solved_B[0]                                                  # The first value of the vecotr is the centerline T_0
        difference_B = abs(T_0_B - T_0_analytical)
        
        T_0_B_plots.append(T_0_B)                                              # Part 7

# ---------------- STEP 8: CALCULATED THE RESIDUALS ---------------------------

        max_residual_A = residual_check(A_A, b_A, T_solved_A)
        max_residual_B = residual_check(A_B, b_B, T_solved_B)
        
# --------------------- PRINT RESULTS PER LOOP -------------------------------

        print(f"\nFor Approach A with N = {current_N}:")
        print(f"Our Calculated Centerline Temperature for N = {current_N} is T_0 = {T_0_A:<15.4f}")
        print(f"Our Absolute Error Difference is: {difference_A}")
        print(f"Our Max System Residual (AT-b) is: {max_residual_A}")
        print(f"\nFor Approach B with N = {current_N}:")
        print(f"Our Calculated Centerline Temperature for N = {current_N} is T_0 = {T_0_B:<15.4f}")
        print(f"Our Absolute Error Difference is: {difference_B}") 
        print(f"Our Max System Residual (AT-b) is: {max_residual_B}\n")
        
# ----------------------- STEP 7: PLOT LE GRAPHS -----------------------------

    plt.figure(figsize=(9,6))
    plt.plot(grid_points_plot, T_0_A_plots, 'go--', label = 'Approach A (Ghost Point (Central Difference))', markersize=8)  # Plotting our Approach A values
    plt.plot(grid_points_plot, T_0_B_plots, 'bs--', label = 'Approach B (Direct DX (Forward Difference))', markersize=8)    # Plotting Approach B values w/ asmptotic convergence curve
    plt.axhline(y = T_0_analytical, color = 'r', linestyle = '-', linewidth = 2, label = 'Analytical Solution')   # Plotting the horizontal analytical line (for N = 5, it should be 1740.5 k)
    
    plt.title('Centerline Temperature Convergence Study for Two Different Approaches', fontsize = 14, fontweight = 'bold', pad = 15)
    plt.xlabel('Number of Grid Points ($N + 1$)', fontsize = 12)
    plt.ylabel('Calculated Centertline Temperature $T_0$ (K)', fontsize = 12)
    
    plt.xticks(grid_points_plot)
    plt.grid(True, linestyle=':', alpha = 0.6)
    plt.legend(loc = 'lower right', fontsize = 11)

    plt.tight_layout()
#    plt.savefig('centerline_convergence_plot.png', dpi=300)                   # In case you wanted to save your graph
    plt.show()    
    
# ---------------------- NE367 SECTION --------------------------------------

def q_triple_prime_non_uniform(q_triple_prime, x, R):                          # This is a function for NE367 students (I am not a NE367 student, this is just to see if I can attempt this problem)
    """ We are now saying that q''' is no longer uniform, but follows a function. 
        Our q''' will now quadratically increase towards the outside of our pellet,
        or in other words, depending on our x position. 
    """
    
    C = (q_triple_prime) / (1 + (2/3))
    new_q_triple_prime = C * (1 + (2 * ((x / R)**2)))

    return new_q_triple_prime

def interior_equation_non_uniform(A_matrix, b_vector, q_triple_prime, h, k, N, R):
    """ Because now we have a non uniform q''', we cant set the right hand side (RHS)
        of our equation to a constant. So now we have to fill our interior points 
        using a position-dependent q'''
    """
    
    for i in range(1, N-1):                                                    # Loops row 1 starting at columnn 1 up to the second to last
        x_i = i * h                                                            # This lets us know the position for each i point [m]
        q_i = q_triple_prime_non_uniform(q_triple_prime, x_i, R)               # This calculates our new q''' value for each position
        RHS_i = (q_i * (h ** 2)) / k                                           # This makes our RHS change for every change in position
        
        A_matrix[i, i-1]   = -1.0                                              # Coefficient of T_{i-1}
        A_matrix[i, i]     = 2.0                                               # Coefficient of T_i
        A_matrix[i, i+1]   = -1.0                                              # Coefficient of T_{i+1}
        b_vector[i]        = RHS_i                                
        
    return A_matrix, b_vector

def outer_surface_equation_non_uniform(A_matrix, b_vector, q_triple_prime, h, k, T_s, N, R):  
    """ We do not need to worry about our centerline equation because at x=0,
        it should be the same as with a constant RHS. But our outer surface 
        equation will change.
    """
    
    x = (N - 1) * h                                                            # Our last row position and should be the position next to our outer surface
    q = q_triple_prime_non_uniform(q_triple_prime, x, R)                       # This is our non uniform q''' for our last x value

    RHS = ((q * (h ** 2)) / k) + T_s   

    A_matrix[N-1, N-2] = -1.0                                                  # Coefficient of T_{N-2}
    A_matrix[N-1, N-1] = 2.0                                                   # Coefficient of T_{N-1}
    b_vector[N-1]      = RHS
    
    return A_matrix, b_vector


def ne367_function(N_list, q_triple_prime, R, k, T_s):
    """ This function will be able to solve, compare, and plot our temperature 
        data for both uniform and non-uniform models
    """
    
    T_0_analytical_uniform = T_s + ((q_triple_prime * (R**2)) / (2 * k))       # The analytical uniform solution 
    T_0_analytical_non_uniform = T_s + 0.8 * ((q_triple_prime * (R**2)) / (2 * k)) # This analytical non uniform solution is around 80% of our uniform solution 
    
    print(f"\nOur Uniform Analystical Uniform Centerline Temperatue (T_0) is: {T_0_analytical_uniform}")
    print(f"Our Non Uniform Analystical Uniform Centerline Temperatue (T_0) is: {T_0_analytical_non_uniform}\n")
   
# --------------------- STEP 7: PLOT DATA ------------------------------------
    
    grid_points_plot = []
    T_0_A_plots = []
    T_0_B_plots = [] 
    T_0_A_non_plots = []
    T_0_B_non_plots = []
    
# ----------------------------------------------------------------------------

    for current_N in N_list:
        current_h = R / current_N
        grid_points = current_N + 1
       
        print(f"For N = {current_N}, our current gid spacing is h = {current_h}")
        print(f"For N = {current_N}, our number of grid points is: {grid_points}:")
        
        grid_points_plot.append(grid_points)                                   # Part 7
        
# ---------------------- Uniform System For Approach A ------------------------
  
        A_A = np.zeros((current_N, current_N))
        b_A = np.zeros(current_N)
        A_A, b_A = centerline_equation_approachA(A_A, b_A, q_triple_prime, current_h, k)
        A_A, b_A = interior_equation(A_A, b_A, q_triple_prime, current_h, k, current_N)
        A_A, b_A = outer_surface_equation(A_A, b_A, q_triple_prime, current_h, k, T_s, current_N)
        
        T_solved_A = np.linalg.solve(A_A, b_A)
        T_0_A = T_solved_A[0]                                                  # The first value of the vector is the centerline T_0
        difference_A = abs(T_0_A - T_0_analytical_uniform)
        
        T_0_A_plots.append(T_0_A)
        
# ----------------------- Uniform System For Approach B -----------------------

        A_B = np.zeros((current_N, current_N))
        b_B = np.zeros(current_N)
        A_B, b_B = centerline_equation_approachB(A_B, b_B, q_triple_prime, current_h, k)
        A_B, b_B = interior_equation(A_B, b_B, q_triple_prime, current_h, k, current_N)
        A_B, b_B = outer_surface_equation(A_B, b_B, q_triple_prime, current_h, k, T_s, current_N)
      
        T_solved_B = np.linalg.solve(A_B, b_B)
        T_0_B = T_solved_B[0]                                                  # The first value of the vector is the centerline T_0
        difference_B = abs(T_0_B - T_0_analytical_uniform)
        
        T_0_B_plots.append(T_0_B)
        
# --------------------Non Uniform System For Approach A -----------------------

        q_nonuniform = q_triple_prime_non_uniform(q_triple_prime, 0.0, R)      # Call upon our non-unform q''' function       

        A_A_non = np.zeros((current_N, current_N))
        b_A_non = np.zeros(current_N)
        A_A_non, b_A_non = centerline_equation_approachA(A_A_non, b_A_non, q_nonuniform, current_h, k)
        A_A_non, b_A_non = interior_equation_non_uniform(A_A_non, b_A_non, q_triple_prime, current_h, k, current_N, R)
        A_A_non, b_A_non = outer_surface_equation_non_uniform(A_A_non, b_A_non, q_triple_prime, current_h, k, T_s, current_N, R)
     
        T_solved_A_non = np.linalg.solve(A_A_non, b_A_non)
        T_0_A_non = T_solved_A_non[0]                                          # The first value of the vector is the centerline T_0
        difference_A_non = abs(T_0_A_non - T_0_analytical_non_uniform)
        
        T_0_A_non_plots.append(T_0_A_non)
        
# --------------------Non Uniform System For Approach B -----------------------

        A_B_non = np.zeros((current_N, current_N))
        b_B_non = np.zeros(current_N)
        A_B_non, b_B_non = centerline_equation_approachB(A_B_non, b_B_non, q_nonuniform, current_h, k)
        A_B_non, b_B_non = interior_equation_non_uniform(A_B_non, b_B_non, q_triple_prime, current_h, k, current_N, R)
        A_B_non, b_B_non = outer_surface_equation_non_uniform(A_B_non, b_B_non, q_triple_prime, current_h, k, T_s, current_N, R)
     
        T_solved_B_non = np.linalg.solve(A_B_non, b_B_non)
        T_0_B_non = T_solved_B_non[0]    
        difference_B_non = abs(T_0_B_non - T_0_analytical_non_uniform)
        
        T_0_B_non_plots.append(T_0_B_non)

# --------------------- Residual Calculations --------------------------------

        max_residual_A = residual_check(A_A, b_A, T_solved_A)
        max_residual_B = residual_check(A_B, b_B, T_solved_B)
        max_residual_A_non = residual_check(A_A_non, b_A_non, T_solved_A_non)
        max_residual_B_non = residual_check(A_B_non, b_B_non, T_solved_B_non)
        
# ---------------------- Print Statement Per Loop -----------------------------

        print(f"\nFor Approach A Uniform with N = {current_N}:")
        print(f"Our Calculated Centerline Temperature for N = {current_N} is T_0 = {T_0_A:<15.4f}")
        print(f"Our Absolute Error Difference is: {difference_A}")
        print(f"Our Max System Residual (AT-b) is: {max_residual_A}")
        
        print(f"\nFor Approach B Uniform with N = {current_N}:")
        print(f"Our Calculated Centerline Temperature for N = {current_N} is T_0 = {T_0_B:<15.4f}")
        print(f"Our Absolute Error Difference is: {difference_B}") 
        print(f"Our Max System Residual (AT-b) is: {max_residual_B}\n")
        
        print(f"\nFor Approach A Non Uniform with N = {current_N}:")
        print(f"Our Calculated Centerline Temperature for N = {current_N} is T_0 = {T_0_A_non:<15.4f}")
        print(f"Our Absolute Error Difference is: {difference_A_non}")
        print(f"Our Max System Residual (AT-b) is: {max_residual_A_non}")
        
        print(f"\nFor Approach B Non Uniform with N = {current_N}:")
        print(f"Our Calculated Centerline Temperature for N = {current_N} is T_0 = {T_0_B_non:<15.4f}")
        print(f"Our Absolute Error Difference is: {difference_B_non}") 
        print(f"Our Max System Residual (AT-b) is: {max_residual_B_non}\n")

# ---------------------------- Plotting Uniform ---------------------------------------

    plt.figure(figsize=(9,6))
    plt.plot(grid_points_plot, T_0_A_plots, 'go-', label = 'Approach A Uniform (Ghost Point (Central Difference))', markersize=8)  # Plotting our Approach A uniformn values
    plt.plot(grid_points_plot, T_0_B_plots, 'bo-', label = 'Approach B Uniform (Direct DX (Forward Difference))', markersize=8)    # Plotting Approach B uniform values w/ asmptotic convergence curve
    plt.axhline(y = T_0_analytical_uniform, color = 'r', linestyle = '-', linewidth = 2, label = 'Analytical Uniform Solution')   # Plotting the uniform horizontal analytical line (for N = 5, it should be 1740.5 k)

# ----------------------------- Plotting Non Uniform --------------------------

    plt.plot(grid_points_plot, T_0_A_non_plots, 'g^--', label = 'Approach A Non Uniform (Ghost Point (Central Difference))', markersize=8)  # Plotting our Approach A non uniform values
    plt.plot(grid_points_plot, T_0_B_non_plots, 'bv--', label = 'Approach B Non Uniform (Direct DX (Forward Difference))', markersize=8)    # Plotting Approach B non uniform values w/ asmptotic convergence curve    
    plt.axhline(y = T_0_analytical_non_uniform, color = 'orange', linestyle = '-', linewidth = 2, label = 'Analytical Non Uniform Solution')   # Plotting the horizontal analytical line (for N = 5, it should be 1740.5 k)

# ------------------------------- Rest of Plotting Details -------------------

    plt.title('Centerline Temperature Convergence Study: Uniform vs. Non-Uniform Volumetric Heat' , fontsize = 14, fontweight = 'bold', pad = 15)
    plt.xlabel('Number of Grid Points ($N + 1$)', fontsize = 12)
    plt.ylabel('Calculated Centertline Temperature $T_0$ (K)', fontsize = 12)
    
    plt.xticks(grid_points_plot)
    plt.grid(True, linestyle=':', alpha = 0.6)
    plt.legend(loc = 'lower right', fontsize = 8, framealpha = 0.4, facecolor = 'white', edgecolor = 'gray', shadow=False)

    plt.tight_layout()
#   plt.savefig('centerline_convergence_plot.png', dpi=300)                   # In case you wanted to save your graph
    plt.show()    

# -------------------- Boundary Condition Temperature Graph --------------------
    


#------ These functions are copy/paste from HW 1 to test for Step 4 ----------

def Gaussian_elimination(A, b):                                                # A is our input matrix and b is the input vector
    # First, find the size of A and b
   
    matrix_size = len(A)
    vector_size = len(b)
    
    if matrix_size != 2 and matrix_size !=3:                                   # Write a check for if this is the appropriate matrix
        return sys.exit("Error! You did not input a 2x2 or 3x3 matrix")
    
    if vector_size != matrix_size:                                             # Write a check to confirm the vector matches the matrix
        return sys.exit("Error! Your matrix and vector does not match")
        
    # Now, lets continue with either a 2x2 or 3x3
                            
                         #### 2v2 Matrix ####
    
    if matrix_size == 2:                                                       # First, let's make our matrix box
        row0 = [A[0][0], A[0][1], b[0]]
        row1 = [A[1][0], A[1][1], b[1]]
        
        # Perform Gaussian Elimination using Partial Pivoting
       
        if abs(row0[0]) < abs(row1[0]):
            row0, row1 = row1, row0
            print("Swapped rows by using Row Permutation.")                    # For homework part 4, report any row permutations
        
        if abs(row0[0]) < 0.00000001:                                          # We check to see if the pivot is equal to zero
            sys.exit("Error! Pivot is zero!")                                  # We use a very small number instead of zero to make sure no floating point number continues
            
        # Now do the Gaussian Elimination
        
        elim_num = row1[0] / row0[0]                                           # This is our elimination number 
        
        row1[0] = row1[0] - (elim_num * row0[0])                               # Do calculations for all of row1, but keep row0 the same
        row1[1] = row1[1] - (elim_num * row0[1])                               # Row1[0] should equal zero
        row1[2] = row1[2] - (elim_num * row0[2])        
        
        # Time to finish up and extract our new matrix and vector
        
        U = [[row0[0], row0[1]],                                               # U is our upper matrix (was previously A)
             [row1[0], row1[1]]]                                               # c is our new input vector (was previously b)
        c = [row0[2], row1[2]]
        
        return U, c                                                            # Return both our new input matrix and vector
    
                     #### 3v3 Matrix ####
    
    if matrix_size == 3:
        row0 = [A[0][0], A[0][1], A[0][2], b[0]]
        row1 = [A[1][0], A[1][1], A[1][2], b[1]]
        row2 = [A[2][0], A[2][1], A[2][2], b[2]]
        
        # Do Row Permuation 
        
        if abs(row1[0]) >= abs(row0[0]) and abs(row1[0]) >= abs(row2[0]):
            row0, row1 = row1, row0
            print("Swapped Row0 and Row1 using Row Permutation")               # For homework part 4, report any row permutations
        elif abs(row2[0]) >= abs(row0[0]) and abs(row2[0]) >= abs(row1[0]):
            row0, row2 = row2, row0
            print("Swapped Row0 and Row2 using Row Permutation")               # For homework part 4, report any row permutations
            
        # Do Gaussian Elimination
        
        if abs(row0[0]) < 0.00000001:
            sys.exit("Error! Your pivot is equal to zero, which is bad!")
        
        elim_num1 = row1[0] / row0[0]                                          # Turns row1[0] into our first zero
        row1[0] = row1[0] - (elim_num1 * row0[0])
        row1[1] = row1[1] - (elim_num1 * row0[1])
        row1[2] = row1[2] - (elim_num1 * row0[2])
        row1[3] = row1[3] - (elim_num1 * row0[3])            
        
        elim_num2 = row2[0] / row0[0]                                          # Turns row2[0] into our second zero
        row2[0] = row2[0] - (elim_num2 * row0[0])
        row2[1] = row2[1] - (elim_num2 * row0[1])
        row2[2] = row2[2] - (elim_num2 * row0[2])
        row2[3] = row2[3] - (elim_num2 * row0[3])   
        
        if abs(row2[1]) > abs(row1[1]):                                        # Swap rows if row2[1] is bigger than row1[1]
            row1, row2 = row2, row1
            print("Swapped Row1 and Row2 using Row Permutation")               # For homework part 4, report any row permutations
        if abs(row1[1]) < 0.00000001:                                          # Check to see if our updated pivot number happened to be zero
            sys.exit("Error! Your pivot equals zero, which is not cool")
        
        elim_num3 = row2[1] / row1[1]                                          # Turn row2[1] into our last zero
        row2[1] = row2[1] - (elim_num3 * row1[1])
        row2[2] = row2[2] - (elim_num3 * row1[2])
        row2[3] = row2[3] - (elim_num3 * row1[3])  
        
        # Gaussian Elimination Complete! Finish new matrix and vector
        
        U = [[row0[0], row0[1], row0[2]],                                      # This is our upper matrix (was previously our A matrix)
             [row1[0], row1[1], row1[2]],
             [row2[0], row2[1], row2[2]]]
        c = [row0[3], row1[3], row2[3]]                                        # This is our new input vector (was previously b)
        
        return U, c                                                            # Returns our new input matrix and input vector
    
    
def back_substitution(U, c):
    vector_size = len(c)

    if vector_size <=1 or vector_size >= 4:                                    # Check to make sure you are working with either a 2x2 or 3x3 matrix system
        sys.exit("Error! This is not the right size!")
        
                     #### 2x2 Matrix ####
                     
    if vector_size == 2:                         
        if abs(U[1][1]) < 0.00000001:                                          # Check to make sure your bottom pivot isn't zero
            sys.exit("Error! Your bottom right pivot is zero!")
        
        x1 = c[1] / U[1][1]                                                    # Calculate your bottom variable
        
        if abs(U[0][0]) < 0.00000001:                                          # Check to make sure your top pivot isn't zero
            sys.exit("Error! Your top pivot is equal to zero!")
            
        x0 = (c[0] - (U[0][1] * x1)) / U[0][0]                                 # Calculate your top variable
        
        x = [x0, x1]
        
        return x                                                               # Return your variables as a list
    
                     #### 3x3 Matrix ####
                     
    if vector_size == 3:
        if abs(U[2][2]) < 0.00000001:                                          # Check to make sure your bottom pivot isn't zero
            sys.exit("Error! Your bottom right pivot is zero!")
        
        x2 = c[2] / U[2][2]                                                    # Calculate your bottom variable
        
        if abs(U[1][1]) < 0.00000001:
            sys.exit("Error! Your middle pivot is equal to zero!")             # Check to make sure your middle pivot isn't zero
        
        x1 = (c[1] - (U[1][2] * x2)) / U[1][1]                                 # Calculate your middle variable
        
        if abs(U[0][0]) < 0.0000001:                                           # Check to make sure your top pivot isn't zero
            sys.exit("Error! Your top left pivot is zero!")

        x0 = (c[0] - (U[0][2] * x2) - (U[0][1] * x1)) / U[0][0]
        
        x = [x0, x1, x2]                                                       # Calculate your top variable
         
        return x                                                               # Return your variables as a list   
    
    
def system_solver(A, b):                                                       # This function puts together everything we've done so far
    """ 
    INPUTS:
        A | A 2D list matrix (2x2 or 3x3) representing the left-side of the equation
        B | A 1D list vector representing the right-side of the equation

    OUTPUTS:
        x | A 1D list vector that contains our final solves variables
        
    IMPORTANT NOTES:
        - This function only accepts 2x2 or 3x3 matrices (A) and our vector (b)
        has to match, otherwise code will terminate
        - If at any point the code detects a pivot number is zero or a small 
        enough number to be essentially zero, code will terminate
"""                                                            
    U, c = Gaussian_elimination(A, b)                                          # We run our Gaussian Elim function to get our U and c 
    x = back_substitution(U, c)                                                # We run our Back Substitution function to get our variables x as a final solution list

    print(f"Your new upper matrix U is: {U}")
    print(f"Your new vector c is: {c}")
   
    return x       

# ---------- GIVEN CONSTANTS -------------------------------------------------

R = 4.10 * (10**-3)             # Outer Radius [m]
k = 3.0                         # Thermal Conductivity [W / mK] 
q_triple_prime = 3.0 * (10**8)  # Volumtetric Power [W/m^3]
T_s = 900.0                     # Boundary Temperature [k]

# ----------------- Problem Introduction ---------------------------------------

aa = "Consider steady-state, one-dimensional heat conduction in one half of a planar approximation of a nuclear fuel. The center of the fuel is at x = 0, and the outer surface is at x = R. Assume constant thermal conductivity and uniform volumetric heat generation (-k (d^2T / dx^2) = q'''). "
ab = "Using our given constants, representative of a PWR fuel pellet, apply a symmetry condition at the pellet center and a specified temperature at the outer surface. "
ac = "This problem can be solved analytically. Using Python, write a program that completes eight tasks. "
ad = "The goal is to compare two different methods (ghost point and dirichlet) to analysize our calculated T_0 value vs the analytical T_0 value. Also make a plot showing this data. "

# ---------- NE367 Problem Introduction -------                                # I am not a NE367 student, I just want to attempt this for myself. So if this is wrong, I hope I dont get graded for it D:

ba = "The neutrons born in the fuel leave the fuel domain to reach the moderator region. After scattering and moderating, the thermal neutrons come back to the fuel from the outside. "
bb = "For this reason, the probability of fission in the outter region of the fuel is higher than closer to the symmertry plane. In this exercise for NE367 students, you will consider that the volumetric-heat generation rate is not uniform anymore, but it follows a function."
bc = "Show analytically that (an integral function) so both cases have the same total heat generation per unit planar area. Perform the same grid-refinement study, plot the temperature profiles for the uniform and nonuniform volumetric-heat generation rate, and compare their centerline temperatures. "
bd = "Explain physically why moving more heat generation toward the outside changes the centerline temperature even though the total heat generation is unchanged"

# -----------------------------------------------------------------------------

user_input = input("Enter your number of elements (default is 5): ")           # This is the number of elements we will have

try:                                                                           # We convert our input into an integer
    N = int(user_input)
    h = R / N
    
    interval = divide_interval(N, h, R)
    T_unknowns = np.zeros(N)   

# ----------------- MAKE OUR MATRICES (Step 2) ---- ---------------------------

# Approach A Construction

    A_A = np.zeros((N, N))                                                     # This makes our N x N matrix with all zeros                              
    b_A = np.zeros(N)                                                          # This makes our b vector with N number of zeros
    A_A, b_A = centerline_equation_approachA(A_A, b_A, q_triple_prime, h, k)   
    A_A, b_A = interior_equation(A_A, b_A, q_triple_prime, h, k, N)
    A_A, b_A = outer_surface_equation(A_A, b_A, q_triple_prime, h, k, T_s, N)
    
# Approach B Construction

    A_B = np.zeros((N, N))                                                     # This makes our N x N matrix with all zero     
    b_B = np.zeros(N)                                                          # This makes our b vector with N number of zeros
    A_B, b_B = centerline_equation_approachB(A_B, b_B, q_triple_prime, h, k)
    A_B, b_B = interior_equation(A_B, b_B, q_triple_prime, h, k, N)
    A_B, b_B = outer_surface_equation(A_B, b_B, q_triple_prime, h, k, T_s, N)
    
# -----------------------------------------------------------------------------
    
    reflect_Step3_Question = "Explain how the Neumann and Dirichlet boundary conditions appear in the system."
    reflect_Step3_Neumann = "The Neumann Condition (Approach A) appears to have the number 2.0 as its main diagonal and there is also a -2.0 on the upper diagnoal. The first index of vector B is 67.24 (the volumetric power number)."
    reflect_Step3_Dirichlet = "The Dirichlet Condition (Approach B) appears to have the number 1.0 as its main diagonal and there is also a -1.0 on the upper diagonal. The first index of vector B is zero."
    
# -------------------- STEP 4 | MATCH HOMEWORK 1 MATRIX ----------------------
 
    A = [[0, 2, 1],
         [1, -2, -3],
         [3, -1, 2]]
    b = [3, -6, 4]
    
    x_homework1 = system_solver(A, b)
    x_numpy = np.linalg.solve(np.array(A), np.array(b))
    
    difference = np.abs(np.array(x_homework1) - x_numpy)
    
    conclusion = "In conclusion, the numpy python tool matches our custom code with 100% accuracy!"

# --------------------- STEP 5 | SOLVE OUR N = {N} SYSTEM ---------------------

    T_numpyA = np.linalg.solve(np.array(A_A), np.array(b_A))                   # Constructs our unknnown T vector without T_s
    T_numpyB = np.linalg.solve(np.array(A_B), np.array(b_B))                   # Constructs our unknown T vector without T_s
    
    T_vectorA = np.append(T_numpyA, T_s)                                       # We appennd T_s to the end of our temperature list
    T_vectorB = np.append(T_numpyB, T_s)                                       # We appennd T_s to the end of our temperature list
    
# --------------------- STEP 6 | FIVE SIMULATIONS -----------------------------

    simulation_list = step_6_Nlist(N)                                          # All the action happens in the two step_6 function in the functions area

# ----------------------- STEP 7 | PLOTTING ----------------------------------

    a = "The results of our graph show that the Centerline Temperature T_0 for Approach A (ghost point) for all of our simulations was the exact same as our analytical solution. "
    b = "For the Centerline Temperature T_0 for Approach B (Dirichlet BC), we see that as we increase our N, our calculated solution gets closer and closer on a curve under our horizontal analytical solution. "
    c = "This shows that for Approach B, this system will eventually converge towards the analytical solution."

# ----------------------- STEP 8: RESIDUAL -----------------------------------

    A = "The maximum system residuals for both approaches have very small numbers (very close to zero) across all of our simulations. "
    B = "This is what I expected because residuals should return zero if we solved our system correctly, which in this case, it seems like we did! "
    C = "The numpy code we used to find that maximum absolute residual did work as intended. "

# ------------------------ NE367 Students ------------------------------------   # I am not a NE367 student, I just want to attempt this for myself
    
    simulation_list_ne367 = step_6_Nlist(N)
    
    aaa = "Moving the heat generation towards the outer radius of the fuel pellet alters the centerline temperature because of thermal resistance scales with conduction distance. "
    aab = "In the core, a large amount of energy is deposited, and as it escapes into the surrounding coolant, it has to conduct through the entire thickness of the fuel material. "
    aac = "The thermal conductivity (k) of the fuel will resist that energy movement, which results in a temperature gradient and a peak of 1740.5 K. "
    aad = "When we use a non-uniform model, we are shifting the fission reactions towards the outer edge of the pellet. While the power generated stays the same, there is now a shorter route for the energy to escape and there is reduced thermal resistance. "
    aae = "Now, because the energy doesn't have to travel through the whole thickness to enter the outer wall and the coolant, it reduces the centerline temperature by (from what I found) about 80% of the uniform version. "
    
    

# ----------------------- General Information / Vocab ------------------------#    
    
    print("\n--------------------- Problem Introduction -----------------------\n")
    print(aa + ab + ac + ad)

    print("\n------------------- Step 1 Verification -------------------------")                      
    print(f"\nCalculated grid spacing h: {h} m")
    print(f"Your generated interval: {interval}")
    print(f"Number of unknowns to solve: {len(T_unknowns)}")
    print(f"Your unknown temperature vector: {T_unknowns}")
    print("")

    print(f"--------------- Step 2 & 3 Verification (N = {N}) ---------------")
    print(f"\n Approach A Neumann Boundary Condition Construction for N = {N}")
    print("Matrix A_A: ")
    print(f"{A_A}")
    print(f"Vector b_A: {b_A}")
    print(f"\n Approach B Dirichlet Boundary Condition Constructiion for N = {N}")
    print("Matrix A_B: ")
    print(f"{A_B}")
    print(f"Vector b_B: {b_B}")
    print(f"\nStep 3 Reflection: {reflect_Step3_Question}\n")
    print(reflect_Step3_Neumann + " " + reflect_Step3_Dirichlet)
    print("")
    
    print("-------- Step 4 - Solver Verification For HW1 -------------------")
    print(f"\nHomework 1 Solver for Unknown Matrix: {x_homework1}")
    print(f"Numpy linalg.solve Build-In Solver: {x_numpy}")
    print(f"The Absolute Difference Found: {difference}")
    print(f"{conclusion}")
    print("")
    
    print("----------- Step 5: Solve System Using Python Package ------------")
    print("\nNOTE: These temperature vectors DO NOT have the T_s term at the end")
    print(f"Our unknown T vector for Approach A for N = {N} is: ")
    print(f"{T_numpyA}")
    print(f"Our unknown T vector for Approach B for N = {N} is: ")
    print(f"{T_numpyB}")
    print("\nNOTE: These temperature vectors DO have the T_s term at the end")
    print(f"Our solved T vector for Approach A for N = {N} is: ")
    print(f"{T_vectorA}")
    print(f"Our solved T vector for Approach B for N = {N} is: ")
    print(f"{T_vectorB}")
    print("")
    
    print("------- Step 6: Five Simulations for Different N Values ----------")
    print(f"\nOur N vector list is: {simulation_list}") 
    simulation_plotting(A_A, b_A, q_triple_prime, simulation_list, N, R, T_s)    # Because our function doesn't return anything, printing this would result in a None message 
    print("")
    
    print("------------------ Step 7: Plotting ------------------------------")
    print("\nDiscuss the results of the graph below:\n")
    print(a + b + c)
    print("")
    
    print("----------------- Step 8: Residuals -------------------------------")
    print("\nThe results of each residual is included in our Step 6: Simulation Data section\n")
    print("Discuss The Change in the Residuals when the Number of Elements is Doubled:\n")
    print(A + B + C)
    print("")

    print("------------------- Non-Uniform Problem --------------------------\n")
    print(ba + bb + bc + bd)
    ne367_function(simulation_list_ne367, q_triple_prime, R, k, T_s)
    print("\nExplain physically why moving more heat generation toward the outside changes the centerline temperature even though the total heat generation is unchanged.\n")
    print(aaa + aab + aac + aad + aae)    
                                
except ValueError:
    print(f"{user_input} is not a valid integer")


