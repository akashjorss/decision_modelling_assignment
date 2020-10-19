# import PuLP
from pulp import*

# Create the 'prob' variable to contain the problem data
prob = LpProblem("The Nutriscore", LpMaximize)

# Create problem variables
x_1=LpVariable("aliment_1",0,20)
x_2=LpVariable("aliment_2",0, 20)
x_3=LpVariable("aliment_3",0,20)
x_4=LpVariable("aliment_4",0, 20)
x_5=LpVariable("aliment_5",0, 20)
x_6=LpVariable("aliment_6",0, 20)
x_7=LpVariable("aliment_7",0, 20)
x_8=LpVariable("aliment_8",0, 20)

# Variables measuring differences between two consecutives classes
epsilon_1=LpVariable("epsilon_1",1, 20)
epsilon_2=LpVariable("epsilon_2",1, 20)
epsilon_3=LpVariable("epsilon_3",1, 20)
epsilon_4=LpVariable("epsilon_4",1, 20)

# Variables associated to the marginal utility functions of food 1 x_1

U1_1669=LpVariable("utilite_1_alim_1",0, 20)
U2_2v37=LpVariable("utilite_2_alim_1",0, 20)
U3_26v3=LpVariable("utilite_3_alim_1",0, 20)
U4_328=LpVariable("utilite_4_alim_1",0, 20)
U5_7v99=LpVariable("utilite_5_alim_1",0, 20)
U6_4v83=LpVariable("utilite_6_alim_1",0, 20)

# Variables associated to the marginal utility functions of food 2 x_2

U1_1962=LpVariable("utilite_1_alim_2",0, 20)
U2_11=LpVariable("utilite_2_alim_2",0, 20)
U3_31=LpVariable("utilite_3_alim_2",0, 20)
U4_0v12=LpVariable("utilite_4_alim_2",0, 20)
U5_5v8=LpVariable("utilite_5_alim_2",0, 20)
U6_0=LpVariable("utilite_6_alim_2",0, 20)

# Variables associated to the marginal utility functions of food 3 x_3

U1_459=LpVariable("utilite_1_alim_3",0, 20)
U2_1v8=LpVariable("utilite_2_alim_3",0, 20)
U3_13v4=LpVariable("utilite_3_alim_3",0, 20)
U4_0v1=LpVariable("utilite_4_alim_3",0, 20)
U5_6v5=LpVariable("utilite_5_alim_3",0, 20)
U6_0v6=LpVariable("utilite_6_alim_3",0, 20)

# Variables associated to the marginal utility functions of food 4 x_4

U1_741=LpVariable("utilite_1_alim_4",0, 20)
U2_6v1=LpVariable("utilite_2_alim_4",0, 20)
U3_18v7=LpVariable("utilite_3_alim_4",0, 20)
U4_60=LpVariable("utilite_4_alim_4",0, 20)
U5_3v6=LpVariable("utilite_5_alim_4",0, 20)

# Variables associated to the marginal utility functions of food 5 x_5

U1_490=LpVariable("utilite_1_alim_5",0, 20)
U2_1=LpVariable("utilite_2_alim_5",0, 20)
U3_0v7=LpVariable("utilite_3_alim_5",0, 20)
U4_0v76=LpVariable("utilite_4_alim_5",0, 20)
U5_22=LpVariable("utilite_5_alim_5",0, 20)

# Variables associated to the marginal utility functions of food 6 x_6

U1_477=LpVariable("utilite_1_alim_6",0, 20)
U3_0v5=LpVariable("utilite_3_alim_6",0, 20)
U4_0v56=LpVariable("utilite_4_alim_6",0, 20)

# Variables associated to the marginal utility functions of food 7 x_7

U1_109=LpVariable("utilite_1_alim_7",0, 20)
U2_0=LpVariable("utilite_2_alim_7",0, 20)
U3_3v6=LpVariable("utilite_3_alim_7",0, 20)
U4_0v2=LpVariable("utilite_4_alim_7",0, 20)
U5_1v4=LpVariable("utilite_5_alim_7",0, 20)

# Variables associated to the marginal utility functions of food 8 x_8

U1_188=LpVariable("utilite_1_alim_8",0, 20)
U2_0v6=LpVariable("utilite_2_alim_8",0, 20)
U3_5v1=LpVariable("utilite_3_alim_8",0, 20)
U4_0v04=LpVariable("utilite_4_alim_8",0, 20)
U5_3v8=LpVariable("utilite_5_alim_8",0, 20)




# The objective function is added to 'prob' first
prob +=epsilon_1+epsilon_2+epsilon_3+epsilon_4, "slack variables (differences between two consecutive classes) to be maximized"


# The eight constraints associated to global utility of each food

prob += U1_1669 + U2_2v37 + U3_26v3 + U4_328  + U5_7v99 + U6_4v83 ==x_1, "aliment_1 constraint" 
prob += U1_1962 + U2_11   + U3_31   + U4_0v12 + U5_5v8  + U6_0    ==x_2, "aliment_2 constraint"
prob += U1_459  + U2_1v8  + U3_13v4 + U4_0v1  + U5_6v5  + U6_0v6  ==x_3, "aliment_3 constraint" 
prob += U1_741  + U2_6v1  + U3_18v7 + U4_60   + U5_3v6  + U6_0    ==x_4, "aliment_4 constraint"  
prob += U1_490  + U2_1    + U3_0v7  + U4_0v76 + U5_22   + U6_0    ==x_5, "aliment_5 constraint" 
prob += U1_477  + U2_1    + U3_0v5  + U4_0v56 + U5_22   + U6_0    ==x_6, "aliment_6 constraint"  
prob += U1_109  + U2_0    + U3_3v6  + U4_0v2  + U5_1v4  + U6_0v6  ==x_7, "aliment_7 constraint"  
prob += U1_188  + U2_0v6  + U3_5v1  + U4_0v04 + U5_3v8  + U6_0    ==x_8, "aliment_8 constraint"  



# Foods of class A are better than foods of class B

prob += x_3 + epsilon_1 <= x_8, " aliment 3 classe apres aliment 8"
prob += x_3 + epsilon_1 <= x_5, " aliment 3 classe apres aliment 5"
prob += x_3 + epsilon_1 <= x_7, " aliment 3 classe apres aliment 7"
prob += x_3 + epsilon_1 <= x_6, " aliment 3 classe apres aliment 6"

# Foods of class B are better than foods of class C

prob += x_1 + epsilon_2 <= x_3, " aliment 1 classe apres aliment 3"

# Foods of class C are better than foods of class D

prob += x_4 + epsilon_3 <= x_1, " aliment 4 classe apres aliment 1"

# Foods of class D are better than foods of class E

prob += x_2 + epsilon_4 <= x_4, " aliment 2 classe apres aliment 4"



# Monotonicity constraints associated to the values of criterion 1

prob += U1_1962  <= U1_1669, " utilite  U1_1962 classe apres utilite U1_1669"
prob += U1_1669  <= U1_741, " utilite  U1_1669 classe apres utilite U1_741"
prob += U1_741   <= U1_490, " utilite  U1_741 classe apres utilite U1_490"
prob += U1_490   <= U1_477 , " utilite  U1_490 classe apres utilite U1_477"
prob += U1_477   <= U1_459 , " utilite  U1_477 classe apres utilite U1_459"
prob += U1_459   <= U1_188 , " utilite  U1_459 classe apres utilite U1_188"
prob += U1_188   <= U1_109 , " utilite  U1_188 classe apres utilite U1_109"

# Monotonicity constraints associated to the values of criterion 2

prob += U2_11    <= U2_6v1, " utilite  U2_11 classe apres utilite U2_6v1"
prob += U2_6v1   <= U2_2v37, " utilite  U2_6v1 classe apres utilite U2_2v37"
prob += U2_2v37  <= U2_1v8, " utilite  U2_2v37 classe apres utilite U2_1v8"
prob += U2_1v8   <= U2_1, " utilite  U2_1v8 classe apres utilite U2_1"
prob += U2_1     <= U2_0v6, " utilite  U2_1 classe apres utilite U2_0v6"
prob += U2_0v6   <= U2_0, " utilite  U2_0v6 classe apres utilite U2_0"



# Monotonicity constraints associated to the values of criterion 3

prob += U3_31    <= U3_26v3, " utilite  U3_31 classe apres utilite UU3_26v3"
prob += U3_26v3  <= U3_18v7, " utilite  U3_26v3 classe apres utilite U3_18v7"
prob += U3_18v7  <= U3_13v4, " utilite  U3_18v7 classe apres utilite U3_13v4"
prob += U3_13v4  <= U3_5v1 , " utilite  U3_13v4 classe apres utilite U3_5v1"
prob += U3_5v1   <= U3_3v6 , " utilite  U3_5v1 classe apres utilite U3_3v6"
prob += U3_3v6   <= U3_0v7 , " utilite  U3_3v6 classe apres utilite U3_0v7"
prob += U3_0v7   <= U3_0v5 , " utilite  U3_0v7 classe apres utilite U3_0v5"


# Monotonicity constraints associated to the values of criterion 4

prob += U4_328   <= U4_60, " utilite  U4_328 classe apres utilite U4_60"
prob += U4_60    <= U4_0v76, " utilite  U4_60 classe apres utilite U4_0v76"
prob += U4_0v76  <= U4_0v56, " utilite  U4_0v76 classe apres utilite U4_0v56"
prob += U4_0v56  <= U4_0v2 , " utilite  U4_0v56 classe apres utilite U4_0v2"
prob += U4_0v2   <= U4_0v12 , " utilite  U4_0v2 classe apres utilite U4_0v12"
prob += U4_0v12  <= U4_0v1 , " utilite  U4_0v12 classe apres utilite U4_0v1"
prob += U4_0v1   <= U4_0v04 , " utilite  U4_0v1 classe apres utilite U4_0v04"


# Monotonicity constraints associated to the values of criterion 5

prob += U5_1v4   <= U5_3v6, " utilite  U5_1v4 classe apres utilite U5_3v6"
prob += U5_3v6   <= U5_3v8, " utilite  U5_3v6 classe apres utilite U5_3v8"
prob += U5_3v8   <= U5_5v8, " utilite  U5_3v8 classe apres utilite U5_5v8"
prob += U5_5v8   <= U5_6v5 , " utilite  U5_5v8 classe apres utilite U5_6v5"
prob += U5_6v5   <= U5_7v99 , " utilite  U5_6v5 classe apres utilite U5_7v99"
prob += U5_7v99  <= U5_22 , " utilite  U5_7v99 classe apres utilite U5_22"

# Monotonicity constraints associated to the values of criterion 6

prob += U6_0    <= U6_0v6, " utilite  U6_0  classe apres utilite U6_0v6"
prob += U6_0v6  <= U6_4v83, " utilite  U6_0v6 classe apres utilite U6_4v83"




# The problem data is written to an .lp file
prob.writeLP("The Nutriscore.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print("Statut:", LpStatus[prob.status])
# Output= 
# Status: Optimal

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)


# The optimised objective function value is printed to the screen
print("Valeur fonction objectif = ", value(prob.objective))
