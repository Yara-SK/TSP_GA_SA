# TSP_GA_SA
Traveling Salesman Problem solved with Genetic Algorithm and Simulated Annealing

1. Problem Statement
There are a set of customers, and a postman. This postman starts from node (0,0), and visits some customers. Suppose there is a road between each pair of nodes. The distance between each pair of two nodes is the Euclidean distance of such two nodes (reserve to 0.01 precision). 

Design the shortest route (visiting sequence to customers) for this postman, such that:
1) Among customers #5, 10, 15, 20, 25, 30, 35, 40, the postman needs to visit at least three nodes of them. For example, if he visits 5 and 10 and 20 then (15, 25, 30, 35, 40) can be ignored.
2) Among customers #2, 4, 6, 8, 10, 20, 22, 32, 33, 35 the postman needs to visit at least five nodes. 
3) For other customers, each one must be visited. 

The locations of each customer can be seen in excel file. You should give the route of postman and the total distance. For example, you can give the route as following format: 
(0,0)-customer 2-customer 5-customer 7- … …- customer 39- (0,0)
Total travel distance is *****. 

2. Solutions
Solutions were obtained by the use of 2 algorithms that are based on natural phenomena, namely Genetic Algorithm (GA_TSP.py) and Simulated Annealing (SA_TSP.py)
