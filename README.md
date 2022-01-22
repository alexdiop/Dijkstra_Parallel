# Dijkstra_Parallel
Parallel implementation of Dijkstra's shortest path algorithm using MPI

Compile:  mpicxx -o dijkstraMPI dijkstraMPI.cpp
Run:      mpirun -np <p> ./dijkstraMPI <graphFile>

#Format of Graph File:

Num of Vertices
Vertex 1
Vertex 2
Vertex 3
.
.
.
Vertex n
Num of Edges
Edge 1
Edge 2
Edge 2
.
.
.
Edge n

#EXAMPLE Graph File:
8
A
B
C
D
F
G
H
J
11
A C 2
A B 4
A G 7
B D 2
C F 8
C G 3
D G 5
D H 6
F J 3
G J 4
H J 2
