//============================================================================
// Name        : dijkstraMPI.cpp
// Author      : Alexander Diop
// Created     : 11/4/21
// Version     : 0.0.1
// Project     : MPI_Dijkstra_IT388_
// Description : Parallel implementation of Dijkstra's shortest path algorithm
//               using MPI.
//
// TO DO       : Measure Timing
//               Test memory management
//               Test Large N graphs (30,000+ nodes)
//
//
// Compile:  mpicxx -o dijkstraMPI dijkstraMPI.cpp
// Run:      mpirun -np <p> ./dijkstraMPI <graphFile>
//============================================================================

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mpi.h>
#include <fstream>

using namespace std;

#define INFINITY 1000000

MPI_Datatype CreateBlockColumn(int n, int localn);
void SpreadMatrix(int* mat, int loc_mat[], int n, int loc_n,  MPI_Datatype MPI_BLOCK_TYPE, int my_rank, MPI_Comm comm);
//void Print_local_matrix(int loc_mat[], int n, int loc_n, int my_rank);
//void Print_matrix(int loc_mat[], int n, int loc_n, MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm);
void Dijkstra(int mat[], int dist[], int localPredecessor[], int n, int loc_n, int my_rank, MPI_Comm comm);
void InitMatrix(const int mat[], int loc_dist[], int localPredecessor[], int known[], int loc_n, int my_rank);
int FindMinDistance(const int loc_dist[], const int known[], int loc_n);
int GlobalVertex(int localVertex, int localNumVertices, int my_rank);
void PrintDistances(map<int, string> &indexToVertex, int loc_dist[], int n, int loc_n, int my_rank, MPI_Comm comm);
void PrintPaths(map<int, string> &indexToVertex, int localPredecessor[], int n, int loc_n, int my_rank, MPI_Comm comm);


int main(int argc, char* argv[]) {
    int *matrix = nullptr, *loc_mat, *loc_dist, *loc_pred;
    int numVertices, loc_n, numProcs, my_rank;
    map<string, int> vertexToIndex;
    map<int, string> indexToVertex;
    MPI_Comm comm;
    MPI_Datatype MPI_BLOCK_TYPE;

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &numProcs);
    MPI_Comm_rank(comm, &my_rank);

    //Check if the number of arguments is correct
    if(argc < 2)
    {
        cout << "Usage: ./serial_sp <inputFile>" << endl;
        exit(1);
    }

    string filename = argv[1];
//  start = Get_time();
    fstream graphFile(filename); //read file in here
    if(graphFile.fail()){
        cout << "Error in reading graph file " << endl;
        exit(1);
    }else{

        if(my_rank == 0){

            string line;
            int numberOfEdges;

            graphFile >> numVertices;
            matrix = static_cast<int *>(malloc(numVertices * numVertices * sizeof(int)));

            //Set up empty matrix and read node names into my maps to be able to translate
            for(int i = 0; i < numVertices; i++){
                for(int j=0; j < numVertices; j++){
                    matrix[(i*numVertices) + j] = INFINITY;
                }
                graphFile >> line;
                vertexToIndex[line] = i;
                indexToVertex[i] = line;
            }

            graphFile >> numberOfEdges;

            //Used to read edges into the matrix
            int from;
            int to;
            int cost;

            //Read edges into the matrix
            for(int i = 0; i < numberOfEdges; i++){
                graphFile >> line; //from node
                from = vertexToIndex[line];

                graphFile >> line; //to node
                to = vertexToIndex[line];

                graphFile >> cost; //cost
                matrix[(from * numVertices) + to] = cost;
                matrix[(to * numVertices) + from] = cost;
            }

            graphFile.close();
        }

        MPI_Barrier(comm);
        MPI_Bcast(&numVertices, 1, MPI_INT, 0, comm);
    }



    loc_n = numVertices / numProcs;
    loc_mat = static_cast<int *>(malloc(numVertices * loc_n * sizeof(int)));
    loc_dist = static_cast<int *>(malloc(numVertices * loc_n * sizeof(int)));
    loc_pred = static_cast<int *>(malloc(numVertices * loc_n * sizeof(int)));

    MPI_BLOCK_TYPE = CreateBlockColumn(numVertices, loc_n);//NEW MPI DATATYPE

    SpreadMatrix(matrix, loc_mat, numVertices, loc_n, MPI_BLOCK_TYPE, my_rank, comm);
    Dijkstra(loc_mat, loc_dist, loc_pred, numVertices, loc_n, my_rank, comm);

    PrintDistances(indexToVertex, loc_dist, numVertices, loc_n, my_rank, comm);
    PrintPaths(indexToVertex, loc_pred, numVertices, loc_n, my_rank, comm);

    free(loc_mat);
    free(loc_dist);
    free(loc_pred);

    MPI_Type_free(&MPI_BLOCK_TYPE);

    MPI_Finalize();
    return 0;
}

/*---------------------------------------------------------------------
 * Function:  CreateBlockColumn
 * Purpose:   Build an MPI_Datatype that represents a block column of
 *            a matrix
 * In args:   n:  number of rows in the matrix and the block column
 *            localn = n/p:  number cols in the block column
 * Returns:   blk_col_mpi_t:  MPI_Datatype that represents a block
 *            column
 */
MPI_Datatype CreateBlockColumn(int n, int localn) {

    MPI_Aint lowerBound, extent; //holds memory addresses
    MPI_Datatype MPI_BLOCK_TYPE;
    MPI_Datatype firstVector;
    MPI_Datatype blockColumnVector;

    MPI_Type_contiguous(localn, MPI_INT, &MPI_BLOCK_TYPE);//make a new datatype
    MPI_Type_get_extent(MPI_BLOCK_TYPE, &lowerBound, &extent);//get the upper (extent) and lowerBound of the new data type

    MPI_Type_vector(n, localn, n, MPI_INT, &firstVector);//creates an MPI_INT vector (firstVector) of n count, with localn blocklength
    MPI_Type_create_resized(firstVector, lowerBound, extent, &blockColumnVector);//create a new vector (blockColumnVector) with a new lowerBound and extent
    MPI_Type_commit(&blockColumnVector);//Makes (blockColumnVector) a formal communication

    MPI_Type_free(&MPI_BLOCK_TYPE);
    MPI_Type_free(&firstVector);

    return blockColumnVector;
}  /* CreateBlockColumn */


/*---------------------------------------------------------------------
 * Function:  SpreadMatrix
 * Purpose:   Distribute it matrix among the processes so that each
 *            process gets a block column with n rows and n/p
 *            columns
 *   Input:   n:  the number of rows in the matrix and the sub-matrices
 *            loc_n = n/p:  the number of columns in the sub-matrices
 *            MPI_BLOCK_TYPE:  the MPI_Datatype used on process 0
 *            my_rank:  rank of current proc
 *            comm:  Communicator consisting of all the processes
 *   Output:  loc_mat:  the calling process' sub-matrix (needs to be
 *               allocated by the caller)
 */
void SpreadMatrix(int* mat, int loc_mat[], int n, int loc_n, MPI_Datatype MPI_BLOCK_TYPE, int my_rank, MPI_Comm comm) {

    MPI_Scatter(mat, 1, MPI_BLOCK_TYPE, loc_mat, n * loc_n, MPI_INT, 0, comm);

    if (my_rank == 0)//TO DO: perhaps all the processes need to do this
        free(mat);
}



/*-------------------------------------------------------------------
 * Function:    Dijkstra
 * Purpose:     Apply Dijkstra's algorithm to the matrix mat
 * In args:     mat: sub matrix
 *              n:  the number of vertices
 *              loc_n: size of loc arrays
 *              my_rank: rank of process
 *              MPI_Comm = MPI Communicator
 * In/Out args: loc_dist: loc dist array
 *              localPredecessor: loc pred array
 */
void Dijkstra(int mat[], int loc_dist[], int localPredecessor[], int n, int loc_n, int my_rank, MPI_Comm comm) {

    int i, vertex, *known, new_dist;
    int loc_u, loc_v;

    known = static_cast<int *>(malloc(loc_n * sizeof(int)));

    InitMatrix(mat, loc_dist, localPredecessor, known, loc_n, my_rank);

    for (i = 1; i < n; i++) {
        loc_u = FindMinDistance(loc_dist, known, loc_n);

        int my_min[2], globalMinimumVal[2];
        int globalMinDistance;

        if (loc_u < INFINITY) {
            my_min[0] = loc_dist[loc_u];
            my_min[1] = GlobalVertex(loc_u, loc_n, my_rank);
        } else {
            my_min[0] = INFINITY;
            my_min[1] = INFINITY;
        }

        MPI_Allreduce(my_min, globalMinimumVal, 1, MPI_2INT, MPI_MINLOC, comm);//locate min value from ALL the proceses
        vertex = globalMinimumVal[1];
        globalMinDistance = globalMinimumVal[0];

        //The process that has the vertex flags it as known
        if (vertex / loc_n == my_rank) {
            loc_u = vertex % loc_n;
            known[loc_u] = 1;
        }

        //Update the current distance
        for (loc_v = 0; loc_v < loc_n; loc_v++)
            if (!known[loc_v]) {
                new_dist = globalMinDistance + mat[vertex * loc_n + loc_v];
                if (new_dist < loc_dist[loc_v]) {
                    loc_dist[loc_v] = new_dist;
                    localPredecessor[loc_v] = vertex;
                }
            }
    }
    free(known);
}


/*--------------------------------------------------------------------
 * Function:    InitMatrix
 * Purpose:     Initialize loc_dist, known, and localPredecessor arrays
 *   Argus:     mat: matrix
 *              loc_dist: loc distance array
 *              localPredecessor: loc predecessor array
 *              known: known array
 */
void InitMatrix(const int mat[], int loc_dist[], int localPredecessor[], int known[], int loc_n, int my_rank) {

    for (int v = 0; v < loc_n; v++) {
        loc_dist[v] = mat[0*loc_n + v];
        localPredecessor[v] = 0;
        known[v] = 0;
    }

    if (my_rank == 0) {
        known[0] = 1;
    }
}


///*-------------------------------------------------------------------
// * Function:    FindMinDistance
// * Purpose:     Find the vertex u with minimum distance to 0
// *              (dist[u]) among the vertices whose distance
// *              to 0 is not known.
// * In args:     loc_dist:   loc distance array
// *              loc_known:  loc known array
// *              loc_n:      size of loc arrays
// * Out args:    local vertex
// */
int FindMinDistance(const int loc_dist[], const int loc_known[], int loc_n) {
    int loc_v, loc_u;
    int loc_min_dist = INFINITY;

    loc_u = INFINITY;
    for (loc_v = 0; loc_v < loc_n; loc_v++)
        if (!loc_known[loc_v])
            if (loc_dist[loc_v] < loc_min_dist) {
                loc_u = loc_v;
                loc_min_dist = loc_dist[loc_v];
            }

    return loc_u;
}


/*-------------------------------------------------------------------
 * Function:    GlobalVertex
 * Purpose:     Convert local vertex to global vertex
 * In args:     localVertex:     local vertex
 *              localNumVertices:     local number of vertices
 *              my_rank:   rank of process
 * Out args:    global_u:  global vertex
 */
int GlobalVertex(int localVertex, int localNumVertices, int my_rank) {
    int globalVertex = localVertex + my_rank * localNumVertices;
    return globalVertex;
}


/*-------------------------------------------------------------------
 * Function:    PrintDistances
 * Purpose:     Print the length of the shortest path from 0 to each
 *              vertex
 * In args:     n:  the number of vertices
 *              dist:  distances from 0 to each vertex v:  dist[v]
 *                 is the length of the shortest path 0->v
 */
void PrintDistances(map<int, string> &indexToVertex, int loc_dist[], int n, int loc_n, int my_rank, MPI_Comm comm) {

    int v;
    int* distances = nullptr;

    if (my_rank == 0) {
        distances = static_cast<int *>(malloc(n * sizeof(int)));
    }

    MPI_Gather(loc_dist, loc_n, MPI_INT, distances, loc_n, MPI_INT, 0, comm);

    if (my_rank == 0) {
        printf("The distance from %s to each vertex is:\n", indexToVertex[0].c_str());
        printf("  V    Dist %s -> V\n", indexToVertex[0].c_str());
        printf("----   ------------\n");

        for (v = 1; v < n; v++){
            printf("%3s       %4d\n", indexToVertex[v].c_str(), distances[v]);
        }
        printf("\n");
        free(distances);
    }
}


/*-------------------------------------------------------------------
 * Function:    PrintPaths
 * Purpose:     Print the shortest path from 0 to each vertex
 * In args:     n:  the number of vertices
 *              pred:  list of predecessors:  pred[v] = u if
 *              u precedes v on the shortest path 0->v
 */
void PrintPaths(map<int, string> &indexToVertex, int localPredecessor[], int n, int loc_n, int my_rank, MPI_Comm comm) {
    int v, w, *path, count, i;

    int* pred = nullptr;

    if (my_rank == 0) {
        pred = static_cast<int *>(malloc(n * sizeof(int)));
    }

    MPI_Gather(localPredecessor, loc_n, MPI_INT, pred, loc_n, MPI_INT, 0, comm);

    if (my_rank == 0) {
        path = static_cast<int *>(malloc(n * sizeof(int)));

        printf("The shortest path from %s to each vertex is:\n", indexToVertex[0].c_str());
        printf("  V     Path %s -> V\n", indexToVertex[0].c_str());
        printf("----    -----------\n");
        for (v = 0; v < n; v++) {
            printf("%3s:    ", indexToVertex[v].c_str());
            count = 0;
            w = v;
            while (w != 0) {
                path[count] = w;
                count++;
                w = pred[w];
            }
            printf("%s ", indexToVertex[0].c_str());
            for (i = count-1; i >= 0; i--)
                printf("%s ", indexToVertex[path[i]].c_str());
            printf("\n");
        }

        free(path);
        free(pred);
    }
}
