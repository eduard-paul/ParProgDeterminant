#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <ctime>

int ProcNum = 0; // The number of the available processes
int ProcRank = 0; // The rank of the current process

// Function for random initialization of the matrix and the vector elements
void RandomDataInitialization(double* pMatrix, double* pVector, int Size) {
    int i, j; // Loop variables
    srand(unsigned(time(0)));
    for (i=0; i<Size; i++) {
        pVector[i] = rand()/double(1000);
        for (j=0; j<Size; j++) {
            if (j <= i)
                pMatrix[i*Size+j] = rand()/double(1000); 
            else
                pMatrix[i*Size+j] = 0;
        }
    }
} 

// Function for memory allocation and data initialization
void ProcessInitialization(double* &pMatrix, double* &pVector, double* &pResult, double* &pProcRows, double* &pProcVector, double* &pProcResult, int &Size, int &RowNum){  
    if (ProcRank == 0) {
        do {
            printf("\nEnter the size of the matrix and the vector: ");
            scanf("%d", &Size);
            if (Size < ProcNum) {
                printf ("Size must be greater than number of processes! \n");
            }
        } while (Size < ProcNum);
    }
    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int RestRows = Size;
    for (int i=0; i<ProcRank; i++)
        RestRows = RestRows-RestRows/(ProcNum-i);
    RowNum = RestRows/(ProcNum-ProcRank);
    pProcRows = new double [RowNum*Size];
    pProcVector = new double [RowNum];
    pProcResult = new double [RowNum];
    if (ProcRank == 0) {
        pMatrix = new double [Size*Size];
        pVector = new double [Size];
        pResult = new double [Size];
    } 
    //Initialization of the matrix and the vector elements
    RandomDataInitialization (pMatrix, pVector, Size); 
} 

void PrintMatrix(double *pMatrix, int Size1, int Size2){
    for (int i=0; i<Size1; i++) {
        for (int j=0; j<Size2; j++) {
            std::cout<<pMatrix[i*Size2+j]<<"\t"; 
        }
        std::cout<<std::endl;
    }
}

void PrintVector(double *pMatrix, int Size){
    for (int i=0; i<Size; i++) {
        std::cout<<pMatrix[i]<<"\t"; 
    }
    std::cout<<std::endl;
}

void main(int argc, char* argv[]) {
    double* pMatrix; // The matrix of the linear system
    double* pVector; // The right parts of the linear system
    double* pResult; // The result vector
    double *pProcRows; // The Rows of matrix A on the process
    double *pProcVector; // The Elements of vector b on the process
    double *pProcResult; // The Elements of vector x on the process
    int Size; // The Sizes of the initial matrix and vector
    int RowNum; // The Number of the matrix rows on the current
    // process
    double Start, Finish, Duration; 
    setvbuf(stdout, 0, _IONBF, 0);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    if (ProcRank == 0)
        printf("Parallel Gauss algorithm for solving linear systems\n"); 
    ProcessInitialization(pMatrix, pVector, pResult, pProcRows, pProcVector,
        pProcResult, Size, RowNum);
    if (ProcRank == 0) {
        printf("Initial matrix \n");
        PrintMatrix(pMatrix, Size, Size);
        printf("Initial vector \n");
        PrintVector(pVector, Size);
    } 
    MPI_Finalize();
} 