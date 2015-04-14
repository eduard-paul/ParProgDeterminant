#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <ctime>

const double EPS = 0.00001;

int ProcNum = 0; // The number of the available processes
int ProcRank = 0; // The rank of the current process
int* pProcInd;
int* pProcNum; 
int *pParallelPivotPos; // The number of rows selected as the pivot ones
int *pProcPivotIter; // The number of iterations, at which the processor
// rows were used as the pivot ones 

// Function for random initialization of the matrix and the vector elements
void RandomDataInitialization(double* pMatrix, double* pVector, int Size) {
    int i, j; // Loop variables
    srand(unsigned(time(0)));
    for (i=0; i<Size; i++) {
        pVector[i] = rand()/double(1000);
        for (j=0; j<Size; j++) {
            if (j <= i)
                pMatrix[i*Size+j] = rand()/double(1000); 
            //pMatrix[i*Size+j] = 1; 
            else
                pMatrix[i*Size+j] = 0;
        }
    }
} 

// Function for memory allocation and data initialization
void ProcessInitialization(double* &pMatrix, double* &pVector, double* &pResult, double* &pProcRows, double* &pProcVector, double* &pProcResult, int &Size, int &RowNum){  
    if (ProcRank == 0) {
        do {
            /*printf("\nEnter the size of the matrix and the vector: ");
            scanf("%d", &Size);*/
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
        //Initialization of the matrix and the vector elements
        RandomDataInitialization (pMatrix, pVector, Size); 
    }     
} 

// Function for computational process termination
void ProcessTermination(double* pMatrix, double* pVector, double* pResult,double*pProcRows,double*pProcVector,double*pProcResult)
{
    if (ProcRank == 0) {
        delete [] pMatrix;
        delete [] pVector;
        delete [] pResult;
    }
    delete [] pProcRows;
    delete [] pProcVector;
    delete [] pProcResult;
} 

// Data distribution among the processes
void DataDistribution(double* pMatrix, double* pProcRows, double* pVector, double* pProcVector, int Size, int RowNum) {
    int *pSendNum; // The number of the elements sent to the process
    int *pSendInd; // The index of the first data element sent
    // to the process
    int RestRows=Size; // The number of rows, that have not been
    // distributed yet
    int i; // Loop variable
    // Alloc memory for temporary objects
    pSendInd = new int [ProcNum];
    pSendNum = new int [ProcNum];
    // Define the disposition of the matrix rows for the current process
    RowNum = (Size/ProcNum);
    pSendNum[0] = RowNum*Size;
    pSendInd[0] = 0;
    for (i=1; i<ProcNum; i++) {
        RestRows -= RowNum;
        RowNum = RestRows/(ProcNum-i);
        pSendNum[i] = RowNum*Size;
        pSendInd[i] = pSendInd[i-1]+pSendNum[i-1];
    }
    // Scatter the rows
    MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcRows,
        pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Free the memory
    delete [] pSendNum;
    delete [] pSendInd; 

    pProcInd = new int [ProcNum];
    pProcNum = new int [ProcNum];
    RestRows = Size;
    pProcInd[0] = 0;
    pProcNum[0] = Size/ProcNum;
    for (int i=1; i<ProcNum; i++) {
        RestRows -= pProcNum[i-1];
        pProcNum[i] = RestRows/(ProcNum-i);
        pProcInd[i] = pProcInd[i-1]+pProcNum[i-1];
    }
    MPI_Scatterv(pVector, pProcNum, pProcInd, MPI_DOUBLE, pProcVector,
        pProcNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD); 
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

// Function for testing the data distribution
void TestDistribution(double* pMatrix, double* pVector, double* pProcRows,double* pProcVector,int Size, int RowNum) {
    if (ProcRank == 0) {
        printf("Initial Matrix: \n");
        PrintMatrix(pMatrix, Size, Size);
        printf("Initial Vector: \n");
        PrintVector(pVector, Size);
    }
    for (int i=0; i<ProcNum; i++) {
        if (ProcRank == i) {
            printf("\nProcRank = %d \n", ProcRank);
            printf(" Matrix Stripe:\n");
            PrintMatrix(pProcRows, RowNum, Size);
            printf(" Vector: \n");
            PrintVector(pProcVector, RowNum);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// Fuction for column elimination
void ParallelEliminateColumns(double* pProcRows, double* pProcVector, double* pPivotRow, int Size, int RowNum, int Iter) {
    //#pragma omp parallel for
    for (int i=0; i<RowNum; i++) {
        if (pProcPivotIter[i] == -1) {
            double PivotFactor = pProcRows[i*Size+Iter] / pPivotRow[Iter];
            for (int j=Iter; j<Size; j++) {
                pProcRows[i*Size + j] -= PivotFactor* pPivotRow[j];
            }
            pProcVector[i] -= PivotFactor * pPivotRow[Size];
        }
    }
} 

// Gaussian elimination
void ParallelGaussianElimination(double* pProcRows, double* pProcVector,int Size, int RowNum) {
    double MaxValue = 0; // The value of the pivot element of thåprocess
    int PivotPos; // The Position of the pivot row in the stripe of thåprocess
    struct { double MaxValue; int ProcRank; } ProcPivot, Pivot;
    double *pPivotRow; // Pivot row of the current iteration
    pPivotRow = new double [Size+1]; 
    // The iterations of the Gaussian elimination
    for (int i=0; i<Size; i++) {
        // Calculating the local pivot row
        MaxValue = 0;
        PivotPos = -1;
        for (int j=0; j<RowNum; j++) {
            if ((pProcPivotIter[j] == -1) &&
                (MaxValue < fabs(pProcRows[j*Size+i]))) {
                    MaxValue = fabs(pProcRows[j*Size+i]); 
                    PivotPos = j;
            }
        }
        // Finding the global pivot row
        ProcPivot.MaxValue = MaxValue;
        ProcPivot.ProcRank = ProcRank;
        // Finding the pivot process
        MPI_Allreduce(&ProcPivot, &Pivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
            MPI_COMM_WORLD); 
        // Storing the number of the pivot row
        if ( ProcRank == Pivot.ProcRank ){
            pProcPivotIter[PivotPos]= i;
            pParallelPivotPos[i]= pProcInd[ProcRank] + PivotPos; 
        }
        MPI_Bcast(&pParallelPivotPos[i], 1, MPI_INT, Pivot.ProcRank,
            MPI_COMM_WORLD); 
        // Broadcasting the pivot row
        if ( ProcRank == Pivot.ProcRank ){
            // Fill the pivot row
            for (int j=0; j<Size; j++) {
                pPivotRow[j] = pProcRows[PivotPos*Size + j];
            }
            pPivotRow[Size] = pProcVector[PivotPos];
        }
        MPI_Bcast(pPivotRow, Size+1, MPI_DOUBLE, Pivot.ProcRank,
            MPI_COMM_WORLD);
        //Column elimination
        ParallelEliminateColumns(pProcRows, pProcVector, pPivotRow, Size, RowNum, i); 
    }
    delete [] pPivotRow; 
} 

// Function for the execution of the parallel Gauss algorithm
void ParallelResultCalculation(double* pProcRows, double* pProcVector,double* pProcResult, int Size, int RowNum) {
    // Memory allocation
    pParallelPivotPos = new int [Size];
    pProcPivotIter = new int [RowNum];
    for (int i=0; i<RowNum; i++)
        pProcPivotIter[i] = -1;
    // Gaussian elimination
    ParallelGaussianElimination (pProcRows, pProcVector, Size, RowNum);
    // Back substitution
    //ParallelBackSubstitution (pProcRows, pProcVector, pProcResult, Size, RowNum);
    // Memory deallocation
    delete [] pParallelPivotPos;
    delete [] pProcPivotIter;
} 

// Function for gathering the result vector
void ResultCollection(double* pMatrix, double* pProcRows, int Size, int RowNum) {
    int *pSendNum; // The number of the elements sent to the process
    int *pSendInd; // The index of the first data element sent
    // to the process
    int RestRows=Size; // The number of rows, that have not been
    // distributed yet
    int i; // Loop variable
    // Alloc memory for temporary objects
    pSendInd = new int [ProcNum];
    pSendNum = new int [ProcNum];
    // Define the disposition of the matrix rows for the current process
    RowNum = (Size/ProcNum);
    pSendNum[0] = RowNum*Size;
    pSendInd[0] = 0;
    for (i=1; i<ProcNum; i++) {
        RestRows -= RowNum;
        RowNum = RestRows/(ProcNum-i);
        pSendNum[i] = RowNum*Size;
        pSendInd[i] = pSendInd[i-1]+pSendNum[i-1];
    }
    // Scatter the rows
    MPI_Gatherv(pProcRows,pSendNum[ProcRank],MPI_DOUBLE,pMatrix,
        pSendNum,pSendInd,MPI_DOUBLE,0,MPI_COMM_WORLD);
    // Free the memory
    delete [] pSendNum;
    delete [] pSendInd; 
} 

double DeterminantCalculation(double *pMatrix, int Size){
    double result = 1;
#pragma omp parallel for reduction(*:result)
    for(int i = 0; i<Size; i++){
        for(int j=0;i<Size;j++){
            if (fabs(pMatrix[i*Size+j])>EPS){
                result *= pMatrix[i*Size+j];
                break;
            }
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
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
    if (argc != 2){
        printf("Usage:\n%s \"Size\"\n");
        return 1;
    }
    Size = atoi(argv[1]);
    ProcessInitialization(pMatrix, pVector, pResult, pProcRows, pProcVector, pProcResult, Size, RowNum);

    //Distributing the initial data between the processes
    DataDistribution(pMatrix, pProcRows, pVector, pProcVector, Size, RowNum); 
    //TestDistribution(pMatrix, pVector, pProcRows, pProcVector, Size, RowNum);
    if (ProcRank == 0) {
        printf("Initial matrix \n");
        PrintMatrix(pMatrix, Size, Size);
    }
    double timePar = 0;
    if (ProcRank == 0) timePar = -MPI_Wtime();
    ParallelResultCalculation (pProcRows, pProcVector, pProcResult, Size,RowNum);
    if (ProcRank == 0) timePar += MPI_Wtime();
    //TestDistribution(pMatrix, pVector, pProcRows, pProcVector, Size, RowNum); 
    ResultCollection(pMatrix,pProcRows,Size,RowNum);
    double Det;
    if(ProcRank == 0) Det = DeterminantCalculation(pMatrix,Size);
    if (ProcRank == 0) {
        printf("Result matrix \n");
        PrintMatrix(pMatrix, Size, Size);
        printf("Determinant = %f\n",Det);
        //printf("Initial vector \n");
        //PrintVector(pVector, Size);
        printf("Parallel time: %f\n",timePar);
    } 
    ProcessTermination (pMatrix, pVector, pResult, pProcRows, pProcVector,pProcResult); 
    MPI_Finalize();
    //system("pause");
} 