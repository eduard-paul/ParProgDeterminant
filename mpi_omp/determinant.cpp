#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <ctime>

const double EPS = 1.0e-10;

int ProcNum = 0; // The number of the available processes
int ProcRank = 0; // The rank of the current process
int* pProcInd;
int* pProcNum; 
int *pParallelPivotPos; // The number of rows selected as the pivot ones
int *pProcPivotIter; // The number of iterations, at which the processor rows were used as the pivot ones 
int* pSerialPivotPos;  // The number of pivot rows selected at the iterations 
int* pSerialPivotIter; // The iterations, at which the rows were pivots 

// Function for random initialization of the matrix and the vector elements
void RandomDataInitialization(double* pMatrix, int Size) {
    int i, j; // Loop variables
    srand(unsigned(time(0)));
    for (i=0; i<Size; i++) {
        for (j=0; j<Size; j++) {
                pMatrix[i*Size+j] = (double)rand()/RAND_MAX/5.5; 
        }
    }
} 

// Finding the pivot row 
int FindPivotRow(double* pMatrix, int Size, int Iter) { 
    int PivotRow = -1; // The index of the pivot row 
    double MaxValue =  0; // The value of the pivot element 
    // Choose the row, that stores the maximum element 
    for (int i=0; i<Size; i++) { 
        if ((pSerialPivotIter[i] == -1) &&  
            (fabs(pMatrix[i*Size+Iter]) > MaxValue)) { 
                PivotRow = i; 
                MaxValue = fabs(pMatrix[i*Size+Iter]); 
        } 
    } 
    return PivotRow; 
} 

void SerialColumnElimination (double* pMatrix, int Pivot,  int Iter, int Size) { 
  double PivotValue, PivotFactor;  
  PivotValue = pMatrix[Pivot*Size+Iter]; 
  for (int i=0; i<Size; i++) { 
    if (pSerialPivotIter[i] == -1) { 
      PivotFactor = pMatrix[i*Size+Iter] / PivotValue; 
      for (int j=Iter; j<Size; j++) { 
        pMatrix[i*Size + j] -= PivotFactor * pMatrix[Pivot*Size+j]; 
      }
    } 
  }   
} 

void SerialGaussianElimination(double* pMatrix,int Size) { 
  int PivotRow;   // The Number of the current pivot row    
      pSerialPivotPos  = new int [Size]; 
    pSerialPivotIter = new int [Size]; 
    for (int i=0; i<Size; i++) { 
        pSerialPivotIter[i] = -1; 
    } 
  for (int Iter=0; Iter<Size; Iter++) { 
    // Finding the pivot row 
    PivotRow = FindPivotRow(pMatrix, Size, Iter); 
    pSerialPivotPos[Iter] = PivotRow; 
    pSerialPivotIter[PivotRow] = Iter; 
    SerialColumnElimination(pMatrix, PivotRow, Iter, Size); 
  } 
  //PrintMatrix(pMatrix, Size, Size); 
      delete [] pSerialPivotPos; 
    delete [] pSerialPivotIter; 
} 

// Function for memory allocation and data initialization
int ProcessInitialization(double* &pMatrix,double* &ParResMatrix, double* &pProcRows, int &Size, int &RowNum){  
    if (ProcRank == 0) {
        if (Size < ProcNum) {
            printf ("Size must be greater than number of processes! \n");
            Size = -1;
            MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            return 1;
        }
    }
    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (Size == -1) return 1;
    int RestRows = Size;
    for (int i=0; i<ProcRank; i++)
        RestRows = RestRows-RestRows/(ProcNum-i);
    RowNum = RestRows/(ProcNum-ProcRank);
    pProcRows = new double [RowNum*Size];
    if (ProcRank == 0) {
        pMatrix = new double [Size*Size];
        ParResMatrix = new double [Size*Size];
        //Initialization of the matrix and the vector elements
        RandomDataInitialization (pMatrix, Size); 
    } 
    return 0;
} 

// Function for computational process termination
void ProcessTermination(double* pMatrix, double*pProcRows)
{
    if (ProcRank == 0) {
        delete [] pMatrix;
    }
    delete [] pProcRows;
} 

// Data distribution among the processes
void DataDistribution(double* pMatrix, double* pProcRows, int Size, int RowNum) {
    int *pSendNum = new int [ProcNum];
    int *pSendInd = new int [ProcNum];
    int RestRows=Size;
    RowNum = (Size/ProcNum);
    pSendNum[0] = RowNum*Size;
    pSendInd[0] = 0;
    for (int i=1; i<ProcNum; i++) {
        RestRows -= RowNum;
        RowNum = RestRows/(ProcNum-i);
        pSendNum[i] = RowNum*Size;
        pSendInd[i] = pSendInd[i-1]+pSendNum[i-1];
    }

    MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcRows,
        pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
} 

void PrintMatrix(double *pMatrix, int Size1, int Size2){
    std::cout.precision(4);
    for (int i=0; i<Size1; i++) {
        for (int j=0; j<Size2; j++) {
            std::cout<<pMatrix[i*Size2+j]<<"\t"; 
        }
        std::cout<<std::endl;
    }
}

// Fuction for column elimination
void ParallelEliminateColumns(double* pProcRows, double* pPivotRow, int Size, int RowNum, int Iter) {
#pragma omp parallel for
    for (int i=0; i<RowNum; i++) {
        if (pProcPivotIter[i] == -1) {
            double PivotFactor = pProcRows[i*Size+Iter] / pPivotRow[Iter];
            for (int j=Iter; j<Size; j++) {
                pProcRows[i*Size + j] -= PivotFactor* pPivotRow[j];
            }
        }
    }
} 

// Gaussian elimination
void ParallelGaussianElimination(double* pProcRows, int Size, int RowNum) {
    double MaxValue = 0; // The value of the pivot element of thå process
    int PivotPos; // The Position of the pivot row in the stripe of thåprocess
    struct { double MaxValue; int ProcRank; } ProcPivot, Pivot;
    double *pPivotRow; // Pivot row of the current iteration
    pPivotRow = new double [Size]; 
    pParallelPivotPos = new int [Size];
    pProcPivotIter = new int [RowNum];
    for (int i=0; i<RowNum; i++)
        pProcPivotIter[i] = -1;
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
        }
        MPI_Bcast(pPivotRow, Size+1, MPI_DOUBLE, Pivot.ProcRank,
            MPI_COMM_WORLD);
        //Column elimination
        ParallelEliminateColumns(pProcRows, pPivotRow, Size, RowNum, i); 
    }
    delete [] pPivotRow; 
    delete [] pParallelPivotPos;
    delete [] pProcPivotIter;
} 

// Function for gathering the result matrix
void ResultCollection(double* pMatrix, double* pProcRows, int Size, int RowNum) {
    int *pSendNum = new int [ProcNum];
    int *pSendInd = new int [ProcNum];
    int RestRows=Size; 
    RowNum = (Size/ProcNum);
    pSendNum[0] = RowNum*Size;
    pSendInd[0] = 0;
    for (int i=1; i<ProcNum; i++) {
        RestRows -= RowNum;
        RowNum = RestRows/(ProcNum-i);
        pSendNum[i] = RowNum*Size;
        pSendInd[i] = pSendInd[i-1]+pSendNum[i-1];
    }

    MPI_Gatherv(pProcRows,pSendNum[ProcRank],MPI_DOUBLE,pMatrix,
        pSendNum,pSendInd,MPI_DOUBLE,0,MPI_COMM_WORLD);

    delete [] pSendNum;
    delete [] pSendInd; 
} 

void QuickSort(int* a, int N, int &revers) {

    long i = 0, j = N; 
    int p = a[ N>>1 ];

    do {
        while ( a[i] < p ) i++;
        while ( a[j] > p ) j--;

        if (i <= j) {
            int temp = a[i]; a[i] = a[j]; a[j] = temp;    
            if (i != j) revers *= -1;
            i++; j--;            
        }
    } while ( i<=j );

    if ( j > 0 ) QuickSort(a, j, revers);
    if ( N > i ) QuickSort(a+i, N-i, revers);
}

double DeterminantCalculation(double *pMatrix, int Size){
    double result = 1;
    int *arr = new int[Size];
    for(int i = 0; i<Size; i++){
        for(int j=0;i<Size;j++){
            if (fabs(pMatrix[i*Size+j])>EPS){
                result *= pMatrix[i*Size+j];
                arr[i] = j;
                break;
            }
        }
    }
    int revers = 1;
    QuickSort(arr,Size-1,revers);
    result *= revers;
    delete[] arr;
    return result;
}

double ParallelDeterminantCalculation(double *pMatrix, int Size){
    double result = 1;
    int *arr = new int[Size];
#pragma omp parallel for reduction(*:result)
    for(int i = 0; i<Size; i++){
        for(int j=0;i<Size;j++){
            if (fabs(pMatrix[i*Size+j])>EPS){
                result *= pMatrix[i*Size+j];
                arr[i] = j;
                break;
            }
        }
    }
    int revers = 1;
    QuickSort(arr,Size-1,revers);
    result *= revers;
    delete[] arr;
    return result;
}

int main(int argc, char* argv[]) {
    double* Matrix, *ParResMatrix;
    double *ProcRows;  // The Rows of matrix on the process
    int Size;           // The Size of the initial matrix
    int RowNum;         // The Number of the matrix rows on the current process
    double timePar, timeSer;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (argc != 2){
        if (ProcRank == 0) 
            printf("Usage:\n\t%s \"Size\"\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    Size = atoi(argv[1]);

    if(ProcessInitialization(Matrix, ParResMatrix, ProcRows, Size, RowNum) == EXIT_FAILURE){
        MPI_Finalize();
        return 1;
    }

    if (ProcRank == 0) timePar = -MPI_Wtime();

    DataDistribution(Matrix, ProcRows, Size, RowNum);  
    ParallelGaussianElimination(ProcRows,Size,RowNum);    
    ResultCollection(ParResMatrix,ProcRows,Size,RowNum);

    if(ProcRank == 0) {
        double parDet = ParallelDeterminantCalculation(ParResMatrix,Size);
        timePar += MPI_Wtime();

        std::cout<<"Parallel time: "<<timePar<<std::endl;

        timeSer = -MPI_Wtime();
        SerialGaussianElimination(Matrix,Size);

        double serDet = DeterminantCalculation(Matrix,Size);
        timeSer += MPI_Wtime();
        
        std::cout<<"Serial time: "<<timeSer<<std::endl;

        if (fabs(serDet-parDet)>fabs(serDet*EPS)){
            std::cout<<"Wrong result!";
        } else{
            std::cout<<"Success!";
        }
    } 
    ProcessTermination (Matrix, ProcRows); 
    MPI_Finalize();
    //system("pause");
} 