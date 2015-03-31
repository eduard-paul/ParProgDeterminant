#include <stdio.h>
#include <iostream>
#include <ctime>
#include <mpi.h>
#include <omp.h>

using namespace std;

void main(int argc, char** argv)
{
    MPI_Init(&argc,&argv);
    srand(time(0));
    int n = 10;
    int needPrint = 0;
    cin>>n>>needPrint;
    double **a = new double*[n];
    double **b = new double*[n];
    for (int i=0; i<n; i++){
        a[i] = new double[n];
        b[i] = new double[n];
    }

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            a[i][j]=-30+rand()%61;
            b[i][j] = a[i][j];
        }
    }
    if(needPrint)
        for (int i=0; i<n; i++)
        {
            for (int j=0; j<n; j++)
                printf("%f   ",a[i][j]);
            printf("\n");
        }

        double timeLin = -MPI_Wtime();
        //Прямой ход-исключение переменных
        for(int k=0;k<n;k++)//цикл по матрицам
        {
            for(int m=k+1;m<n;m++)
            {
                double left = a[m][k];
                for(int j=k;j<n;j++)
                    a[m][j]=a[m][j]-left*a[k][j]/a[k][k];
            }
        }
        timeLin += MPI_Wtime();

        double timePar = -MPI_Wtime();
        //Прямой ход-исключение переменных
#pragma omp parallel
        for(int k=0;k<n;k++)//цикл по матрицам
        {
            #pragma omp for
            for(int m=k+1;m<n;m++)
            {
                double left = a[m][k];
                for(int j=k;j<n;j++)
                    b[m][j]=b[m][j]-left*b[k][j]/b[k][k];
            }
        }
        timePar += MPI_Wtime();



        printf("%f  %f\n",timeLin,timePar);
        if(needPrint)
            for (int i=0; i<n; i++)
            {
                for (int j=0; j<n; j++)
                    printf("%f   ",a[i][j]);
                printf("\n");
            }
            MPI_Finalize();
            system("pause");
}

/*
printf("\n Korni:\n\n");

printf("\n\n");
for (i=0; i<n; i++)
{
for (j=0; j<n; j++)
printf("%f   ",a[i][j]);
printf("%f   \n",b[i]);
}
*/