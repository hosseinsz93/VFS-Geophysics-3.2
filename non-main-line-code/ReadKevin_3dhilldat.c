// Read the experimental data A matrix

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <string>
#include <unistd.h>
#include <fftw3.h>
#include <time.h>

#include "petscvec.h"
//#include "petscda.h"
#include "petscksp.h"
#include "petscsnes.h"

#ifdef TECIO
#include "TECIO.h"
#endif

using namespace std;

/* 
  format of the matrix
  
  Nx Ny
  Y[ny-1] V[....] V[....] . . . V[....] V[....]
  Y[Ny-2] V[....] V[....] . . . V[....] V[....]
     .       .                             .
     .       .                             .
     .       .                             .
  Y[   1] V[....] V[....] . . . V[....] V[....]
  Y[   0] V[....] V[....] . . . V[....] V[....]
          X[   0] X[   1] . . . X[Nx-2] X[Nx-1]

*/

char path[256]; 

double **u;
double *X, *Y, *Z;
int Nx, Ny, Nz;

int main(int argc, char **argv) 
{

	Ny=1;
  
	sprintf(path, "./");
	int k,j,i;

	char filen1[80], filen[80];
	
	cout << "The name of the file without .txt \n";
	cin >> filen1;

	sprintf(filen, "%s.txt", filen1);

    	FILE *fd;
 	char string[128]; 
        fd = fopen(filen, "r");
//	fgets(string, 128, fd);	
        fscanf(fd, "%d %d", &Nx, &Nz);
        printf("Nx=%d Nz=%d \n", Nx, Nz);
//	fgets(string, 128, fd);	

//	Nz=1;
	// allocate memory
	u = (double**) malloc(sizeof(double)*Nz);
	X = (double*) malloc(sizeof(double)*Nx);
	Z = (double*) malloc(sizeof(double)*Nz);

	for(k=0;k<Nz;k++) {
		u[k] = (double*) malloc(sizeof(double)*Nx);
	}

       	for (k=0; k<Nz; k++) { 
        	fscanf(fd, "%le", &Z[k]);
//        	printf("Z=%le \n", Z[k]);
        	for(i=0;i<Nx; i++) {
			fscanf(fd, "%le", &u[k][i]);
//        	 	printf("u=%le \n", u[k][i]);
		}
        }
        for(i=0;i<Nx;i++) { 
		fscanf(fd, "%le", &X[i]);
//        	printf("X=%le \n", X[i]);
	}
        fclose(fd);

	
	sprintf(filen, "%s.plt", filen1);

	INTEGER4 I, Debug, VIsDouble, DIsDouble, III, IMax, JMax, KMax;
	VIsDouble = 0;
	DIsDouble = 0;
	Debug = 0;

	double SolTime = 0.0;
	INTEGER4 StrandID = 0; 	
	INTEGER4 ParentZn = 0;	
	INTEGER4 TotalNumFaceNodes = 1; 
	INTEGER4 NumConnectedBoundaryFaces = 1; 
	INTEGER4 TotalNumBoundaryConnections = 1; 
	INTEGER4 FileType = 0;


	
	I = TECINI112((char*)"Flow", (char*)"X Y Z val", filen, (char*)".", &FileType, &Debug, &VIsDouble);

	INTEGER4	ZoneType=0, ICellMax=0, JCellMax=0, KCellMax=0;
	INTEGER4	IsBlock=1, NumFaceConnections=0, FaceNeighborMode=0;
	INTEGER4    ShareConnectivityFromZone=0;
	INTEGER4	LOC[40] = {1, 1, 1, 1}; 
		

	I = TECZNE112((char*)"Zone 1",
                        &ZoneType,      
                        &Nx,
                        &Ny,
                        &Nz,
                        &ICellMax,
                        &JCellMax,
                        &KCellMax,
                        &SolTime,
                        &StrandID,
			&ParentZn,
			&IsBlock,
			&NumFaceConnections,
			&FaceNeighborMode,
			&TotalNumFaceNodes,
			&NumConnectedBoundaryFaces,
			&TotalNumBoundaryConnections,
			NULL,
			LOC,
			NULL,
			&ShareConnectivityFromZone);

	III = Nx*Ny*Nz;
	
	float *x;
	x = new float [III];
		
	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = X[i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = 0.0;
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = Z[k];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);

	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = u[k][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);

	I = TECEND112();


	
}

