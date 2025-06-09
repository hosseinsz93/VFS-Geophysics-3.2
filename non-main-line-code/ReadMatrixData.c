#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <math.h>

using namespace std;

int NXi=100, NY=100;

int main(int argc, char **argv) 
{  
	int i,j,k;


	double **U, **u, **uw;

	U=(double**)malloc(sizeof(double*)*NY);
	u=(double**)malloc(sizeof(double*)*NY);
	uw=(double**)malloc(sizeof(double*)*NY);
		
	for(j=0; j<NY; j++) {
		U[j]=(double*)malloc(sizeof(double)*NX);
		u[j]=(double*)malloc(sizeof(double)*NX);
		uw[j]=(double*)malloc(sizeof(double)*NX);
	}
	

	char filen[80];


	sprintf(filen, "%sHill_segment1%06d.plt", prefix, ti);

	INTEGER4 I, Debug, VIsDouble, DIsDouble, III, IMax, JMax, KMax;
	VIsDouble = 0;
	DIsDouble = 0;
	Debug = 0;

	double SolTime = ((double)ti) * dt;
	INTEGER4 StrandID = 0; /* StaticZone */	
	INTEGER4 ParentZn = 0;	
	INTEGER4 TotalNumFaceNodes = 1; /* Not used for FEQuad zones*/
	INTEGER4 NumConnectedBoundaryFaces = 1; /* Not used for FEQuad zones*/
	INTEGER4 TotalNumBoundaryConnections = 1; /* Not used for FEQuad zones*/
	INTEGER4 FileType = 0;

	I = TECINI112((char*)"Flow", (char*)"X Y Z U <greek>s</greek><sub>u</sub>, uw", filen, (char*)".", &FileType, &Debug, &VIsDouble);

	INTEGER4	ZoneType=0, ICellMax=0, JCellMax=0, KCellMax=0;
	INTEGER4	IsBlock=1, NumFaceConnections=0, FaceNeighborMode=0;
	INTEGER4    	ShareConnectivityFromZone=0;
	INTEGER4	LOC[40] = {1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0}; /* 1 is cell-centered 0 is node centered */
		

	I = TECZNE112((char*)"Zone 1",
                        &ZoneType,      /* Ordered zone */
                        &IMax,
                        &JMax,
                        &KMax,
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

	III = IMax*JMax*KMax;
		
	float *x;
	x = new float [III];
		
	for (k=zs; k<ze-1; k++)
	for (j=ys; j<ye-1; j++)
	for (i=xs; i<xe-1; i++) {
		x[(k-zs) * IMax*JMax + (j-ys)*IMax + (i-xs)] = coor[k][j][i].x;
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);

	I = TECEND112();
	return 0;

}
