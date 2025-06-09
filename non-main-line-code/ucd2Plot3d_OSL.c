#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <math.h>

  // format of plot3d
  //        1
  //        3        3        1
  //  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00  5.00000000000e-01
  //  5.00000000000e-01  5.00000000000e-01  1.00000000000e+00  1.00000000000e+00
  //  1.00000000000e+00
  //  0.00000000000e+00  5.00000000000e-01  1.00000000000e+00  0.00000000000e+00
  //  5.00000000000e-01  1.00000000000e+00  0.00000000000e+00  5.00000000000e-01
  //  1.00000000000e+00
  //  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00
  //  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00
  //  0.00000000000e+00

// Read triangular surface file of ucd format with elevation
// Read plot3d mesh without elevation
// Write the plot3d mesh with elevation



using namespace std;

struct Cmpnts
{
	double x;
	double y;
	double z;
};


extern bool IsPinTri(Cmpnts *Y, Cmpnts *X1, Cmpnts *X2, Cmpnts *X3);
extern Cmpnts d(Cmpnts *Y, Cmpnts *X1, Cmpnts *X2, Cmpnts *X3);

double pi = 3.14159265358979323846;

int IntpCirc=0;
int IntpTri=1;

double r_circle=0.1;

int main(int argc, char **argv) 
{
  
	int k,j,i;
	int Nx, Ny, Nz;

	int NNodes, NElmts;	
	double *x_ucd, *y_ucd, *z_ucd;
	int *nc1, *nc2, *nc3;

	double **X, **Y, **Z;

	/*
	cout << "Interpolate from a circle ? \n";
	cin >> IntpCirc;	

	cout << "Interpolate from the triangle ? \n";
	cin >> IntpTri;	


	if (IntpCirc) {
	cout << "the circle radius for interpolation \n";
	cin >> r_circle;	
	}
	*/

    	FILE *fd;
	printf("READ OSL plot3d grid with y=0 \n");
    	char filen[80];  
    	sprintf(filen,"OSLGeo_y=0.grd");

    	fd = fopen(filen, "r"); 

	char string[128];

	int tmp;
	fscanf(fd, "%d\n", &tmp);
	fscanf(fd, "%d %d %d \n", &Nx, &Nz, &Ny);

	X=(double**) malloc(sizeof(double)*Nz); 
	Y=(double**) malloc(sizeof(double)*Nz); 
	Z=(double**) malloc(sizeof(double)*Nz); 


	for(k=0;k<Nz;k++) {
		X[k]=(double*) malloc(sizeof(double)*Nx); 
		Y[k]=(double*) malloc(sizeof(double)*Nx); 
		Z[k]=(double*) malloc(sizeof(double)*Nx); 
	}

	printf("Nx=%d, Ny=%d, Nz=%d \n", Nx, Ny, Nz);

       	for (k=0; k<Nz; k++)  
       	for (i=0; i<Nx; i++) { 
        	fscanf(fd, "%le ", &X[k][i] );
        }
 
        	fscanf(fd, "\n ");
       	for (k=0; k<Nz; k++)  
       	for (i=0; i<Nx; i++) { 
        	fscanf(fd, "%le ", &Y[k][i] );
        }
       
        	fscanf(fd, "\n ");
       	for (k=0; k<Nz; k++)  
       	for (i=0; i<Nx; i++) { 
        	fscanf(fd, "%le ", &Z[k][i] );
        }
       
	fclose(fd);


	printf("READ UCD grid with elevation \n");
    	sprintf(filen,"ibmdata00");

    	fd = fopen(filen, "r"); 

	fgets(string,128,fd);
	fgets(string,128,fd);
	fgets(string,128,fd);

	fscanf(fd, "%d %d %d %d %d", &NNodes, &NElmts, &tmp, &tmp, &tmp);

	x_ucd=(double*) malloc(sizeof(double)*NNodes); 
	y_ucd=(double*) malloc(sizeof(double)*NNodes); 
	z_ucd=(double*) malloc(sizeof(double)*NNodes); 

	nc1=(int*) malloc(sizeof(int)*NElmts); 
	nc2=(int*) malloc(sizeof(int)*NElmts); 
	nc3=(int*) malloc(sizeof(int)*NElmts); 


	printf("NNodes=%d, NElmts=%d \n", NNodes, NElmts);
	double tmp1;
	double xc, yc, zc;

       	for (i=0; i<NNodes; i++) { 
        	fscanf(fd, "%d %le %le %le ", &tmp, &x_ucd[i], &y_ucd[i], &z_ucd[i]);
        }


	char ss[20];
       
       	for (i=0; i<NElmts; i++) { 
		fscanf(fd, "%d %d %s %d %d %d ", &tmp,&tmp, ss, &(nc1[i]), &(nc2[i]), &(nc3[i]));
		nc1[i]-=1;
		nc2[i]-=1;
		nc3[i]-=1;
	}

	fclose(fd);

	double *nf_x, *nf_y, *nf_z;
	
	nf_x=(double*) malloc(sizeof(double)*NElmts); 
	nf_y=(double*) malloc(sizeof(double)*NElmts); 
	nf_z=(double*) malloc(sizeof(double)*NElmts); 

       	for (i=0; i<NElmts; i++) { 

      		double dx12 = x_ucd[nc2[i]] - x_ucd[nc1[i]];
      		double dy12 = y_ucd[nc2[i]] - y_ucd[nc1[i]];
      		double dz12 = z_ucd[nc2[i]] - z_ucd[nc1[i]];
      
      		double dx13 = x_ucd[nc3[i]] - x_ucd[nc1[i]];
      		double dy13 = y_ucd[nc3[i]] - y_ucd[nc1[i]];
      		double dz13 = z_ucd[nc3[i]] - z_ucd[nc1[i]];
      
      		nf_x[i] = dy12 * dz13 - dz12 * dy13;
      		nf_y[i] = -dx12 * dz13 + dz12 * dx13;
      		nf_z[i] = dx12 * dy13 - dy12 * dx13;
      
      		double dr = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i] + nf_z[i]*nf_z[i]);
      
      		nf_x[i] /=dr; nf_y[i]/=dr; nf_z[i]/=dr;
 
	}

	printf("Interpolate the  elevation \n");

	if (IntpCirc) {
	for (k=0; k<Nz; k++) 
	for (i=0; i<Nx; i++) {
		double numx=0.0;

		Y[k][i]=0.0;
		int TooSmallCircle=1; 
		
		double YY=0.0;

		for (j=0; j<NNodes; j++) {
			
			if (y_ucd[j]>238.5) {
			double dx=X[k][i]-x_ucd[j];
			double dz=Z[k][i]-z_ucd[j];
		
			double dist=(sqrt(dx*dx+dz*dz)+1.e-9);
	
			if (dist<r_circle) {
				TooSmallCircle=0;
				YY+=y_ucd[j]/dist;
				numx+=1.0/dist;
			}

			}
		}	
		
		if (TooSmallCircle) {
			printf("The radius of circle r=%le for interpolation is too small at k=%d i=%d", r_circle, k,i);
			exit(0);
		}

		YY/=(numx+1.e-9);


		Y[k][i]/=(numx+1.e-9);


	}

	}



	/* the elevation is interpolated from the corresponding triangle */

	if (IntpTri) {

	Cmpnts P, X1, X2, X3, Dis;

       	for (k=0; k<Nz; k++)  
       	for (i=0; i<Nx; i++) { 
		P.x=X[k][i];	
		P.y=0.0;	
		P.z=Z[k][i];	

		Y[k][i]=0.0;

		bool FindTri=false;
		for (j=0; j<NElmts; j++) {

			if (y_ucd[j]>238.5 && nf_y[j]>0.4) {
			
			X1.x=x_ucd[nc1[j]];
			X1.y=0.0;
			X1.z=z_ucd[nc1[j]];

			X2.x=x_ucd[nc2[j]];
			X2.y=0.0;
			X2.z=z_ucd[nc2[j]];

			X3.x=x_ucd[nc3[j]];
			X3.y=0.0;
			X3.z=z_ucd[nc3[j]];
	
			bool InTri=IsPinTri(&P, &X1, &X2, &X3);

			if (InTri) {
				FindTri=true;
				Dis=d(&P, &X1, &X2, &X3);

				double Sum_Dis=Dis.x+Dis.y+Dis.z;
				double fac1=Dis.x/Sum_Dis;
				double fac2=Dis.y/Sum_Dis;
				double fac3=Dis.z/Sum_Dis;

				double YY=fac1*y_ucd[nc1[j]]+fac2*y_ucd[nc2[j]]+fac3*y_ucd[nc3[j]];

				if (YY>Y[k][i]) Y[k][i]=YY;
			}
			}
		}

	}

	}

        sprintf(filen, "OSLBathmPlot3d.grd");
        fd = fopen(filen, "w");

 	fprintf(fd, "%d \n", 1 );
 	fprintf(fd, "%d %d %d \n", Nx, Nz, 1 );
       	for (k=0; k<Nz; k++)  
       	for (i=0; i<Nx; i++) { 
        	fprintf(fd, "%le ", X[k][i] );
        }
        fprintf(fd, "\n");
       
        for (k=0; k<Nz; k++) 
 	for (i=0; i<Nx; i++) {
        	fprintf(fd, "%le ", Y[k][i] );
        }
        fprintf(fd, "\n");

        for (k=0; k<Nz; k++) 
        for (i=0; i<Nx; i++) {
        	fprintf(fd, "%le ", Z[k][i] );
        }

        fclose(fd);

	printf("Finish ~~~~\n");
}


bool IsPinTri(Cmpnts *Y, Cmpnts *X1, Cmpnts *X2, Cmpnts *X3)
{
	Cmpnts nY1, nY2, nY3;
	Cmpnts nX1, nX2, nX3;

	nY1.x=Y->x-X1->x; nY1.y=Y->y-X1->y; nY1.z=Y->z-X1->z; 
	nY2.x=Y->x-X2->x; nY2.y=Y->y-X2->y; nY2.z=Y->z-X2->z; 
	nY3.x=Y->x-X3->x; nY3.y=Y->y-X3->y; nY3.z=Y->z-X3->z; 

	nX1.x=X2->x-X1->x; nX1.y=X2->y-X1->y; nX1.z=X2->z-X1->z; 
	nX2.x=X3->x-X2->x; nX2.y=X3->y-X2->y; nX2.z=X3->z-X2->z; 
	nX3.x=X1->x-X3->x; nX3.y=X1->y-X3->y; nX3.z=X1->z-X3->z; 

	double n1=nY1.x*nX1.x+nY1.y*nX1.y+nY1.z*nX1.z;
	double n2=nY2.x*nX2.x+nY2.y*nX2.y+nY2.z*nX2.z;
	double n3=nY3.x*nX3.x+nY3.y*nX3.y+nY3.z*nX3.z;

	if (n1>0 && n2>0 && n3>0) {
		return true;
	} else return false;
	
}


Cmpnts d(Cmpnts *Y, Cmpnts *X1, Cmpnts *X2, Cmpnts *X3)
{

	Cmpnts nY1, nY2, nY3;
	nY1.x=Y->x-X1->x; nY1.y=Y->y-X1->y; nY1.z=Y->z-X1->z; 
	nY2.x=Y->x-X2->x; nY2.y=Y->y-X2->y; nY2.z=Y->z-X2->z; 
	nY3.x=Y->x-X3->x; nY3.y=Y->y-X3->y; nY3.z=Y->z-X3->z; 

	Cmpnts Distance;

	Distance.x=sqrt(nY1.x*nY1.x+nY1.y*nY1.y+nY1.z*nY1.z)+1.e-9;
	Distance.y=sqrt(nY2.x*nY2.x+nY2.y*nY2.y+nY2.z*nY2.z)+1.e-9;
	Distance.z=sqrt(nY3.x*nY3.x+nY3.y*nY3.y+nY3.z*nY3.z)+1.e-9;

	
}
