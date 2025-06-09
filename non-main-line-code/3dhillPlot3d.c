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


using namespace std;
// kevin's exp 
//

int NX = 301, NZ=601;
double pi = 3.14159265358979323846;
double x1=-0.15, x2=0.15;
double z1=0.3, z2=1.5;
double xc_hill=0.0, zc_hill=0.5, r_hill=0.125, h_hill=0.083333; 
int ReadP=0;

/*
int NX = 101, NZ=101;
double pi = 3.14159265358979323846;
double x1=-1, x2=1;
double z1=-1, z2=1;
double xc_hill=0.0, zc_hill=0.0, r_hill=0.5, h_hill=0.25; 
*/


int main(int argc, char **argv) 
{
  
	int k,j,i;

//	cout << "Read Parameters? 0--no, 1--yes \n";
//	printf("%d \n", ReadP);

		cout << "NX \n";
		printf("%d \n", NX);

		cout << "NZ \n";
		printf("%d \n",NZ);

	        cout << "center of hill \n";
	        printf("x=%le z=%le \n", xc_hill, zc_hill);

	        cout << "radius and height of hill \n";
	        printf("r_hill=%le h_hill=%le \n", r_hill, h_hill);


	        cout << "Domain size \n";
	        printf("x1=%le x2=%le z1=%le z2=%le \n", x1, x2, z1, z2);


	double x[NZ][NX], y[NZ][NX], z[NZ][NX];

	double dx = (x2-x1)/((double)NX-1.0); 
	double dz = (z2-z1)/((double)NZ-1.0); 


	for (k=0; k<NZ; k++) 
       	for (i=0; i<NX; i++) {
        	x[k][i]=x1+dx*((double)i);
        }

	for (k=0; k<NZ; k++) 
       	for (i=0; i<NX; i++) {
        	z[k][i]=z1+dz*((double)k);
        }

	for (k=0; k<NZ; k++) 
       	for (i=0; i<NX; i++) {

		double rr=sqrt(pow((x[k][i]-xc_hill),2)+pow((z[k][i]-zc_hill),2));
		
		if (rr-r_hill>1.e-11) y[k][i]=0.0;
		else y[k][i]=0.5*h_hill+0.5*h_hill*cos(rr*pi/r_hill);
        }


    	FILE *fd;
    	char filen[80];  
        sprintf(filen, "3dhill.grd");
        fd = fopen(filen, "w");

 	fprintf(fd, "%d \n", 1 );
 	fprintf(fd, "%d %d %d \n", NX, NZ, 1 );
       	for (k=0; k<NZ; k++)  
       	for (i=0; i<NX; i++) { 
        	fprintf(fd, "%le ", x[k][i] );
        }
        fprintf(fd, "\n");
       
        for (k=0; k<NZ; k++) 
 	for (i=0; i<NX; i++) {

        	fprintf(fd, "%le ", y[k][i] );
        }
        fprintf(fd, "\n");

        for (k=0; k<NZ; k++) 
        for (i=0; i<NX; i++) {

        	fprintf(fd, "%le ", z[k][i] );
        }

        fclose(fd);

}
