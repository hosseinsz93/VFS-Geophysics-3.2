#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <math.h>

#include <vector>
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

int NX,NY,NZ;
double pi = 3.14159265358979323846;
double xc_hill=0.0, zc_hill=0.5, r_hill=0.125, h_hill=0.025; 
int N_domn;

int Type=2;
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

    	FILE *fd;
    	char filen[80];  
        sprintf(filen, "terrain.input");
        fd = fopen(filen, "r");
	fscanf(fd, "%le %le %le %le \n", &xc_hill, &zc_hill, &r_hill, &h_hill);
        fclose(fd);


        sprintf(filen, "surface_bot.grd");
        fd = fopen(filen, "r");
	fscanf(fd, "%i \n", &N_domn);
	fscanf(fd, "%i %i %i\n", &NZ, &NX, &NY);

	printf("Nx=%d Ny=%d Nz=%d \n",NX, NY, NZ);
//	double x[NZ][NX], y[NZ][NX], z[NZ][NX];

	std::vector< std::vector<double> > x (NZ);
	std::vector< std::vector<double> > y (NZ);
	std::vector< std::vector<double> > z (NZ);
	for( k=0; k<NZ; k++) x[k].resize(NZ);
	for( k=0; k<NZ; k++) y[k].resize(NZ);
	for( k=0; k<NZ; k++) z[k].resize(NZ);

        printf(" hill center: x=%le z=%le \n", xc_hill, zc_hill);
        printf("r_hill=%le h_hill=%le \n", r_hill, h_hill);

	double buffer;

       	for (k=0; k<NZ; k++)  
       	for (i=0; i<NX; i++) { 
        	fscanf(fd, "%le ", &buffer );
		x[k][i]=buffer;
        }
       
        for (k=0; k<NZ; k++) 
 	for (i=0; i<NX; i++) {
        	fscanf(fd, "%le ", &buffer );
		y[k][i]=buffer;

        }

        for (k=0; k<NZ; k++) 
        for (i=0; i<NX; i++) {
        	fscanf(fd, "%le ", &buffer );
		z[k][i]=buffer;

        }

        fclose(fd);




	
	for (k=0; k<NZ; k++) 
       	for (i=0; i<NX; i++) {

		if (Type==1) {
			double rr=sqrt(pow((x[k][i]-xc_hill),2)+pow((z[k][i]-zc_hill),2));		
			if (rr-r_hill<0.0) y[k][i]=0.5*h_hill+0.5*h_hill*cos(rr*pi/r_hill);
		}else if (Type==2) {
			double rr=sqrt(pow((z[k][i]-zc_hill),2));		
			if (rr-r_hill<0.0) y[k][i]=0.5*h_hill+0.5*h_hill*cos(rr*pi/r_hill);
		}

        }


        sprintf(filen, "surface_terrain.grd");
        fd = fopen(filen, "w");

 	fprintf(fd, "%d \n", 1 );
 	fprintf(fd, "%d %d %d \n", NZ, NX, 1 );
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
