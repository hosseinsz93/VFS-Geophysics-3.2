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

int imax = 9841571;
double pi = 3.14159265358979323846;
int NX=201, NY=201;
//double xr_0=614836.8, yr_0=253241; // rotating orientation, small domain
double xr_0=609548, yr_0=243028.1; // rotating orientation, large domain
double angle_r=0.0; //-atan(6.0/8.0); //rotating angle
double x1=0.0, x2=26393;
double ybg=0.0, y2=26383; //Domain size
int main(int argc, char **argv) 
{
  
	int k,j,i;

	cout << "NO of xyz points \n";
	printf("%d \n", imax);

	cout << "NX \n";
	printf("%d \n", NX);

	cout << "NY \n";
	printf("%d \n",NY);

        cout << "Rotation angle \n";
        printf("%le \n", angle_r);

        cout << "Rotation center \n";
        printf("x=%le y=%le \n", xr_0, yr_0);

        cout << "Domain size \n";
        printf("x1=%le x2=%le y1=%le y2=%le \n", x1, x2, ybg, y2);

	

	double xb[imax], yb[imax], zb[imax];
	double x[NY][NX], y[NY][NX], z[NY][NX];
	
	char ctmp;

    	FILE *fd;
	printf("READ xyz data\n");
    	char filen[80];  
    	sprintf(filen,"Contours_2ft_April_2012.xyz");


    	fd = fopen(filen, "r"); 

      	for (i=0; i<imax; i++) {
		fscanf(fd, "%le %le %le \n", &xb[i],  &yb[i],  &zb[i]);
	}
	fclose(fd);

	double xmax=-1000000.0, ymax=-1000000.0, zmax=-1000000.0, xmin=1000000.0, ymin=10000000.0, zmin=1000000.0;

	for (i=0; i<imax; i++) {
		xmax = max(xb[i], xmax);
		xmin = min(xb[i], xmin);

		ymax = max(yb[i], ymax);
		ymin = min(yb[i], ymin);

		zmax = max(zb[i], zmax);
		zmin = min(zb[i], zmin);


	}	
		
	printf("xmin= %le ,xmax=%le ,ymin= %le,  ymax=%le,  zmin= %le,  zmax=%le\n ", xmin, xmax, ymin, ymax, zmin, zmax);

	double dx = (x2-x1)/((double)NX-1.0); 
	double dy = (y2-ybg)/((double)NY-1.0); 


	for (j=0; j<NY; j++) 
       	for (i=0; i<NX; i++) {
        	x[j][i]=x1+dx*((double)i);
        }

	for (j=0; j<NY; j++) 
       	for (i=0; i<NX; i++) {
        	y[j][i]=ybg+dy*((double)j);
        }

	for (j=0; j<NY; j++) 
       	for (i=0; i<NX; i++) {
        	z[j][i]=0.0;
        }

	double x_1[NY][NX], y_1[NY][NX];


        for (j=0; j<NY; j++)
        for (i=0; i<NX; i++) {
                x_1[j][i]=xr_0+x[j][i]*cos(angle_r)-y[j][i]*sin(angle_r);
                y_1[j][i]=yr_0+x[j][i]*sin(angle_r)+y[j][i]*cos(angle_r);
        }


	double numx=0.0;
	double dist_2ij;
	double dist_min = 1.0e9;
	int imin_dist;
	for (j=0; j<NY; j++) 
       	for (i=0; i<NX; i++) {
		numx=0.0;
		dist_min = 1.0e9;
		printf("i=%d, j=%d \n", i, j);
		for (k=0; k<imax; k++) {
			dist_2ij = sqrt(pow((xb[k]-x_1[j][i]),2) + pow((yb[k]-y_1[j][i]),2));
			if (dist_2ij<=dist_min) {
				imin_dist = k;
				dist_min = dist_2ij;
			}
        		if (xb[k] >= x_1[j][i]-dx && xb[k] <= x_1[j][i]+dx && yb[k] >= y_1[j][i]-dy && yb[k] <= y_1[j][i]+dy) {
				double dist = pow((xb[k]-x_1[j][i]),2) + pow((yb[k]-y_1[j][i]),2);
				dist = 1.0/(sqrt(dist)+1.0e-9);
				z[j][i]+=dist*zb[k];
				numx+=dist;
			}
		}

		z[j][i] /= (numx+1.0e-9);
		if (numx < 1.0e-9) z[j][i]=zb[imin_dist];
        }

        sprintf(filen, "surface.grd");
        fd = fopen(filen, "w");

 	fprintf(fd, "%d \n", 1 );
 	fprintf(fd, "%d %d %d \n", NX, NY, 1 );
       	for (j=0; j<NY; j++)  
       	for (i=0; i<NX; i++) { 
        	fprintf(fd, "%le ", x[j][i] );
        }
        fprintf(fd, "\n");
       
        for (j=0; j<NY; j++) 
 	for (i=0; i<NX; i++) {

        	fprintf(fd, "%le ", y[j][i] );
        }
        fprintf(fd, "\n");

        for (j=0; j<NY; j++) 
        for (i=0; i<NX; i++) {

        	fprintf(fd, "%le ", z[j][i] );
        }

        fclose(fd);

}
