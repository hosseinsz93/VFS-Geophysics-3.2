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

int nx=236370, ny=472736, nz=111;
double pi = 3.14159265358979323846;

int main(int argc, char **argv) 
{
 
	double x[nx], y[ny], z[nz], U[nz], V[nz], W[nz], uu[nz], vv[nz], ww[nz], uv[nz], vw[nz], wu[nz]; 

	FILE *ftec_read;
    	char filen_tecread[80];  

	FILE *fucd;
    	char filen_ucd[80];  

	FILE *ftec_write;
    	char filen_tecwrite[80];  

	char string[128];

        sprintf(filen_tecread, "OSL_tec.dat");
        ftec_read = fopen(filen_tecread, "r");

      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);
      	fgets(string, 128, ftec_read);

        sprintf(filen_tecwrite, "OSL_tec_.dat");
        ftec_write = fopen(filen_tecwrite, "w");

	fprintf(ftec_write, "TITLE     = \"\" \n" );
	fprintf(ftec_write, "VARIABLES = \"x\" \n" );
	fprintf(ftec_write, "\"y\" \n" );
	fprintf(ftec_write, "\"z\" \n" );
	fprintf(ftec_write, "ZONE T=\"'TRIANGLES'\" \n" );
	fprintf(ftec_write, "STRANDID=0, SOLUTIONTIME=0 \n" );
	fprintf(ftec_write, "Nodes=%d, Elements=%d, ZONETYPE=FETriangle \n", NNodes,NElmt );
	fprintf(ftec_write, "DATAPACKING=POINT \n" );
	fprintf(ftec_write, " DT=(SINGLE SINGLE SINGLE ) \n" );

	sprintf(filen_ucd, "OSL_ucd.dat");
        fucd = fopen(filen_ucd, "w");

        fprintf(fucd, "ucd format \n" );
        fprintf(fucd, "from Tecplot \n" );
        fprintf(fucd, "Today \n" );
        fprintf(fucd, "%d %d %d %d %d\n", NNodes,NElmt,0,0,0 );

	int i;

	for(i==0;i<NNodes;i++) {
		double x,y,z;
	      	fscanf(ftec_read, "%le %le %le", &x, &y, &z);
	      	fprintf(ftec_write, "%le %le %le \n ", y, z, x);
	        fprintf(fucd, "%d   %le   %le   %le \n", i+1, y, z, x );
	}

	int j;
	for(j==0;j<NElmt;j++) {
		int nc1,nc2,nc3;
	      	fscanf(ftec_read, "%d %d %d", &nc1, &nc2, &nc3);
	      	fprintf(ftec_write, "%d %d %d \n", nc3, nc2, nc1);
	        fprintf(fucd, "%d   %d  tri  %d   %d   %d \n", j+1, 0, nc3, nc2, nc1 );
	}

	fclose(ftec_read);
	fclose(ftec_write);
	fclose(fucd);

}
