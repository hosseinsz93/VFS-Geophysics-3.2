#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <string>
#include <unistd.h>
#include <fftw3.h>
#include <time.h>
#include <vector>

#include "TECIO.h"

using namespace std;

double PI=3.14159265359;

int Nt;
double dt;
int Nx, Ny;
int bc[4];

int save_inflow_period, read_inflow_period, inflow_recycle_perioid;
char path[256]; 
char path_new[256]; 

typedef struct {
        double x, y, z;
} Cmpnts;

int main(int argc, char **argv) 
{
  
	sprintf(path, "./inflow");
	sprintf(path_new, "./inflow_new");
	int k,j,i;
	
	Nt=40000; 
	dt=0.0005;
	Nx=242;
	Ny=202;
	save_inflow_period=500;
	inflow_recycle_perioid=40000;
	bc[0]=0;
	bc[1]=0;
	bc[2]=0;
	bc[3]=1;

	read_inflow_period=save_inflow_period;

	std::vector< std::vector<Cmpnts> > ucat(Ny);
	std::vector< std::vector<Cmpnts> > ucat_new(Ny);
	std::vector< std::vector<Cmpnts> > ucat_avg(Ny);
	for( j=0; j<Ny; j++) ucat[j].resize(Nx);
	for( j=0; j<Ny; j++) ucat_new[j].resize(Nx);
	for( j=0; j<Ny; j++) ucat_avg[j].resize(Nx);
	
	double ucat_xavg[Ny];

	// read the inflow data on grid g1
	
	FILE *f;
	FILE *f_new;
    	char fname[80];  
    	char fname_new[80];  
	int ti;
	int ti2;

	int ti_name;

	double fac;

	for (j=0; j<Ny; j++) 
	for (i=0; i<Nx; i++) {
		ucat_avg[j][i].x=0.0;
		ucat_avg[j][i].y=0.0;
		ucat_avg[j][i].z=0.0;
	}
	
	fac=1.0/(double)Nt;
	for (ti=0;ti<Nt-1;ti++) {

		printf("read inflow at ti=%d\n", ti);
		ti2=ti;
		
		if(ti2>inflow_recycle_perioid) {
			ti2 -= (ti2/inflow_recycle_perioid) * inflow_recycle_perioid;
		}
		ti_name = ( ti2 / read_inflow_period ) * read_inflow_period + 1;

		sprintf(fname, "%s/inflow_%06d_dt=%g.dat", path, ti_name, dt);

		if(ti2==0 || (ti2>90 && ti2%read_inflow_period==1) ) {
			
			if(ti!=0) fclose(f);
			f=fopen(fname, "rb");
			if(!f) printf("\n******************* Cannot open %s ! *******************\n", fname),exit(0);
		}
		for(j=0; j<Ny; j++) fread(&ucat[j][0], sizeof(Cmpnts), Nx, f);
	
		for (j=0; j<Ny; j++) 
		for (i=0; i<Nx; i++) {
			ucat_avg[j][i].x+=ucat[j][i].x*fac;
			ucat_avg[j][i].y+=ucat[j][i].y*fac;
			ucat_avg[j][i].z+=ucat[j][i].z*fac;
		}

	}

	for (j=0; j<Ny; j++) {
		ucat_xavg[j]=0.0;
	}

	fac=1.0/(double)Nx;
	for (j=0; j<Ny; j++) 
	for (i=0; i<Nx; i++) {
		ucat_xavg[j]+=ucat_avg[j][i].z*fac;

	}

	for (ti=0;ti<Nt-1;ti++) {

		printf("read inflow at ti=%d\n", ti);
		ti2=ti;
		
		if(ti2>inflow_recycle_perioid) {
			ti2 -= (ti2/inflow_recycle_perioid) * inflow_recycle_perioid;
		}
		ti_name = ( ti2 / read_inflow_period ) * read_inflow_period + 1;

		sprintf(fname, "%s/inflow_%06d_dt=%g.dat", path, ti_name, dt);

		if(ti2==0 || (ti2>90 && ti2%read_inflow_period==1) ) {
			
			if(ti!=0) fclose(f);
			f=fopen(fname, "rb");
			if(!f) printf("\n******************* Cannot open %s ! *******************\n", fname),exit(0);
		}
		for(j=0; j<Ny; j++) fread(&ucat[j][0], sizeof(Cmpnts), Nx, f);
	

		for (j=0; j<Ny; j++) 
		for (i=0; i<Nx; i++) {
			ucat_new[j][i].x=(ucat[j][i].x-ucat_avg[j][i].x);
			ucat_new[j][i].y=(ucat[j][i].y-ucat_avg[j][i].y);
			ucat_new[j][i].z=ucat_xavg[j]+(ucat[j][i].z-ucat_avg[j][i].z);

		}

		printf("write inflow of new at ti=%d\n", ti);
	      	ti_name = ( (ti-1) / save_inflow_period ) * save_inflow_period + 1;

              	sprintf(fname_new, "%s/inflow_%06d_dt=%g.dat", path_new, ti_name, dt);
	        if(ti==0 || ti%save_inflow_period==1) unlink(fname_new);    // delete existing file

              	f_new=fopen(fname_new, "ab");
       		if(!f_new) printf("\n******************* Cannot open %s ! *******************\n", fname_new),exit(0);

               	for(j=0; j<Ny; j++) fwrite(&ucat_new[j][0], sizeof(Cmpnts), Nx, f_new);
               	fclose(f_new);

	}

	
	for (j=0; j<Ny; j++) 
	for (i=0; i<Nx; i++) {
		ucat_avg[j][i].x=0.0;
		ucat_avg[j][i].y=0.0;
		ucat_avg[j][i].z=0.0;
	}
	

	fac=1.0/(double)Nt;
	for(ti=0; ti<Nt; ti++) {

		printf("read inflow at ti=%d\n", ti);
		ti2=ti;
		
		if(ti2>Nt) {
			ti2 -= (ti2/Nt) * Nt;
		}
		ti_name = ( ti2 / read_inflow_period ) * read_inflow_period + 1;

		sprintf(fname_new, "%s/inflow_%06d_dt=%g.dat", path_new, ti_name, dt);

		if(ti2==0 || (ti2>90 && ti2%read_inflow_period==1) ) {
			
			if(ti!=0) fclose(f_new);
			f_new=fopen(fname_new, "rb");
			if(!f_new) printf("\n******************* Cannot open %s ! *******************\n", fname_new),exit(0);
		}
		for(j=0; j<Ny; j++) fread(&ucat_new[j][0], sizeof(Cmpnts), Nx, f_new);


	
		for (j=0; j<Ny; j++) 
		for (i=0; i<Nx; i++) {
			ucat_avg[j][i].x+=ucat_new[j][i].x*fac;
			ucat_avg[j][i].y+=ucat_new[j][i].y*fac;
			ucat_avg[j][i].z+=ucat_new[j][i].z*fac;
		}


		/*

		if(ti%50==0) {
	      		printf("output tecplot file\n");
    			FILE *f;
	    		char filen_tec[80];
                	sprintf(filen_tec, "%s/inflowcheck_%06d_dt=%g.plt", path_new, ti, dt);
	    		f = fopen(filen_tec, "w");
	    		fprintf(f, "Variables=x,y,z,U,V,W\n");

			fprintf(f, "ZONE T='QUADRILATERAL', N=%d, E=%d, F=FEBLOCK, ET=QUADRILATERAL, VARLOCATION=([1-6]=NODAL)\n", (Nx-1)*(Ny-1), (Nx-2)*(Ny-2));

	                for (j=1;j<Ny;j++)
	                for (i=1;i<Nx;i++) {
      				fprintf(f, "%le\n", (double)i);
			}

	                for (j=1;j<Ny;j++)
	                for (i=1;i<Nx;i++) {
	                        fprintf(f, "%le\n", (double)j);
	                }

        	        for (j=1;j<Ny;j++)
                	for (i=1;i<Nx;i++) {
	                       	fprintf(f, "%le\n", 1.0);
	                }


	                for (j=1;j<Ny;j++)
	                for (i=1;i<Nx;i++) {
	                        fprintf(f, "%le\n", ucat[j][i].x);
	                }

	                for (j=1;j<Ny;j++)
	                for (i=1;i<Nx;i++) {
        	                fprintf(f, "%le\n", ucat[j][i].y);
	                }

	                for (j=1;j<Ny;j++)
	                for (i=1;i<Nx;i++) {
        	                fprintf(f, "%le\n", ucat[j][i].z);
	                }


                	for (j=1;j<Ny-1;j++)
	                for (i=1;i<Nx-1;i++) {
			     	fprintf(f, "%d %d %d %d\n", (j-1)*(Nx-1)+i, (j-1)*(Nx-1)+i+1, (j)*(Nx-1)+i+1, (j)*(Nx-1)+i);
			}
    
		    	fclose(f);
		}
		*/
	}

	      		printf("output tecplot file\n");
	    		char filen_tec[80];
                	sprintf(filen_tec, "%s/uavg.plt", path_new);
	    		f = fopen(filen_tec, "w");
	    		fprintf(f, "Variables=x,y,z,U,V,W\n");

			fprintf(f, "ZONE T='QUADRILATERAL', N=%d, E=%d, F=FEBLOCK, ET=QUADRILATERAL, VARLOCATION=([1-6]=NODAL)\n", (Nx-1)*(Ny-1), (Nx-2)*(Ny-2));

	                for (j=1;j<Ny;j++)
	                for (i=1;i<Nx;i++) {
      				fprintf(f, "%le\n", (double)i);
			}

	                for (j=1;j<Ny;j++)
	                for (i=1;i<Nx;i++) {
	                        fprintf(f, "%le\n", (double)j);
	                }

        	        for (j=1;j<Ny;j++)
                	for (i=1;i<Nx;i++) {
	                       	fprintf(f, "%le\n", 1.0);
	                }


	                for (j=1;j<Ny;j++)
	                for (i=1;i<Nx;i++) {
	                        fprintf(f, "%le\n", ucat_avg[j][i].x);
	                }

	                for (j=1;j<Ny;j++)
	                for (i=1;i<Nx;i++) {
        	                fprintf(f, "%le\n", ucat_avg[j][i].y);
	                }

	                for (j=1;j<Ny;j++)
	                for (i=1;i<Nx;i++) {
        	                fprintf(f, "%le\n", ucat_avg[j][i].z);
	                }


                	for (j=1;j<Ny-1;j++)
	                for (i=1;i<Nx-1;i++) {
			     	fprintf(f, "%d %d %d %d\n", (j-1)*(Nx-1)+i, (j-1)*(Nx-1)+i+1, (j)*(Nx-1)+i+1, (j)*(Nx-1)+i);
			}
    
		    	fclose(f);



}

