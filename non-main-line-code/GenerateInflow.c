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
#include "petscvec.h"
//#include "petscda.h"
#include "petscksp.h"
#include "petscsnes.h"


#ifdef TECIO
#include "TECIO.h"
#endif

using namespace std;


/* 
  format of plot3d
        1
        3        3        1
  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00  5.00000000000e-01
  5.00000000000e-01  5.00000000000e-01  1.00000000000e+00  1.00000000000e+00
  1.00000000000e+00
  0.00000000000e+00  5.00000000000e-01  1.00000000000e+00  0.00000000000e+00
  5.00000000000e-01  1.00000000000e+00  0.00000000000e+00  5.00000000000e-01
  1.00000000000e+00
  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00
  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00  0.00000000000e+00
  0.00000000000e+00
*/

int Nx=64, Ny=64, Nz=64; // The number of grid for synthetic turbulence

double Lx=5.0, Ly=2.0, Lz=1.0;

double Gamma=0.0; //3.9;//3.9;

double ustar=0.18;

double L_coef=0.59;

double D_coef=3.2;

double Y_loc=0.5; // Y_loc=0.1*Lz;

double Karman_const=0.4;

double Beta;


double Z_intp=0.0; // 

double PI=3.14159265359;

int Nt_LES=10000; //28800; //28800;
double dt_LES =0.001; //0.25; // seconds
int Nx_LES=128, Ny_LES=92;
//int Nx_LES=122, Ny_LES=122;

int save_inflow_period=1000;
int read_inflow_period=1000;

int inflow_recycle_perioid=10000;
double L_ref=1.0;
double V_ref=1.0;
double Tmprt_ref=291.0;

int Turb=1;

char path[256]; 

double Uz_Convective;

typedef struct {
        double x, y, z;
} Cmpnts;

extern double randn_notrig(); 

double ***U, ***V, ***W;
double *X, *Y, *Z;
int **i_intp, **j_intp, **k_intp;

extern void GenerateTurb();

int InletProfile=1;

double u_inlet=0.0;
double w_inlet=1.0;

int levelset=1;
double y_water=0.5;
int Temperature=0;

double z0=0.00003;

double H=1000.0;

int Ny_AvgIn=25;

int Geninflow=1;
int main(int argc, char **argv) 
{
  
	sprintf(path, "./inflow");
	int t,k,j,i;

	FILE *fd;
	char filen[80];  
       	sprintf(filen, "xyz.dat");
       	fd = fopen(filen, "r");
       	printf("Read xyz file\n");
	int ttt;
        fscanf(fd, "%d %d %d \n", &Nx_LES, &Ny_LES, &ttt);
	Nx_LES+=1;
	Ny_LES+=1;



	// allocate memory
	//
	std::vector< std::vector<double> > X_LES (Ny_LES);
	std::vector< std::vector<double> > Y_LES (Ny_LES);
	std::vector< std::vector<double> > Z_LES (Ny_LES);
	std::vector< std::vector<double> > Level (Ny_LES);
	std::vector< std::vector<double> > Nvert (Ny_LES);
	std::vector< std::vector<Cmpnts> > ucat (Ny_LES);
	std::vector< std::vector<double> > Tmprt (Ny_LES);
	for( j=0; j<Ny_LES; j++) X_LES[j].resize(Nx_LES);
	for( j=0; j<Ny_LES; j++) Y_LES[j].resize(Nx_LES);
	for( j=0; j<Ny_LES; j++) Z_LES[j].resize(Nx_LES);
	for( j=0; j<Ny_LES; j++) Level[j].resize(Nx_LES);
	for( j=0; j<Ny_LES; j++) Nvert[j].resize(Nx_LES);
	for( j=0; j<Ny_LES; j++) ucat[j].resize(Nx_LES);
	for( j=0; j<Ny_LES; j++) Tmprt[j].resize(Nx_LES);
	

	std::vector< std::vector<int> > i_intp (Ny_LES);
	std::vector< std::vector<int> > j_intp (Ny_LES);
	for( j=0; j<Ny_LES; j++) i_intp[j].resize(Nx_LES);
	for( j=0; j<Ny_LES; j++) j_intp[j].resize(Nx_LES);
	

        for(j=0;j<Ny_LES;j++)
        for(i=0;i<Nx_LES;i++) {
		ucat[j][i].x=0.0;
		ucat[j][i].y=0.0;
		ucat[j][i].z=0.0;
		Level[j][i]=0.0;
		Nvert[j][i]=0.0;
		X_LES[j][i]=0.0;
		Y_LES[j][i]=0.0;
		Z_LES[j][i]=0.0;
		Tmprt[j][i]=0.0;
	}




	double tmp, x,y,z;
	for(i=0;i<Nx_LES-1;i++) {
	        fscanf(fd, "%le %le %le \n", &x, &tmp, &tmp);
		for (j=0;j<Ny_LES-1;j++) {
			X_LES[j][i]=x;
		} 
	}

	for(j=0;j<Ny_LES-1;j++) {
	        fscanf(fd, "%le %le %le \n", &tmp, &y, &tmp);
		for (i=0;i<Nx_LES-1;i++) {
			Y_LES[j][i]=y;
		} 
		//printf("Y1=%le \n", y);
	}

	double XX=X_LES[0][0];
	double YY=Y_LES[0][0];
	for(i=0;i<Nx_LES-1;i++) {
	for (j=0;j<Ny_LES-1;j++) {
			X_LES[j][i]-=XX;
			Y_LES[j][i]-=YY;
	} 
	}


	for(j=0;j<Ny_LES-1;j++) {
	for (i=0;i<Nx_LES-1;i++) {
		Z_LES[j][i]=0.0;
	} 
	}


        fclose(fd);


	printf("Y1=%le, Y2=%le \n", Y_LES[Ny_LES-3][0], Y_LES[0][0]);

	if (Geninflow) {

	double *W_AvgIn;
	double *T_AvgIn;
	double *Y_AvgIn;
	W_AvgIn = (double*) malloc(sizeof(double)*Ny_AvgIn);
	T_AvgIn = (double*) malloc(sizeof(double)*Ny_AvgIn);
	Y_AvgIn = (double*) malloc(sizeof(double)*Ny_AvgIn);

	if (InletProfile==3) {
		FILE *fd;
	    	char filen[80];  
        	sprintf(filen, "Wavg_In.dat");
        	fd = fopen(filen, "r");
        	printf("Read Mean Inflow file\n");
		for (j=0;j<Ny_AvgIn;j++) {
        	 	fscanf(fd, "%le %le \n", &Y_AvgIn[j], &W_AvgIn[j]);
		}

        	fclose(fd);

		if (Temperature) {
        	sprintf(filen, "Tavg_In.dat");
        	fd = fopen(filen, "r");
        	printf("Read Mean Inflow file Temperature\n");
		for (j=0;j<Ny_AvgIn;j++) {
        	 	fscanf(fd, "%le %le ", &Y_AvgIn[j], &T_AvgIn[j]);
		}

        	fclose(fd);
		}

	}

	

	double T_ref=L_ref/V_ref;
	double dt_LES_scaled=dt_LES/T_ref;

	// Read LES Grid
	//
	/*
    	FILE *fd;
    	char filen[80];  
 	char string[128]; 
        sprintf(filen, "InflowPlane.dat");
        fd = fopen(filen, "r");
	fgets(string, 128, fd);	
	fgets(string, 128, fd);	
	fgets(string, 128, fd);	
	fgets(string, 128, fd);	
	fgets(string, 128, fd);	
	fgets(string, 128, fd);	
	fgets(string, 128, fd);	
	fgets(string, 128, fd);	
	fgets(string, 128, fd);	
	fgets(string, 128, fd);	
	if (levelset) fgets(string, 128, fd);	

	printf("read LES grid, levelset and immersed body information\n");

	double tmp;
       	for (j=0; j<Ny_LES-1; j++) {
       	for (i=0; i<Nx_LES-1; i++) { 
       	//for (j=0; j<Ny_LES-1; j++) {
        	if (levelset) fscanf(fd, "%le %le %le %le %le ", &X_LES[j][i], &Y_LES[j][i], &Z_LES[j][i], &Nvert[j][i], &Level[j][i]);
        	if (!levelset) fscanf(fd, "%le %le %le %le ", &X_LES[j][i], &Y_LES[j][i], &Z_LES[j][i], &Nvert[j][i]);
        }
	}
        fclose(fd);
	*/
	
//       	for (j=2; j<Ny_LES; j++) { 
//       		for (i=1; i<Nx_LES; i++) { 
//			Y_LES[j][i]=Y_LES[j][i]-Y_LES[1][i];
//	        }
//       		for (i=1; i<Nx_LES; i++) { 
//			Y_LES[1][i]=0.0;
//		}
//	}
	

	

	// Generating Inflow Turbulence
	printf("generating synthetic turbulence\n");

	Ly=fabs(X_LES[0][Nx_LES-2]-X_LES[0][0]);
	Lz=fabs(Y_LES[Ny_LES-2][0]-Y_LES[0][0]);
	Lx=4.0*Ly;

	printf("Y1=%le, Y2=%le \n", Y_LES[Ny_LES-2][0], Y_LES[0][0]);
	printf("Lx=%le, Ly=%le, Lz=%le\n", Lx, Ly, Lz);

	Ny=Nx_LES-1;
	Nz=Ny_LES-1;
	Nx=4*Ny;

	if (Nx%2!=0) Nx+=1;
	if (Ny%2!=0) Ny+=1;
	if (Nz%2!=0) Nz+=1;

	Y_loc=0.1*Lz/L_ref;

	printf("%d %d %d\n", Nx, Ny, Nz);
	printf("%le %le %le\n", Lx, Ly, Lz);
	if (Turb) {
		GenerateTurb();

	//for(k=0;k<Nz;k++) {
	//	printf("k=%d, Z=%le \n", k, Z[k]);
	//}

	//exit(0);
		printf("search interpolate infos\n");
	       	for (j=0; j<Ny_LES-1; j++)  
	       	for (i=0; i<Nx_LES-1; i++) { 
			int ii, jj, kk;
			for (jj=1;jj<Ny;jj++) {
				if (X_LES[j][i]-Y[jj-1]>=-1e-9 && X_LES[j][i]-Y[jj]<1.e-9) i_intp[j][i]=jj-1;
			}

			for (kk=1;kk<Nz;kk++) {
		//		if (j==0) printf("kk=%d Y_LES=%le Z1=%le Z2=%le \n", kk, Y_LES[j][i], Z[kk-1], Z[kk]);
				if (Y_LES[j][i]-Z[kk-1]>=-1e-9 && Y_LES[j][i]-Z[kk]<1.e-9) j_intp[j][i]=kk-1;
			}

			//printf("j_intp(0,%d)_tt=%d  \n", i, j_intp[0][i] );
		}

//	exit(0);

	//		printf("j_intp(0,0)=%d  \n", j_intp[0][0] );
		printf("end search interpolate infos\n");
	}

	int t_LES, t_WRF, tWRF_intp;
	
	for(t_LES=0;t_LES<Nt_LES;t_LES++) {

		printf("t_LES %d \n", t_LES);

                for(j=0;j<Ny_LES-1;j++)
                for(i=0;i<Nx_LES-1;i++) {
			if (InletProfile==1) {
				ucat[j][i].x=u_inlet;
				ucat[j][i].y=0.0;
				ucat[j][i].z=w_inlet;
			} 
			else if (InletProfile==2) {
				ucat[j][i].x=0.0;
				ucat[j][i].y=0.0;
				if (Y_LES[j][i]<H) ucat[j][i].z=ustar*log(Y_LES[j][i]/z0)/0.4;
				else ucat[j][i].z=0.0;
			}	
			else if (InletProfile==3) {
				ucat[j][i].x=0.0;
				ucat[j][i].y=0.0;
				int jj;
				for(jj=0;jj<Ny_AvgIn-1; jj++) {
					if (Y_LES[j][i]>=Y_AvgIn[jj] && Y_LES[j][i]<=Y_AvgIn[jj+1]) {
						double fac1=(Y_AvgIn[jj+1]-Y_LES[j][i])/(Y_AvgIn[jj+1]-Y_AvgIn[jj]);
						double fac2=(-Y_AvgIn[jj]+Y_LES[j][i])/(Y_AvgIn[jj+1]-Y_AvgIn[jj]);
						ucat[j][i].z=W_AvgIn[jj]*fac1+W_AvgIn[jj+1]*fac2;

						if (Temperature) Tmprt[j][i]=T_AvgIn[jj]*fac1+T_AvgIn[jj+1]*fac2;
					}
				}
			}

		}

		/*
                for(j=1;j<Ny_LES-1;j++)
                for(i=1;i<Nx_LES-1;i++) {
			if (Nvert[j][i]>0.1 || Level[j][i]<0) {
				ucat[j][i].z=0.0;
				ucat[j][i].x=0.0;
				ucat[j][i].y=0.0;
			}
		}
		*/


		if (Turb) {
//			Uz_Convective=w_inlet;

			//if (InletProfile==2) {	
			double fac_Uz=1.0/((Nx_LES-1)*(Ny_LES-1));
			Uz_Convective=0.0;
			for(j=0;j<Ny_LES-1;j++)
			for(i=0;i<Nx_LES-1;i++) {
				Uz_Convective+=ucat[j][i].z*fac_Uz/V_ref;
			}

			//}
			//else if (InletProfile==1) Uz_Convective=sqrt(w_inlet*w_inlet+u_inlet*u_inlet);
			

			Z_intp+=Uz_Convective*dt_LES_scaled;
			
			double XX_intp=Z_intp;				
			if ((Z_intp-X[Nx-1])>1.e-9) {
				int kkk=Z_intp/X[Nx-1];
				XX_intp=Z_intp-kkk*X[Nx-1];	
			}	

			int II_intp;
			for(i=1;i<Nx;i++) {
				if ((XX_intp-X[i-1])>=-1.e-20 && (XX_intp-X[i])<=1.e-20) II_intp=i;
			}

			printf("****** U_Convec=%le  \n", Uz_Convective);
			printf("****** XX_intp=%le  \n", XX_intp);
			printf("****** X0=%le  \n", X[0]);
			printf("****** X1=%le  \n", X[Nx-1]);

        	        for(j=0;j<Ny_LES-1;j++)
	                for(i=0;i<Nx_LES-1;i++) {
				double fac_11=(XX_intp-X[II_intp-1])/(X[II_intp]-X[II_intp-1]);
				double fac_22=(-XX_intp+X[II_intp])/(X[II_intp]-X[II_intp-1]);

				int jj=j_intp[j][i];
				int ii=i_intp[j][i];


			//printf("j_intp(0,%d)_3=%d  \n", i, j_intp[0][i] );
			//printf("j_intp(0,0)_3=%d  \n", j_intp[0][0] );
			//	printf("i=%d, j=%d \n", i, j);			
			//	printf("ii=%d, jj=%d \n", ii, jj);			
		
				double fac1=(X_LES[j][i]-Y[ii])/(Y[ii+1]-Y[ii]);
				double fac2=(-X_LES[j][i]+Y[ii+1])/(Y[ii+1]-Y[ii]);
				double fac3=(Y_LES[j][i]-Z[jj])/(Z[jj+1]-Z[jj]);
				double fac4=(-Y_LES[j][i]+Z[jj+1])/(Z[jj+1]-Z[jj]);
			
			//	printf("X_syn1=%le, X_LES=%le X_syn2=%le\n", Y[ii+1], X_LES[j][i], Y[ii] );			
			//	printf("Y_syn1=%le, Y_LES=%le Y_syn2=%le\n", Z[jj+1], Y_LES[j][i], Z[jj] );			

				double uu1=fac4*(fac2*V[jj][ii][II_intp-1]+fac1*V[jj][ii+1][II_intp-1])+fac3*(fac2*V[jj+1][ii][II_intp-1]+fac1*V[jj+1][ii+1][II_intp-1]);
				double uu2=fac4*(fac2*V[jj][ii][II_intp]+fac1*V[jj][ii+1][II_intp])+fac3*(fac2*V[jj+1][ii][II_intp]+fac1*V[jj+1][ii+1][II_intp]);
	
				double vv1=fac4*(fac2*W[jj][ii][II_intp-1]+fac1*W[jj][ii+1][II_intp-1])+fac3*(fac2*W[jj+1][ii][II_intp-1]+fac1*W[jj+1][ii+1][II_intp-1]);
				double vv2=fac4*(fac2*W[jj][ii][II_intp]+fac1*W[jj][ii+1][II_intp])+fac3*(fac2*W[jj+1][ii][II_intp]+fac1*W[jj+1][ii+1][II_intp]);
	
				double ww1=fac4*(fac2*U[jj][ii][II_intp-1]+fac1*U[jj][ii+1][II_intp-1])+fac3*(fac2*U[jj+1][ii][II_intp-1]+fac1*U[jj+1][ii+1][II_intp-1]);
				double ww2=fac4*(fac2*U[jj][ii][II_intp]+fac1*U[jj][ii+1][II_intp])+fac3*(fac2*U[jj+1][ii][II_intp]+fac1*U[jj+1][ii+1][II_intp]);

			//	printf("fac_11=%le, fac_22=%le \n", fac_11, fac_22);			
			//	printf("uu1=%le, uu2=%le \n", uu1, uu2);			
			//	printf("vv1=%le, vv2=%le \n", vv1, vv2);			
			//	printf("ww1=%le, ww2=%le \n", ww1, ww2);			

				if (!levelset || (levelset && Y_LES[j][i]<y_water) ) {
					ucat[j][i].x+=fac_22*uu1+fac_11*uu2;
					ucat[j][i].y+=fac_22*vv1+fac_11*vv2;
					ucat[j][i].z+=fac_22*ww1+fac_11*ww2;
				}
			}

		}

                for(j=0;j<Ny_LES-1;j++)
                for(i=0;i<Nx_LES-1;i++) {
			if (Nvert[j][i]>1.1) {
				ucat[j][i].z=0.0;
				ucat[j][i].x=0.0;
				ucat[j][i].y=0.0;
			}
		}


                for(j=0;j<Ny_LES-1;j++)
                for(i=0;i<Nx_LES-1;i++) {
//			if (levelset && Level[j][i]<0) {
//				ucat[j][i].z=0.0;
//				ucat[j][i].x=0.0;
//				ucat[j][i].y=0.0;
//			}
		}


		/*
                for(j=0;j<Ny_LES;j++)
                for(i=0;i<Nx_LES;i++) {
			if (j==0) {
				ucat[j][i].z=0.0;
				ucat[j][i].x=0.0;
				ucat[j][i].y=0.0;
				Tmprt[j][i]=Tmprt[j+1][i];
			}
			if (j==Ny_LES-1) {
				ucat[j][i].z=ucat[j-1][i].z;
				ucat[j][i].x=ucat[j-1][i].x;
				ucat[j][i].y=0.0;
				Tmprt[j][i]=Tmprt[j-1][i];
			}

			if (i==0) {
				ucat[j][0].z=ucat[j][Nx_LES-2].z;
				ucat[j][0].x=ucat[j][Nx_LES-2].x;
				ucat[j][0].y=ucat[j][Nx_LES-2].y;
				Tmprt[j][i]=Tmprt[j][i+1];
			}
			if (i==Nx_LES-1) {
				ucat[j][Nx_LES-1].z=ucat[j][1].z;
				ucat[j][Nx_LES-1].x=ucat[j][1].x;
				ucat[j][Nx_LES-1].y=ucat[j][1].y;
				Tmprt[j][i]=Tmprt[j][i-1];
			}
		}
		*/


		printf("Write LES Inflow \n");
                char fname_LES[256];
                int ti_name = ( (t_LES-1) / save_inflow_period ) * save_inflow_period + 1;

                sprintf(fname_LES, "%s/inflow_%06d_dt=%g.dat", path, ti_name, dt_LES_scaled);
                if(t_LES==0 || t_LES%save_inflow_period==1) unlink(fname_LES);    // delete existing file

                FILE *fp_LES=fopen(fname_LES, "ab");
                if(!fp_LES) printf("\n******************* Cannot open %s ! *******************\n", fname_LES),exit(0);

                for(j=0; j<Ny_LES; j++) fwrite(&ucat[j][0], sizeof(Cmpnts), Nx_LES, fp_LES);
                fclose(fp_LES);


		if (Temperature) {
		printf("Write Temperature Inflow \n");
                char fname_tp[256];
               	ti_name = ( (t_LES-1) / save_inflow_period ) * save_inflow_period + 1;

                sprintf(fname_tp, "%s/inflow_t_%06d_dt=%g.dat", path, ti_name, dt_LES_scaled);

	        if(t_LES==0 || t_LES%save_inflow_period==1) unlink(fname_tp);    // delete existing file

        	FILE *fp_tp=fopen(fname_tp, "ab");
	        if(!fp_tp) printf("\n******************* Cannot open %s ! *******************\n", fname_tp),exit(0);

      	        for(j=0; j<Ny_LES; j++) fwrite(&Tmprt[j][0], sizeof(double), Nx_LES, fp_tp);
                fclose(fp_tp);
		}
	

		if(t_LES%50==0) {
	      		printf("output tecplot file\n");
    			FILE *f;
	    		char filen_tec[80];
                	sprintf(filen_tec, "%s/inflowcheck_%06d_dt=%g.plt", path, t_LES, dt_LES_scaled);
//	    		sprintf(filen_tec, "InFlowField%06d.dat",t_LES);
	    		f = fopen(filen_tec, "w");
	    		fprintf(f, "Variables=x,y,z,U,V,W,T\n");

			fprintf(f, "ZONE T='QUADRILATERAL', N=%d, E=%d, F=FEBLOCK, ET=QUADRILATERAL, VARLOCATION=([1-7]=NODAL)\n", (Nx_LES-1)*(Ny_LES-1), (Nx_LES-2)*(Ny_LES-2));

	                for (j=0;j<Ny_LES-1;j++)
	                for (i=0;i<Nx_LES-1;i++) {
      				fprintf(f, "%le\n", X_LES[j][i]);
			}

	                for (j=0;j<Ny_LES-1;j++)
	                for (i=0;i<Nx_LES-1;i++) {
	                        fprintf(f, "%le\n", Y_LES[j][i]);
	                }

        	        for (j=0;j<Ny_LES-1;j++)
                	for (i=0;i<Nx_LES-1;i++) {
	                       	fprintf(f, "%le\n", Z_LES[j][i]);
	                }


	                for (j=0;j<Ny_LES-1;j++)
	                for (i=0;i<Nx_LES-1;i++) {
	                        fprintf(f, "%le\n", ucat[j][i].x);
	                }

	                for (j=0;j<Ny_LES-1;j++)
	                for (i=0;i<Nx_LES-1;i++) {
        	                fprintf(f, "%le\n", ucat[j][i].y);
	                }

	                for (j=0;j<Ny_LES-1;j++)
	                for (i=0;i<Nx_LES-1;i++) {
        	                fprintf(f, "%le\n", ucat[j][i].z);
	                }

	                for (j=0;j<Ny_LES-1;j++)
	                for (i=0;i<Nx_LES-1;i++) {
        	                fprintf(f, "%le\n", Tmprt[j][i]);
	                }



                	for (j=1;j<Ny_LES-1;j++)
	                for (i=1;i<Nx_LES-1;i++) {
			     	fprintf(f, "%d %d %d %d\n", (j-1)*(Nx_LES-1)+i, (j-1)*(Nx_LES-1)+i+1, (j)*(Nx_LES-1)+i+1, (j)*(Nx_LES-1)+i);
			}
    
		    	fclose(f);
	
		}


	}

	}

//  	Average 

	std::vector< std::vector<Cmpnts> > U_g1 (Ny_LES);
	for( j=0; j<Ny_LES; j++) U_g1[j].resize(Nx_LES);
	
	std::vector< std::vector<double> > uu_g1 (Ny_LES);
	for( j=0; j<Ny_LES; j++) uu_g1[j].resize(Nx_LES);
						
	std::vector< std::vector<double> > vv_g1 (Ny_LES);
	for( j=0; j<Ny_LES; j++) vv_g1[j].resize(Nx_LES);

	std::vector< std::vector<double> > ww_g1 (Ny_LES);
	for( j=0; j<Ny_LES; j++) ww_g1[j].resize(Nx_LES);

	std::vector< std::vector<double> > uv_g1 (Ny_LES);
	for( j=0; j<Ny_LES; j++) uv_g1[j].resize(Nx_LES);

	std::vector< std::vector<double> > vw_g1 (Ny_LES);
	for( j=0; j<Ny_LES; j++) vw_g1[j].resize(Nx_LES);

	std::vector< std::vector<double> > wu_g1 (Ny_LES);
	for( j=0; j<Ny_LES; j++) wu_g1[j].resize(Nx_LES);


       	for (j=0; j<Ny_LES; j++)  
	for (i=0; i<Nx_LES; i++) { 
 		U_g1[j][i].x=0.0;
 		U_g1[j][i].y=0.0;
 		U_g1[j][i].z=0.0;
		uu_g1[j][i]=0.0;
		vv_g1[j][i]=0.0;
		ww_g1[j][i]=0.0;
		uv_g1[j][i]=0.0;
		vw_g1[j][i]=0.0;
		wu_g1[j][i]=0.0;
	}

	int ti, ti2; 
    	char fname_g1[80];  
	FILE *f_g1;
	for (ti=0;ti<Nt_LES;ti++) {

		printf("Average inflow at ti=%d\n", ti);
		ti2=ti;
		
		if(ti2>inflow_recycle_perioid) {
			ti2 -= (ti2/inflow_recycle_perioid) * inflow_recycle_perioid;
		}
		int ti_name = ( ti2 / read_inflow_period ) * read_inflow_period + 1;

		sprintf(fname_g1, "%s/inflow_%06d_dt=%g.dat", path, ti_name, dt_LES);

		if(ti2==0 || (ti2>90 && ti2%read_inflow_period==1) ) {
		
			if(ti!=0) fclose(f_g1);
			f_g1=fopen(fname_g1, "rb");
			if(!f_g1) printf("\n******************* Cannot open %s ! *******************\n", fname_g1),exit(0);
		}
		for(j=0; j<Ny_LES; j++) fread(&ucat[j][0], sizeof(Cmpnts), Nx_LES, f_g1);

       		for (j=1; j<Ny_LES; j++)  
	       	for (i=1; i<Nx_LES; i++) { 
 			U_g1[j][i].x+=ucat[j][i].x;
 			U_g1[j][i].y+=ucat[j][i].y;
 			U_g1[j][i].z+=ucat[j][i].z;
			uu_g1[j][i]+=ucat[j][i].x*ucat[j][i].x;
			vv_g1[j][i]+=ucat[j][i].y*ucat[j][i].y;
			ww_g1[j][i]+=ucat[j][i].z*ucat[j][i].z;
			uv_g1[j][i]+=ucat[j][i].x*ucat[j][i].y;
			vw_g1[j][i]+=ucat[j][i].y*ucat[j][i].z;
			wu_g1[j][i]+=ucat[j][i].z*ucat[j][i].x;
		}

	}


       	for (j=1; j<Ny_LES; j++)  
	for (i=1; i<Nx_LES; i++) { 
 		U_g1[j][i].x/=(double)Nt_LES;
 		U_g1[j][i].y/=(double)Nt_LES;
 		U_g1[j][i].z/=(double)Nt_LES;
		uu_g1[j][i]/=(double)Nt_LES;
		uu_g1[j][i]-=U_g1[j][i].x*U_g1[j][i].x;

		vv_g1[j][i]/=(double)Nt_LES;
		vv_g1[j][i]-=U_g1[j][i].y*U_g1[j][i].y;

		ww_g1[j][i]/=(double)Nt_LES;
		ww_g1[j][i]-=U_g1[j][i].z*U_g1[j][i].z;

		uv_g1[j][i]/=(double)Nt_LES;
		uv_g1[j][i]-=U_g1[j][i].x*U_g1[j][i].y;

		vw_g1[j][i]/=(double)Nt_LES;
		vw_g1[j][i]-=U_g1[j][i].y*U_g1[j][i].z;

		wu_g1[j][i]/=(double)Nt_LES;
		wu_g1[j][i]-=U_g1[j][i].z*U_g1[j][i].x;
	}

	FILE *f_avg;
 	char filen_avg[80];

	printf("output tecplot file\n");
        sprintf(filen_avg, "%s/inflow_avg.plt",path);

	f_avg = fopen(filen_avg, "w");
	fprintf(f_avg, "Variables=x,y,z,U,V,W,uu,vv,ww,uv,vw,wu\n");

	fprintf(f_avg, "ZONE T='QUADRILATERAL', N=%d, E=%d, F=FEBLOCK, ET=QUADRILATERAL, VARLOCATION=([1-12]=NODAL)\n", (Nx_LES-1)*(Ny_LES-1), (Nx_LES-2)*(Ny_LES-2));

	for (j=0;j<Ny_LES-1;j++)
	for (i=0;i<Nx_LES-1;i++) {
      		fprintf(f_avg, "%le\n", X_LES[j][i]);
	}
	for (j=0;j<Ny_LES-1;j++)
	for (i=0;i<Nx_LES-1;i++) {
      		fprintf(f_avg, "%le\n", Y_LES[j][i]);
	}

	for (j=0;j<Ny_LES-1;j++)
	for (i=0;i<Nx_LES-1;i++) {
      		fprintf(f_avg, "%le\n", Z_LES[j][i]);
	}


	for (j=0;j<Ny_LES-1;j++)
	for (i=0;i<Nx_LES-1;i++) {
      		fprintf(f_avg, "%le\n", U_g1[j][i].x);
	}

	for (j=0;j<Ny_LES-1;j++)
	for (i=0;i<Nx_LES-1;i++) {
      		fprintf(f_avg, "%le\n", U_g1[j][i].y);
	}


	for (j=0;j<Ny_LES-1;j++)
	for (i=0;i<Nx_LES-1;i++) {
      		fprintf(f_avg, "%le\n", U_g1[j][i].z);
	}

	for (j=0;j<Ny_LES-1;j++)
	for (i=0;i<Nx_LES-1;i++) {
      		fprintf(f_avg, "%le\n", uu_g1[j][i]);
	}

	for (j=0;j<Ny_LES-1;j++)
	for (i=0;i<Nx_LES-1;i++) {
      		fprintf(f_avg, "%le\n", vv_g1[j][i]);
	}


	for (j=0;j<Ny_LES-1;j++)
	for (i=0;i<Nx_LES-1;i++) {
      		fprintf(f_avg, "%le\n", ww_g1[j][i]);
	}

	for (j=0;j<Ny_LES-1;j++)
	for (i=0;i<Nx_LES-1;i++) {
      		fprintf(f_avg, "%le\n", uv_g1[j][i]);
	}

	for (j=0;j<Ny_LES-1;j++)
	for (i=0;i<Nx_LES-1;i++) {
      		fprintf(f_avg, "%le\n", vw_g1[j][i]);
	}

	for (j=0;j<Ny_LES-1;j++)
	for (i=0;i<Nx_LES-1;i++) {
      		fprintf(f_avg, "%le\n", wu_g1[j][i]);
	}


        for (j=1;j<Ny_LES-1;j++)
	for (i=1;i<Nx_LES-1;i++) {
	     	fprintf(f_avg, "%d %d %d %d\n", (j-1)*(Nx_LES-1)+i, (j-1)*(Nx_LES-1)+i+1, (j)*(Nx_LES-1)+i+1, (j)*(Nx_LES-1)+i);
	}
    
    	fclose(f_avg);
	


	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		free(U[j][i]);
		free(V[j][i]);
		free(W[j][i]);
	}

	free(X);
	free(Y);
	free(Z);

}









void GenerateTurb__( ) 
{
  
	int t,k,j,i;

	double TC11, TC12, TC13, TC21, TC22, TC23, TC31, TC32, TC33;

	double *Kx, *Ky, *Kz;
//	double *X, *Y, *Z;

	double C11, C12, C13, C21, C22, C23, C31, C32, C33;
//	double ***U, ***V, ***W;
	double ***U_i, ***V_i, ***W_i;
	double ***n1_Gauss, ***n2_Gauss, ***n3_Gauss; 
	double ***Divg_k;
	double ***Divg;
	

	double ***dUdx, ***dVdy, ***dWdz;
	double ***dUdx_i, ***dVdy_i, ***dWdz_i;

/*
	Ly=fabs(X_LES[1][1][Nx_LES-1]-X_LES[1][1][1]);
	Lz=fabs(Y_LES[1][Ny_LES-1][1]-Y_LES[1][1][1]);
//	Lx=fabs(Z_LES[Nz_LES-1][1][1]-Z_LES[1][1][1]);
	Lx=4.0*Ly;

//	Nx=Nz_LES-2;
	Ny=Nx_LES-2;
	Nz=Ny_LES-2;
	Nx=4*Ny;

	Y_loc=0.1*Lz;
*/

	Kx = (double*) malloc(sizeof(double)*Nx);
	Ky = (double*) malloc(sizeof(double)*Ny);
	Kz = (double*) malloc(sizeof(double)*Nz);

	X = (double*) malloc(sizeof(double)*Nx);
	Y = (double*) malloc(sizeof(double)*Ny);
	Z = (double*) malloc(sizeof(double)*Nz);

	U = (double***) malloc(sizeof(double)*Nz);
	V = (double***) malloc(sizeof(double)*Nz);
	W = (double***) malloc(sizeof(double)*Nz);

	U_i = (double***) malloc(sizeof(double)*Nz);
	V_i = (double***) malloc(sizeof(double)*Nz);
	W_i = (double***) malloc(sizeof(double)*Nz);

	n1_Gauss = (double***) malloc(sizeof(double)*Nz);
	n2_Gauss = (double***) malloc(sizeof(double)*Nz);
	n3_Gauss = (double***) malloc(sizeof(double)*Nz);

	Divg_k = (double***) malloc(sizeof(double)*Nz);
	Divg = (double***) malloc(sizeof(double)*Nz);


	dUdx = (double***) malloc(sizeof(double)*Nz);
	dVdy = (double***) malloc(sizeof(double)*Nz);
	dWdz = (double***) malloc(sizeof(double)*Nz);

	dUdx_i = (double***) malloc(sizeof(double)*Nz);
	dVdy_i = (double***) malloc(sizeof(double)*Nz);
	dWdz_i = (double***) malloc(sizeof(double)*Nz);


	for(k=0;k<Nz;k++) {
		U[k] = (double**) malloc(sizeof(double)*Ny);
		V[k] = (double**) malloc(sizeof(double)*Ny);
		W[k] = (double**) malloc(sizeof(double)*Ny);

		U_i[k] = (double**) malloc(sizeof(double)*Ny);
		V_i[k] = (double**) malloc(sizeof(double)*Ny);
		W_i[k] = (double**) malloc(sizeof(double)*Ny);

		n1_Gauss[k] = (double**) malloc(sizeof(double)*Ny);
		n2_Gauss[k] = (double**) malloc(sizeof(double)*Ny);
		n3_Gauss[k] = (double**) malloc(sizeof(double)*Ny);
		Divg_k[k] = (double**) malloc(sizeof(double)*Ny);
		Divg[k] = (double**) malloc(sizeof(double)*Ny);


		dUdx[k] = (double**) malloc(sizeof(double)*Ny);
		dVdy[k] = (double**) malloc(sizeof(double)*Ny);
		dWdz[k] = (double**) malloc(sizeof(double)*Ny);

		dUdx_i[k] = (double**) malloc(sizeof(double)*Ny);
		dVdy_i[k] = (double**) malloc(sizeof(double)*Ny);
		dWdz_i[k] = (double**) malloc(sizeof(double)*Ny);

	}


	for(k=0;k<Nz;k++) 
	for(j=0;j<Ny;j++) {
		U[k][j] = (double*) malloc(sizeof(double)*Nx);
		V[k][j] = (double*) malloc(sizeof(double)*Nx);
		W[k][j] = (double*) malloc(sizeof(double)*Nx);

		U_i[k][j] = (double*) malloc(sizeof(double)*Nx);
		V_i[k][j] = (double*) malloc(sizeof(double)*Nx);
		W_i[k][j] = (double*) malloc(sizeof(double)*Nx);

		n1_Gauss[k][j] = (double*) malloc(sizeof(double)*Nx);
		n2_Gauss[k][j] = (double*) malloc(sizeof(double)*Nx);
		n3_Gauss[k][j] = (double*) malloc(sizeof(double)*Nx);
		Divg_k[k][j] = (double*) malloc(sizeof(double)*Nx);
		Divg[k][j] = (double*) malloc(sizeof(double)*Nx);


		dUdx[k][j] = (double*) malloc(sizeof(double)*Nx);
		dVdy[k][j] = (double*) malloc(sizeof(double)*Nx);
		dWdz[k][j] = (double*) malloc(sizeof(double)*Nx);

		dUdx_i[k][j] = (double*) malloc(sizeof(double)*Nx);
		dVdy_i[k][j] = (double*) malloc(sizeof(double)*Nx);
		dWdz_i[k][j] = (double*) malloc(sizeof(double)*Nx);


	}




	for(i=0;i<Nx/2+1;i++) {
		Kx[i]=i*2.0*PI/Lx;
	}

	for(j=0;j<Ny/2+1;j++) {
		Ky[j]=j*2.0*PI/Ly;
	}

	for(k=0;k<Nz/2+1;k++) {
		Kz[k]=k*2.0*PI/Lz;
	}

	for(i=Nx/2+1;i<Nx;i++) {
		Kx[i]=(i-Nx)*2.0*PI/Lx;
	}

	for(j=Ny/2+1;j<Ny;j++) {
		Ky[j]=(j-Ny)*2.0*PI/Ly;
	}

	for(k=Nz/2+1;k<Nz;k++) {
		Kz[k]=(k-Nz)*2.0*PI/Lz;
	}


//	Kx[Nx/2]=0.0;
//	Ky[Ny/2]=0.0;
//	Kz[Nz/2]=0.0;


	double dy=Ly/(Ny-1);

	for(j=0;j<Ny;j++) {
		Y[j]=double(j)*dy;
	}

	double dx=Lx/(Nx-1);

	for(i=0;i<Nx;i++) {
		X[i]=double(i)*dx;
	}

	double dz=Lz/(Nz-1);

	for(k=0;k<Nz;k++) {
		Z[k]=double(k)*dz;
	}


        FILE *f_tmp;
        char fname_tmp[80];  
        sprintf(fname_tmp, "E_k.dat");
        f_tmp = fopen(fname_tmp, "w");
	j=0; k=0;
	for (i=0;i<Nx;i++) {

		double K = sqrt(Kx[i]*Kx[i]+Ky[j]*Ky[j]+Kz[k]*Kz[k]);
		double fac=1.0/(sqrt(4.0*PI)*K*K+1.e-9);
		double fac1 = D_coef*ustar*ustar/pow(Y_loc,2.0/3.0);
		double L = L_coef*Y_loc;
		double fac2 = pow(L,5.0/3.0);
		double fac3 = pow(L,4)*pow(K,4)/pow((1+pow(L,2)*pow(K,2)),17.0/6.0);

		double E=  fac1 * fac2 * fac3;   // alpha * e^2/3 *  L^5/3 *  L^4 k^4 / (1+L^2K^2)^(17/6) 
		
		double FF=E;
        	fprintf( f_tmp, "%le %le \n", K, FF );
	}


	fclose(f_tmp);

	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		n1_Gauss[k][j][i] = randn_notrig();
		n2_Gauss[k][j][i] = randn_notrig();
		n3_Gauss[k][j][i] = randn_notrig();
		Divg_k[k][j][i] = 0.0;
	}
	

	fftw_complex *CKu, *CKv, *CKw;
	fftw_complex *CKu_o, *CKv_o, *CKw_o;

	CKu = (fftw_complex*) fftw_malloc(Nx*Ny*Nz*sizeof(fftw_complex));
	CKu_o = (fftw_complex*) fftw_malloc(Nx*Ny*Nz*sizeof(fftw_complex));
	CKv = (fftw_complex*) fftw_malloc(Nx*Ny*Nz*sizeof(fftw_complex));
	CKv_o = (fftw_complex*) fftw_malloc(Nx*Ny*Nz*sizeof(fftw_complex));
	CKw = (fftw_complex*) fftw_malloc(Nx*Ny*Nz*sizeof(fftw_complex));
	CKw_o = (fftw_complex*) fftw_malloc(Nx*Ny*Nz*sizeof(fftw_complex));


	fftw_plan pu, pv, pw;

	pu=fftw_plan_dft_3d(Nz, Ny, Nx, CKu, CKu_o, FFTW_BACKWARD, FFTW_ESTIMATE);
	pv=fftw_plan_dft_3d(Nz, Ny, Nx, CKv, CKv_o, FFTW_BACKWARD, FFTW_ESTIMATE);
	pw=fftw_plan_dft_3d(Nz, Ny, Nx, CKw, CKw_o, FFTW_BACKWARD, FFTW_ESTIMATE);


	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {

			double K = sqrt(Kx[i]*Kx[i]+Ky[j]*Ky[j]+Kz[k]*Kz[k]);
			double fac=1.0/(sqrt(4.0*PI)*K*K);
			double fac1 = D_coef*ustar*ustar/pow(Y_loc,2.0/3.0);
			double L = L_coef*Y_loc;
			double fac2 = pow(L,5.0/3.0);
			double fac3 = pow(L,4)*pow(K,4)/pow((1+pow(L,2)*pow(K,2)),17.0/6.0);

			double E=  fac1 * fac2 * fac3;   // alpha * e^2/3 *  L^5/3 *  L^4 k^4 / (1+L^2K^2)^(17/6) 

			double dkx=2*PI/Lx, dky=2*PI/Ly, dkz=2*PI/Lz;
		
			double fac4=sqrt(dkx*dky*dkz);

			double facc = fac4*sqrt(E)*fac;

			if (fabs(K)<1.e-20) facc=0.0;

			C11=0.0; C22=0.0; C33=0.0; 

			C12=facc * Kz[k] ; C13=-facc * Ky[j] ; 
			C21=-facc * Kz[k] ; C23=facc * Kx[i] ; 
			C31=facc * Ky[j] ; C32=-facc * Kx[i] ; 

			Beta=Gamma*(ustar/Karman_const/Y_loc)*pow((K*L+1.e-9),-2.0/3.0);

			double K30=Kz[k];
			double K3=K30-Beta*Kx[i];

			double K0=K;
			K = sqrt(Kx[i]*Kx[i]+Ky[j]*Ky[j]+K3*K3);

			double C1=Beta*pow(Kx[i],2)*(pow(K0,2)-2*pow(K30,2)+Beta*Kx[i]*K30)/(pow(K,2)*(pow(Kx[i],2)+pow(Ky[j],2))+1.e-9);

			double tmp=atan(Beta*Kx[i]*sqrt(pow(Kx[i],2)+pow(Ky[j],2))/(pow(K0,2)-K30*Kx[i]*Beta+1.e-9));
			double C2=Ky[j]*pow(K0,2)*tmp/(pow(pow(Kx[i],2)+pow(Ky[j],2),3.0/2.0)+1.e-9);

			double zeta1=C1-Ky[j]*C2/(Kx[i]+1.e-9);

			double zeta2=Ky[j]*C1/(Kx[i]+1.e-9)+C2;

			TC11=1.0;
			TC12=0.0;
			TC13=zeta1;

			TC21=0.0;
			TC22=1.0;
			TC23=zeta2;

			TC31=0.0;
			TC32=0.0;
			TC33=pow(K0,2)/(pow(K,2)+1.e-9);

			double C11_tmp=C11, C12_tmp=C12, C13_tmp=C13;
			double C21_tmp=C21, C22_tmp=C22, C23_tmp=C23;
			double C31_tmp=C31, C32_tmp=C32, C33_tmp=C33;


			C11=TC11*C11_tmp+TC12*C21_tmp+TC13*C31_tmp;
			C12=TC11*C12_tmp+TC12*C22_tmp+TC13*C32_tmp;
			C13=TC11*C13_tmp+TC12*C23_tmp+TC13*C33_tmp;

			C21=TC21*C11_tmp+TC22*C21_tmp+TC23*C31_tmp;
			C22=TC21*C12_tmp+TC22*C22_tmp+TC23*C32_tmp;
			C23=TC21*C13_tmp+TC22*C23_tmp+TC23*C33_tmp;

			C31=TC31*C11_tmp+TC32*C21_tmp+TC33*C31_tmp;
			C32=TC31*C12_tmp+TC32*C22_tmp+TC33*C32_tmp;
			C33=TC31*C13_tmp+TC32*C23_tmp+TC33*C33_tmp;

//             		an array of rank d whose dimensions are n0 × n1 × n2 ×· · · × nd−1
//  			id−1 + nd−1 (id−2 + nd−2 (. . . + n1 i0 ))

			int id=i+Nx*(j+Ny*k);

			CKu[id][0]=C11*n1_Gauss[k][j][i]+C12*n2_Gauss[k][j][i]+C13*n3_Gauss[k][j][i];
			CKu[id][1]=0.0;
			CKv[id][0]=C21*n1_Gauss[k][j][i]+C22*n2_Gauss[k][j][i]+C23*n3_Gauss[k][j][i];
			CKv[id][1]=0.0;
			CKw[id][0]=C31*n1_Gauss[k][j][i]+C32*n2_Gauss[k][j][i]+C33*n3_Gauss[k][j][i];
			CKw[id][1]=0.0;

/*
			double Kx_00=Nx*PI*2.0/Lx/3.0;
			double Ky_00=Ny*PI*2.0/Ly/3.0;
			double Kz_00=Nz*PI*2.0/Lz/3.0;
			if((fabs(Kx[i])-Kx_00)>1.e-19 || (fabs(Ky[j])-Ky_00)>1.e-19 || (fabs(Kz[k])-Kz_00)>1.e-19) {

				CKu[id][0]=0.0;
				CKv[id][0]=0.0;
				CKw[id][0]=0.0;
			}
*/

	}

	double kDivg_Max=0.0;
	double kDivg_Min=1.0e20;
	double kDivg_Sum=0.0;


	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		int id=i+Nx*(j+Ny*k);
		Divg_k[k][j][i] = CKu[id][0]*Kx[i]+CKv[id][0]*Ky[j]+CKw[id][0]*Kz[k];
		kDivg_Sum+=fabs(Divg_k[k][j][i]);
		if ((fabs(Divg_k[k][j][i])-kDivg_Max)>1.e-20) kDivg_Max=fabs(Divg_k[k][j][i]);
		if ((fabs(Divg_k[k][j][i])-kDivg_Min)<1.e-20) kDivg_Min=fabs(Divg_k[k][j][i]);
	}
	
	printf("the Maximum Divergence is %le \n", kDivg_Max);
	printf("the Minimum Divergence is %le \n", kDivg_Min);
	printf("the Average Divergence is %le \n", kDivg_Sum/(Nx*Ny*Nz));
	

	fftw_execute(pu); 
	fftw_execute(pv); 
	fftw_execute(pw); 


	double fac_111=1.0;
	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		int id=i+Nx*(j+Ny*k);
	 	U[k][j][i] = CKu_o[id][0]*fac_111;
	 	V[k][j][i] = CKv_o[id][0]*fac_111;
	 	W[k][j][i] = CKw_o[id][0]*fac_111;
	 	U_i[k][j][i] = CKu_o[id][1]*fac_111;
	 	V_i[k][j][i] = CKv_o[id][1]*fac_111;
	 	W_i[k][j][i] = CKw_o[id][1]*fac_111;
	}



	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		int id=i+Nx*(j+Ny*k);
		CKu[id][1]=CKu[id][0]*Kx[i];
		CKv[id][1]=CKv[id][0]*Ky[j];
		CKw[id][1]=CKw[id][0]*Kz[k];
		
		CKu[id][0]=0.0;
		CKv[id][0]=0.0;
		CKw[id][0]=0.0;

/*
		double Kx_00=Nx*PI*2.0/Lx/3.0;
		double Ky_00=Ny*PI*2.0/Ly/3.0;
		double Kz_00=Nz*PI*2.0/Lz/3.0;
		if((fabs(Kx[i])-Kx_00)>1.e-19 || (fabs(Ky[j])-Ky_00)>1.e-19 || (fabs(Kz[k])-Kz_00)>1.e-19) {

			CKu[id][1]=0.0;
			CKv[id][1]=0.0;
			CKw[id][1]=0.0;
		}

*/
	}



	fftw_execute(pu); 
	fftw_execute(pv); 
	fftw_execute(pw); 

	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		int id=i+Nx*(j+Ny*k);
	 	dUdx[k][j][i] = CKu_o[id][0]*fac_111;
	 	dVdy[k][j][i] = CKv_o[id][0]*fac_111;
	 	dWdz[k][j][i] = CKw_o[id][0]*fac_111;
	 	dUdx_i[k][j][i] = CKu_o[id][1]*fac_111;
	 	dVdy_i[k][j][i] = CKv_o[id][1]*fac_111;
	 	dWdz_i[k][j][i] = CKw_o[id][1]*fac_111;
	}


	fftw_destroy_plan(pu);
	fftw_destroy_plan(pv);
	fftw_destroy_plan(pw);


	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		Divg[k][j][i] = 0.0;
	}
	
	double Divg_Max=0.0;
	double Divg_Min=1.0e20;
	double Divg_Sum=0.0;

	for(k=1;k<Nz-1;k++)
	for(j=1;j<Ny-1;j++) 
	for(i=1;i<Nx-1;i++) {
//		Divg[k][j][i] = 0.5*(U[k][j][i+1]-U[k][j][i-1])/dx+0.5*(V[k][j+1][i]-V[k][j-1][i])/dy+0.5*(W[k+1][j][i]-W[k-1][j][i])/dz;
		Divg[k][j][i] = dUdx[k][j][i]+dVdy[k][j][i]+dWdz[k][j][i];
		Divg_Sum+=fabs(Divg[k][j][i]);
		if ((fabs(Divg[k][j][i])-Divg_Max)>1.e-20) Divg_Max=fabs(Divg[k][j][i]);
		if ((fabs(Divg[k][j][i])-Divg_Min)<1.e-20) Divg_Min=fabs(Divg[k][j][i]);
	}

	printf("the Maximum Divergence is %le \n", Divg_Max);
	printf("the Minimum Divergence is %le \n", Divg_Min);
	printf("the Average Divergence is %le \n", Divg_Sum/(Nx*Ny*Nz));
	
	char filen[80];
	sprintf(filen, "Turbulence%06d.plt", 0);

	INTEGER4 I, Debug, VIsDouble, DIsDouble, III, IMax, JMax, KMax;
	VIsDouble = 0;
	DIsDouble = 0;
	Debug = 0;

	double SolTime = 0.0;
	INTEGER4 StrandID = 0; /* StaticZone */	
	INTEGER4 ParentZn = 0;	
	INTEGER4 TotalNumFaceNodes = 1; /* Not used for FEQuad zones*/
	INTEGER4 NumConnectedBoundaryFaces = 1; /* Not used for FEQuad zones*/
	INTEGER4 TotalNumBoundaryConnections = 1; /* Not used for FEQuad zones*/
	INTEGER4 FileType = 0;


	
	I = TECINI112((char*)"Flow", (char*)"X Y Z U V W U_i V_i W_i Divg_k Divg", filen, (char*)".", &FileType, &Debug, &VIsDouble);

	INTEGER4	ZoneType=0, ICellMax=0, JCellMax=0, KCellMax=0;
	INTEGER4	IsBlock=1, NumFaceConnections=0, FaceNeighborMode=0;
	INTEGER4    ShareConnectivityFromZone=0;
	INTEGER4	LOC[40] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}; /* 1 is cell-centered 0 is node centered */
		

	I = TECZNE112((char*)"Zone 1",
                        &ZoneType,      /* Ordered zone */
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
		x[k* Nx*Ny + j*Nx + i] = Y[j];
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
		x[k* Nx*Ny + j*Nx + i] = U[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = V[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = W[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = U_i[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = V_i[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = W_i[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);

	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = Divg_k[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);

	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = Divg[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);

	I = TECEND112();

	free(Kx);
	free(Ky);
	free(Kz);
//	free(X);
//	free(Y);
//	free(Z);

	for(k=0;k<Nz;k++) 
	for(j=0;j<Ny;j++) {
//		free(U[k][j]);
//		free(V[k][j]);
//		free(W[k][j]);

		free(U_i[k][j]);
		free(V_i[k][j]);
		free(W_i[k][j]);

		free(n1_Gauss[k][j]);
		free(n2_Gauss[k][j]);
		free(n3_Gauss[k][j]);

		free(Divg_k[k][j]);
		free(Divg[k][j]);


		free(dUdx[k][j]);
		free(dVdy[k][j]);
		free(dWdz[k][j]);

		free(dUdx_i[k][j]);
		free(dVdy_i[k][j]);
		free(dWdz_i[k][j]);

	}


	fftw_free(CKu);
	fftw_free(CKu_o);
	fftw_free(CKv);
	fftw_free(CKv_o);
	fftw_free(CKw);
	fftw_free(CKw_o);

}



void GenerateTurb( ) 
{
  
	int t,k,j,i;

	double TC11, TC12, TC13, TC21, TC22, TC23, TC31, TC32, TC33;

	double *Kx, *Ky, *Kz;
//	double *X, *Y, *Z;

	double C11, C12, C13, C21, C22, C23, C31, C32, C33;
//	double ***U, ***V, ***W;
	double ***U_i, ***V_i, ***W_i;
	double ***n1_Gauss, ***n2_Gauss, ***n3_Gauss; 
	double ***Divg_k;
	double ***Divg;
	

	double ***dUdx, ***dVdy, ***dWdz;
	double ***dUdx_i, ***dVdy_i, ***dWdz_i;

/*
	Ly=fabs(X_LES[1][1][Nx_LES-1]-X_LES[1][1][1]);
	Lz=fabs(Y_LES[1][Ny_LES-1][1]-Y_LES[1][1][1]);
//	Lx=fabs(Z_LES[Nz_LES-1][1][1]-Z_LES[1][1][1]);
	Lx=4.0*Ly;

//	Nx=Nz_LES-2;
	Ny=Nx_LES-2;
	Nz=Ny_LES-2;
	Nx=4*Ny;

	Y_loc=0.1*Lz;
*/

	Kx = (double*) malloc(sizeof(double)*Nx);
	Ky = (double*) malloc(sizeof(double)*Ny);
	Kz = (double*) malloc(sizeof(double)*Nz);

	X = (double*) malloc(sizeof(double)*Nx);
	Y = (double*) malloc(sizeof(double)*Ny);
	Z = (double*) malloc(sizeof(double)*Nz);

	U = (double***) malloc(sizeof(double)*Nz);
	V = (double***) malloc(sizeof(double)*Nz);
	W = (double***) malloc(sizeof(double)*Nz);

	U_i = (double***) malloc(sizeof(double)*Nz);
	V_i = (double***) malloc(sizeof(double)*Nz);
	W_i = (double***) malloc(sizeof(double)*Nz);

	n1_Gauss = (double***) malloc(sizeof(double)*Nz);
	n2_Gauss = (double***) malloc(sizeof(double)*Nz);
	n3_Gauss = (double***) malloc(sizeof(double)*Nz);

	Divg_k = (double***) malloc(sizeof(double)*Nz);
	Divg = (double***) malloc(sizeof(double)*Nz);


	dUdx = (double***) malloc(sizeof(double)*Nz);
	dVdy = (double***) malloc(sizeof(double)*Nz);
	dWdz = (double***) malloc(sizeof(double)*Nz);

	dUdx_i = (double***) malloc(sizeof(double)*Nz);
	dVdy_i = (double***) malloc(sizeof(double)*Nz);
	dWdz_i = (double***) malloc(sizeof(double)*Nz);


	for(k=0;k<Nz;k++) {
		U[k] = (double**) malloc(sizeof(double)*Ny);
		V[k] = (double**) malloc(sizeof(double)*Ny);
		W[k] = (double**) malloc(sizeof(double)*Ny);

		U_i[k] = (double**) malloc(sizeof(double)*Ny);
		V_i[k] = (double**) malloc(sizeof(double)*Ny);
		W_i[k] = (double**) malloc(sizeof(double)*Ny);

		n1_Gauss[k] = (double**) malloc(sizeof(double)*Ny);
		n2_Gauss[k] = (double**) malloc(sizeof(double)*Ny);
		n3_Gauss[k] = (double**) malloc(sizeof(double)*Ny);
		Divg_k[k] = (double**) malloc(sizeof(double)*Ny);
		Divg[k] = (double**) malloc(sizeof(double)*Ny);


		dUdx[k] = (double**) malloc(sizeof(double)*Ny);
		dVdy[k] = (double**) malloc(sizeof(double)*Ny);
		dWdz[k] = (double**) malloc(sizeof(double)*Ny);

		dUdx_i[k] = (double**) malloc(sizeof(double)*Ny);
		dVdy_i[k] = (double**) malloc(sizeof(double)*Ny);
		dWdz_i[k] = (double**) malloc(sizeof(double)*Ny);

	}


	for(k=0;k<Nz;k++) 
	for(j=0;j<Ny;j++) {
		U[k][j] = (double*) malloc(sizeof(double)*Nx);
		V[k][j] = (double*) malloc(sizeof(double)*Nx);
		W[k][j] = (double*) malloc(sizeof(double)*Nx);

		U_i[k][j] = (double*) malloc(sizeof(double)*Nx);
		V_i[k][j] = (double*) malloc(sizeof(double)*Nx);
		W_i[k][j] = (double*) malloc(sizeof(double)*Nx);

		n1_Gauss[k][j] = (double*) malloc(sizeof(double)*Nx);
		n2_Gauss[k][j] = (double*) malloc(sizeof(double)*Nx);
		n3_Gauss[k][j] = (double*) malloc(sizeof(double)*Nx);
		Divg_k[k][j] = (double*) malloc(sizeof(double)*Nx);
		Divg[k][j] = (double*) malloc(sizeof(double)*Nx);


		dUdx[k][j] = (double*) malloc(sizeof(double)*Nx);
		dVdy[k][j] = (double*) malloc(sizeof(double)*Nx);
		dWdz[k][j] = (double*) malloc(sizeof(double)*Nx);

		dUdx_i[k][j] = (double*) malloc(sizeof(double)*Nx);
		dVdy_i[k][j] = (double*) malloc(sizeof(double)*Nx);
		dWdz_i[k][j] = (double*) malloc(sizeof(double)*Nx);


	}




	for(i=0;i<Nx/2+1;i++) {
		Kx[i]=i*2.0*PI/Lx;
	}

	for(j=0;j<Ny/2+1;j++) {
		Ky[j]=j*2.0*PI/Ly;
	}

	for(k=0;k<Nz/2+1;k++) {
		Kz[k]=k*2.0*PI/Lz;
	}

	for(i=Nx/2+1;i<Nx;i++) {
		Kx[i]=(i-Nx)*2.0*PI/Lx;
	}

	for(j=Ny/2+1;j<Ny;j++) {
		Ky[j]=(j-Ny)*2.0*PI/Ly;
	}

	for(k=Nz/2+1;k<Nz;k++) {
		Kz[k]=(k-Nz)*2.0*PI/Lz;
	}


//	Kx[Nx/2]=0.0;
//	Ky[Ny/2]=0.0;
//	Kz[Nz/2]=0.0;


	double dy=Ly/(Ny-1);

	for(j=0;j<Ny;j++) {
		Y[j]=double(j)*dy;
	}

	double dx=Lx/(Nx-1);

	for(i=0;i<Nx;i++) {
		X[i]=double(i)*dx;
	}

	double dz=Lz/(Nz-1);

	for(k=0;k<Nz;k++) {
		Z[k]=double(k)*dz;
	}


        FILE *f_tmp;
        char fname_tmp[80];  
        sprintf(fname_tmp, "E_k.dat");
        f_tmp = fopen(fname_tmp, "w");
	j=0; k=0;
	for (i=0;i<Nx;i++) {

		double K = sqrt(Kx[i]*Kx[i]+Ky[j]*Ky[j]+Kz[k]*Kz[k]);
		double fac=1.0/(sqrt(4.0*PI)*K*K+1.e-9);
		double fac1 = D_coef*ustar*ustar/pow(Y_loc,2.0/3.0);
		double L = L_coef*Y_loc;
		double fac2 = pow(L,5.0/3.0);
		double fac3 = pow(L,4)*pow(K,4)/pow((1+pow(L,2)*pow(K,2)),17.0/6.0);

		double E=  fac1 * fac2 * fac3;   // alpha * e^2/3 *  L^5/3 *  L^4 k^4 / (1+L^2K^2)^(17/6) 
		
		double FF=E;
        	fprintf( f_tmp, "%le %le \n", K, FF );
	}


	fclose(f_tmp);

	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		n1_Gauss[k][j][i] = randn_notrig();
		n2_Gauss[k][j][i] = randn_notrig();
		n3_Gauss[k][j][i] = randn_notrig();
		Divg_k[k][j][i] = 0.0;
	}
	

	fftw_complex *CKu, *CKv, *CKw;
	fftw_complex *CKu_o, *CKv_o, *CKw_o;

	CKu = (fftw_complex*) fftw_malloc(Nx*Ny*Nz*sizeof(fftw_complex));
	CKu_o = (fftw_complex*) fftw_malloc(Nx*Ny*Nz*sizeof(fftw_complex));
	CKv = (fftw_complex*) fftw_malloc(Nx*Ny*Nz*sizeof(fftw_complex));
	CKv_o = (fftw_complex*) fftw_malloc(Nx*Ny*Nz*sizeof(fftw_complex));
	CKw = (fftw_complex*) fftw_malloc(Nx*Ny*Nz*sizeof(fftw_complex));
	CKw_o = (fftw_complex*) fftw_malloc(Nx*Ny*Nz*sizeof(fftw_complex));


	fftw_plan pu, pv, pw;

	pu=fftw_plan_dft_3d(Nz, Ny, Nx, CKu, CKu_o, FFTW_BACKWARD, FFTW_ESTIMATE);
	pv=fftw_plan_dft_3d(Nz, Ny, Nx, CKv, CKv_o, FFTW_BACKWARD, FFTW_ESTIMATE);
	pw=fftw_plan_dft_3d(Nz, Ny, Nx, CKw, CKw_o, FFTW_BACKWARD, FFTW_ESTIMATE);


	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {

			double K = sqrt(Kx[i]*Kx[i]+Ky[j]*Ky[j]+Kz[k]*Kz[k]);
			double fac=1.0/(sqrt(4.0*PI)*K*K);
			double fac1 = D_coef*ustar*ustar/pow(Y_loc,2.0/3.0);
			double L = L_coef*Y_loc;
			double fac2 = pow(L,5.0/3.0);
			double fac3 = pow(L,4)*pow(K,4)/pow((1+pow(L,2)*pow(K,2)),17.0/6.0);

			double E=  fac1 * fac2 * fac3;   // alpha * e^2/3 *  L^5/3 *  L^4 k^4 / (1+L^2K^2)^(17/6) 

			double dkx=2*PI/Lx, dky=2*PI/Ly, dkz=2*PI/Lz;
		
			double fac4=sqrt(dkx*dky*dkz);

			double facc = fac4*sqrt(E)*fac;

			if (fabs(K)<1.e-20) facc=0.0;

			C11=0.0; C22=0.0; C33=0.0; 

			C12=facc * Kz[k] ; C13=-facc * Ky[j] ; 
			C21=-facc * Kz[k] ; C23=facc * Kx[i] ; 
			C31=facc * Ky[j] ; C32=-facc * Kx[i] ; 

			Beta=Gamma*(ustar/Karman_const/Y_loc)*pow((K*L+1.e-9),-2.0/3.0);

			double K30=Kz[k];
			double K3=K30-Beta*Kx[i];

			double K0=K;
			K = sqrt(Kx[i]*Kx[i]+Ky[j]*Ky[j]+K3*K3);

			double C1=Beta*pow(Kx[i],2)*(pow(K0,2)-2*pow(K30,2)+Beta*Kx[i]*K30)/(pow(K,2)*(pow(Kx[i],2)+pow(Ky[j],2))+1.e-9);

			double tmp=atan(Beta*Kx[i]*sqrt(pow(Kx[i],2)+pow(Ky[j],2))/(pow(K0,2)-K30*Kx[i]*Beta+1.e-9));
			double C2=Ky[j]*pow(K0,2)*tmp/(pow(pow(Kx[i],2)+pow(Ky[j],2),3.0/2.0)+1.e-9);

			double zeta1=C1-Ky[j]*C2/(Kx[i]+1.e-9);

			double zeta2=Ky[j]*C1/(Kx[i]+1.e-9)+C2;

			TC11=1.0;
			TC12=0.0;
			TC13=zeta1;

			TC21=0.0;
			TC22=1.0;
			TC23=zeta2;

			TC31=0.0;
			TC32=0.0;
			TC33=pow(K0,2)/(pow(K,2)+1.e-9);

			double C11_tmp=C11, C12_tmp=C12, C13_tmp=C13;
			double C21_tmp=C21, C22_tmp=C22, C23_tmp=C23;
			double C31_tmp=C31, C32_tmp=C32, C33_tmp=C33;


			C11=TC11*C11_tmp+TC12*C21_tmp+TC13*C31_tmp;
			C12=TC11*C12_tmp+TC12*C22_tmp+TC13*C32_tmp;
			C13=TC11*C13_tmp+TC12*C23_tmp+TC13*C33_tmp;

			C21=TC21*C11_tmp+TC22*C21_tmp+TC23*C31_tmp;
			C22=TC21*C12_tmp+TC22*C22_tmp+TC23*C32_tmp;
			C23=TC21*C13_tmp+TC22*C23_tmp+TC23*C33_tmp;

			C31=TC31*C11_tmp+TC32*C21_tmp+TC33*C31_tmp;
			C32=TC31*C12_tmp+TC32*C22_tmp+TC33*C32_tmp;
			C33=TC31*C13_tmp+TC32*C23_tmp+TC33*C33_tmp;

//             		an array of rank d whose dimensions are n0 × n1 × n2 ×· · · × nd−1
//  			id−1 + nd−1 (id−2 + nd−2 (. . . + n1 i0 ))

			int id=i+Nx*(j+Ny*k);

			CKu[id][0]=C11*n1_Gauss[k][j][i]+C12*n2_Gauss[k][j][i]+C13*n3_Gauss[k][j][i];
			CKu[id][1]=0.0;
			CKv[id][0]=C21*n1_Gauss[k][j][i]+C22*n2_Gauss[k][j][i]+C23*n3_Gauss[k][j][i];
			CKv[id][1]=0.0;
			CKw[id][0]=C31*n1_Gauss[k][j][i]+C32*n2_Gauss[k][j][i]+C33*n3_Gauss[k][j][i];
			CKw[id][1]=0.0;

/*
			double Kx_00=Nx*PI*2.0/Lx/3.0;
			double Ky_00=Ny*PI*2.0/Ly/3.0;
			double Kz_00=Nz*PI*2.0/Lz/3.0;
			if((fabs(Kx[i])-Kx_00)>1.e-19 || (fabs(Ky[j])-Ky_00)>1.e-19 || (fabs(Kz[k])-Kz_00)>1.e-19) {

				CKu[id][0]=0.0;
				CKv[id][0]=0.0;
				CKw[id][0]=0.0;
			}
*/

	}

	double kDivg_Max=0.0;
	double kDivg_Min=1.0e20;
	double kDivg_Sum=0.0;


	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		int id=i+Nx*(j+Ny*k);
		Divg_k[k][j][i] = CKu[id][0]*Kx[i]+CKv[id][0]*Ky[j]+CKw[id][0]*Kz[k];
		kDivg_Sum+=fabs(Divg_k[k][j][i]);
		if ((fabs(Divg_k[k][j][i])-kDivg_Max)>1.e-20) kDivg_Max=fabs(Divg_k[k][j][i]);
		if ((fabs(Divg_k[k][j][i])-kDivg_Min)<1.e-20) kDivg_Min=fabs(Divg_k[k][j][i]);
	}
	
	printf("the Maximum Divergence is %le \n", kDivg_Max);
	printf("the Minimum Divergence is %le \n", kDivg_Min);
	printf("the Average Divergence is %le \n", kDivg_Sum/(Nx*Ny*Nz));
	

	fftw_execute(pu); 
	fftw_execute(pv); 
	fftw_execute(pw); 


	double fac_111=1.0;
	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		int id=i+Nx*(j+Ny*k);
	 	U[k][j][i] = CKu_o[id][0]*fac_111;
	 	V[k][j][i] = CKv_o[id][0]*fac_111;
	 	W[k][j][i] = CKw_o[id][0]*fac_111;
	 	U_i[k][j][i] = CKu_o[id][1]*fac_111;
	 	V_i[k][j][i] = CKv_o[id][1]*fac_111;
	 	W_i[k][j][i] = CKw_o[id][1]*fac_111;
	}



	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		int id=i+Nx*(j+Ny*k);
		CKu[id][1]=CKu[id][0]*Kx[i];
		CKv[id][1]=CKv[id][0]*Ky[j];
		CKw[id][1]=CKw[id][0]*Kz[k];
		
		CKu[id][0]=0.0;
		CKv[id][0]=0.0;
		CKw[id][0]=0.0;

/*
		double Kx_00=Nx*PI*2.0/Lx/3.0;
		double Ky_00=Ny*PI*2.0/Ly/3.0;
		double Kz_00=Nz*PI*2.0/Lz/3.0;
		if((fabs(Kx[i])-Kx_00)>1.e-19 || (fabs(Ky[j])-Ky_00)>1.e-19 || (fabs(Kz[k])-Kz_00)>1.e-19) {

			CKu[id][1]=0.0;
			CKv[id][1]=0.0;
			CKw[id][1]=0.0;
		}

*/
	}



	fftw_execute(pu); 
	fftw_execute(pv); 
	fftw_execute(pw); 

	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		int id=i+Nx*(j+Ny*k);
	 	dUdx[k][j][i] = CKu_o[id][0]*fac_111;
	 	dVdy[k][j][i] = CKv_o[id][0]*fac_111;
	 	dWdz[k][j][i] = CKw_o[id][0]*fac_111;
	 	dUdx_i[k][j][i] = CKu_o[id][1]*fac_111;
	 	dVdy_i[k][j][i] = CKv_o[id][1]*fac_111;
	 	dWdz_i[k][j][i] = CKw_o[id][1]*fac_111;
	}


	fftw_destroy_plan(pu);
	fftw_destroy_plan(pv);
	fftw_destroy_plan(pw);


	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++) 
	for(i=0;i<Nx;i++) {
		Divg[k][j][i] = 0.0;
	}
	
	double Divg_Max=0.0;
	double Divg_Min=1.0e20;
	double Divg_Sum=0.0;

	for(k=1;k<Nz-1;k++)
	for(j=1;j<Ny-1;j++) 
	for(i=1;i<Nx-1;i++) {
//		Divg[k][j][i] = 0.5*(U[k][j][i+1]-U[k][j][i-1])/dx+0.5*(V[k][j+1][i]-V[k][j-1][i])/dy+0.5*(W[k+1][j][i]-W[k-1][j][i])/dz;
		Divg[k][j][i] = dUdx[k][j][i]+dVdy[k][j][i]+dWdz[k][j][i];
		Divg_Sum+=fabs(Divg[k][j][i]);
		if ((fabs(Divg[k][j][i])-Divg_Max)>1.e-20) Divg_Max=fabs(Divg[k][j][i]);
		if ((fabs(Divg[k][j][i])-Divg_Min)<1.e-20) Divg_Min=fabs(Divg[k][j][i]);
	}

	printf("the Maximum Divergence is %le \n", Divg_Max);
	printf("the Minimum Divergence is %le \n", Divg_Min);
	printf("the Average Divergence is %le \n", Divg_Sum/(Nx*Ny*Nz));
	
	char filen[80];
	sprintf(filen, "Turbulence%06d.plt", 0);

	INTEGER4 I, Debug, VIsDouble, DIsDouble, III, IMax, JMax, KMax;
	VIsDouble = 0;
	DIsDouble = 0;
	Debug = 0;

	double SolTime = 0.0;
	INTEGER4 StrandID = 0; /* StaticZone */	
	INTEGER4 ParentZn = 0;	
	INTEGER4 TotalNumFaceNodes = 1; /* Not used for FEQuad zones*/
	INTEGER4 NumConnectedBoundaryFaces = 1; /* Not used for FEQuad zones*/
	INTEGER4 TotalNumBoundaryConnections = 1; /* Not used for FEQuad zones*/
	INTEGER4 FileType = 0;


	
	I = TECINI112((char*)"Flow", (char*)"X Y Z U V W U_i V_i W_i Divg_k Divg", filen, (char*)".", &FileType, &Debug, &VIsDouble);

	INTEGER4	ZoneType=0, ICellMax=0, JCellMax=0, KCellMax=0;
	INTEGER4	IsBlock=1, NumFaceConnections=0, FaceNeighborMode=0;
	INTEGER4    ShareConnectivityFromZone=0;
	INTEGER4	LOC[40] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}; /* 1 is cell-centered 0 is node centered */
		

	I = TECZNE112((char*)"Zone 1",
                        &ZoneType,      /* Ordered zone */
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
		x[k* Nx*Ny + j*Nx + i] = Y[j];
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
		x[k* Nx*Ny + j*Nx + i] = U[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = V[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = W[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = U_i[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = V_i[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);


	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = W_i[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);

	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = Divg_k[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);

	for (k=0; k<Nz; k++)
	for (j=0; j<Ny; j++)
	for (i=0; i<Nx; i++) {
		x[k* Nx*Ny + j*Nx + i] = Divg[k][j][i];
	}
	I = TECDAT112(&III, &x[0], &DIsDouble);

	I = TECEND112();

	free(Kx);
	free(Ky);
	free(Kz);
//	free(X);
//	free(Y);
//	free(Z);

	for(k=0;k<Nz;k++) 
	for(j=0;j<Ny;j++) {
//		free(U[k][j]);
//		free(V[k][j]);
//		free(W[k][j]);

		free(U_i[k][j]);
		free(V_i[k][j]);
		free(W_i[k][j]);

		free(n1_Gauss[k][j]);
		free(n2_Gauss[k][j]);
		free(n3_Gauss[k][j]);

		free(Divg_k[k][j]);
		free(Divg[k][j]);


		free(dUdx[k][j]);
		free(dVdy[k][j]);
		free(dWdz[k][j]);

		free(dUdx_i[k][j]);
		free(dVdy_i[k][j]);
		free(dWdz_i[k][j]);

	}


	fftw_free(CKu);
	fftw_free(CKu_o);
	fftw_free(CKv);
	fftw_free(CKv_o);
	fftw_free(CKw);
	fftw_free(CKw_o);

}



double randn_notrig() {


	static bool deviateAvailable=false;   
	static float storedDeviate;
	double polar, rsquared, var1, var2;
	double mu=0.0; double sigma=1.0;
	if (!deviateAvailable) {
		do {
			var1=2.0*( double(rand())/double(RAND_MAX) ) - 1.0;
			var2=2.0*( double(rand())/double(RAND_MAX) ) - 1.0;
			rsquared=var1*var1+var2*var2;
		} while ( rsquared>=1.0 || rsquared == 0.0);

		polar=sqrt(-2.0*log(rsquared)/rsquared);
		storedDeviate=var1*polar;
		deviateAvailable=true;
		return var2*polar*sigma + mu;
	}
	else {
		deviateAvailable=false;
		return storedDeviate*sigma + mu;
	}
}




