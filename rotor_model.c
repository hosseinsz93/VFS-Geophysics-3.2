/*Modeling the rotor blades */

#include "variables.h"

#include <algorithm>
using namespace std;

extern	double dfunc_2h(double r);
extern	double dfunc_4h(double r);
extern  double dfunc_s3h(double r);
extern  double dfunc_s4h(double r);
extern double dfunc_nh(double r, double n);

extern	PetscErrorCode Calc_U_lagr(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int NumberOfObjects); 

extern Cmpnts ArbitraryRotate(Cmpnts p,double theta,Cmpnts r);
double  coef_cr1 = 2.5, coef_cr2 = 1.0;

int Itpwidth=4;


PetscErrorCode disk_read_ucd(IBMNodes *ibm, int ibi, FSInfo *fsi, int OneDmZ, char fname[80])
{
  	int	rank;
  	int	n_v , n_elmt ;
  	PetscReal	*x_bp , *y_bp , *z_bp ;
  	int	*nv1 , *nv2 , *nv3 ;
  	PetscReal	*nf_x, *nf_y, *nf_z;
  	int	i,ii;
  	int	n1e, n2e, n3e;
  	PetscReal	dx12, dy12, dz12, dx13, dy13, dz13;
  	PetscReal     dr;
  	PetscReal     *dA ;//area
  	PetscReal	*nt_x, *nt_y, *nt_z;
  	PetscReal	*ns_x, *ns_y, *ns_z;

  	char   ss[20];
  	char string[128];

  	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  	if(!rank) { // root processor read in the data
    		FILE *fd;
    		PetscPrintf(PETSC_COMM_SELF, "READ %s\n", fname);
    		char filen[160];  
    		sprintf(filen,"%s/%s" ,path, fname);
 
    		fd = fopen(filen, "r"); 
    		if (!fd) printf("Cannot open %s !!\n", filen),exit(1);
    		else printf("Opened %s !\n", filen);
    		n_v =0;

    		if (fd) {
      			fgets(string, 128, fd);
      			fgets(string, 128, fd);
      			fgets(string, 128, fd);
      
      			fscanf(fd, "%i %i %i %i %i",&n_v,&n_elmt,&ii,&ii,&ii);
      			PetscPrintf(PETSC_COMM_SELF, "number of nodes & elements %d %d\n",n_v, n_elmt);
      
      			ibm->n_v = n_v;
      			ibm->n_elmt = n_elmt;      
      
      			MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
      
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));	// added by seokkoo 03.04.2009
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
      
      			x_bp = ibm->x_bp;	// seokkoo
      			y_bp = ibm->y_bp;
      			z_bp = ibm->z_bp;

      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_i));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_i));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_i));
      
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
 
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

      			for (i=0; i<n_v; i++) {
				fscanf(fd, "%i %le %le %le", &ii, &x_bp[i], &y_bp[i], &z_bp[i]);//, &t, &t, &t);
	
				x_bp[i]=x_bp[i]/reflength_wt;
				y_bp[i]=y_bp[i]/reflength_wt;
				z_bp[i]=z_bp[i]/reflength_wt;

				// rotate the axis. Assume the one from gridgen is (0,0,1)
				/*
				double n_rot[3];
				n_rot[0]=-fsi->ny_tb; n_rot[1]=fsi->nx_tb; n_rot[2]=0.0;
				double ux=n_rot[0], uy=n_rot[1], uz=n_rot[2];

				double ang_rot;
				double rrr=fsi->nz_tb;
				ang_rot=acos(rrr);
				double cos_=cos(ang_rot), sin_=sin(ang_rot);
				*/

				// Problem for some cases    .........................
				/*
				Cmpnts p, r, q;
				p.x=x_bp[i]; p.y=y_bp[i]; p.z=z_bp[i];
				r.x=-fsi->ny_tb; r.y=fsi->nx_tb; r.z=0.0;
				double theta=acos(fsi->nz_tb);
				q=ArbitraryRotate(p,theta,r);
				x_bp[i]=q.x;	
				y_bp[i]=q.y;	
				z_bp[i]=q.z;	
				*/
				// 				.................................
				
				/*
				double mat_rot[3][3];

				mat_rot[0][0]=cos_+pow(ux,2)*(1.0-cos_);
				mat_rot[0][1]=ux*uy*(1.0-cos_)-uz*sin_;
				mat_rot[0][2]=ux*uz*(1.0-cos_)+uy*sin_;

				mat_rot[1][0]=uy*ux*(1.0-cos_)+uz*sin_;
				mat_rot[1][1]=cos_+pow(uy,2)*(1.0-cos_);
				mat_rot[1][2]=uy*uz*(1.0-cos_)-ux*sin_;

				mat_rot[2][0]=uz*ux*(1.0-cos_)-uy*sin_;
				mat_rot[2][1]=uz*uy*(1.0-cos_)+ux*sin_;
				mat_rot[2][2]=cos_+pow(uz,2)*(1.0-cos_);
				*/

				// double xb=x_bp[i], yb=y_bp[i], zb=z_bp[i];

				/*
				x_bp[i]=mat_rot[0][0]*xb+mat_rot[0][1]*yb+mat_rot[0][2]*zb;
				y_bp[i]=mat_rot[1][0]*xb+mat_rot[1][1]*yb+mat_rot[1][2]*zb;
				z_bp[i]=mat_rot[2][0]*xb+mat_rot[2][1]*yb+mat_rot[2][2]*zb;
				*/

				/*
				double wCv[3];
				wCv[0]=uy*zb-uz*yb; wCv[1]=uz*xb-ux*zb; wCv[2]=ux*yb-uy*xb;
				double wPv=ux*xb+uy*yb+uz*zb;
				x_bp[i]=xb*cos_+wCv[0]*sin_+wPv*ux*(1.0-cos_);
				y_bp[i]=yb*cos_+wCv[1]*sin_+wPv*uy*(1.0-cos_);
				z_bp[i]=zb*cos_+wCv[2]*sin_+wPv*uz*(1.0-cos_);
				*/

			        ibm->x_bp_i[i] = x_bp[i];
			        ibm->y_bp_i[i] = y_bp[i];
			        ibm->z_bp_i[i] = z_bp[i];

			        x_bp[i] += fsi->x_c;
			        y_bp[i] += fsi->y_c;
			        z_bp[i] += fsi->z_c;

			        ibm->x_bp0[i] = x_bp[i];
			        ibm->y_bp0[i] = y_bp[i];
			        ibm->z_bp0[i] = z_bp[i];


				if (OneDmZ) {
					double rr = 2.0*r_rotor/reflength_wt;
					x_bp[i]-=rr*fsi->nx_tb; 
					y_bp[i]-=rr*fsi->ny_tb; 
					z_bp[i]-=rr*fsi->nz_tb; 
				}
	
				ibm->x_bp[i] = x_bp[i];
				ibm->y_bp[i] = y_bp[i];
				ibm->z_bp[i] = z_bp[i];

				ibm->x_bp_o[i] = x_bp[i];
				ibm->y_bp_o[i] = y_bp[i];
				ibm->z_bp_o[i] = z_bp[i];

				ibm->u[i].x = 0.;
				ibm->u[i].y = 0.;
				ibm->u[i].z = 0.;

				ibm->uold[i].x = 0.;
				ibm->uold[i].y = 0.;
				ibm->uold[i].z = 0.;

				ibm->urm1[i].x = 0.;
				ibm->urm1[i].y = 0.;
				ibm->urm1[i].z = 0.;
			}
      			i=0;
		      	PetscPrintf(PETSC_COMM_WORLD, "xyz_bp %le %le %le\n", x_bp[i], y_bp[i], z_bp[i]);

      			MPI_Bcast(ibm->x_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      			MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      			MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
		      	MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      			MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);

      			PetscMalloc(n_elmt*sizeof(int), &nv1);
      			PetscMalloc(n_elmt*sizeof(int), &nv2);
      			PetscMalloc(n_elmt*sizeof(int), &nv3);
      
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);
      
      			PetscMalloc(n_elmt*sizeof(PetscReal), &dA); //Area

      			PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

      			PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);
      
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));

			// rotor model

        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_x));
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_y));
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_z));

		        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_x));
		        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_y));
		        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_z));


		        PetscMalloc(n_elmt*sizeof(int), &(ibm->i_min));
		        PetscMalloc(n_elmt*sizeof(int), &(ibm->j_min));
		        PetscMalloc(n_elmt*sizeof(int), &(ibm->k_min));

		        PetscMalloc(n_elmt*sizeof(int), &(ibm->i_max));
		        PetscMalloc(n_elmt*sizeof(int), &(ibm->j_max));
		        PetscMalloc(n_elmt*sizeof(int), &(ibm->k_max));

			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhx));
       		        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhy));
                	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhz));


        		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_attack));
        		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_pitch));
        		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->chord_blade));

                       	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CD));
                       	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CL));


      			for (i=0; i<n_elmt; i++) {
				fscanf(fd, "%i %i %s %i %i %i\n", &ii,&ii, &ss, nv1+i, nv2+i, nv3+i);
				nv1[i] = nv1[i] - 1; nv2[i] = nv2[i]-1; nv3[i] = nv3[i] - 1;
		      	}
		        ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;

		        i=0;
		        PetscPrintf(PETSC_COMM_WORLD, "nv %d %d %d\n", nv1[i], nv2[i], nv3[i]);

		        fclose(fd);
 	      	}
      
	      	for (i=0; i<n_elmt; i++) {
      
			n1e = nv1[i]; n2e =nv2[i]; n3e = nv3[i];
		        dx12 = x_bp[n2e] - x_bp[n1e];
		        dy12 = y_bp[n2e] - y_bp[n1e];
		        dz12 = z_bp[n2e] - z_bp[n1e];
      
		        dx13 = x_bp[n3e] - x_bp[n1e];
		        dy13 = y_bp[n3e] - y_bp[n1e];
		        dz13 = z_bp[n3e] - z_bp[n1e];
      
		        nf_x[i] = dy12 * dz13 - dz12 * dy13;
		        nf_y[i] = -dx12 * dz13 + dz12 * dx13;
		        nf_z[i] = dx12 * dy13 - dy12 * dx13;
      
		        dr = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i] + nf_z[i]*nf_z[i]);
      
		        nf_x[i] /=dr; nf_y[i]/=dr; nf_z[i]/=dr;
      
		      	if ((((1.-nf_z[i])<=1e-6 )&((-1.+nf_z[i])<1e-6))|
			  (((nf_z[i]+1.)<=1e-6 )&((-1.-nf_z[i])<1e-6))) {
				ns_x[i] = 1.;     
				ns_y[i] = 0.;     
				ns_z[i] = 0. ;
	
				nt_x[i] = 0.;
				nt_y[i] = 1.;
				nt_z[i] = 0.;
		        } else {
				ns_x[i] =  nf_y[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);      
				ns_y[i] = -nf_x[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);     
				ns_z[i] = 0. ;
	
				nt_x[i] = -nf_x[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
				nt_y[i] = -nf_y[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
				nt_z[i] = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
		        }
      
		        dA[i] = dr/2.; 
      
      			// Calc the center of the element
      			ibm->cent_x[i]= (x_bp[n1e]+x_bp[n2e]+x_bp[n3e])/3.;
		        ibm->cent_y[i]= (y_bp[n1e]+y_bp[n2e]+y_bp[n3e])/3.;
		        ibm->cent_z[i]= (z_bp[n1e]+z_bp[n2e]+z_bp[n3e])/3.;	
		}
    
    
    		ibm->nf_x = nf_x; ibm->nf_y = nf_y;  ibm->nf_z = nf_z;
    
    		ibm->dA = dA;
    		ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
	        ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    
    
	        MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    
    	        MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
    		MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    		MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    		MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    		MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

    		int ti=0;
    		FILE *f;
    		sprintf(filen, "%s/%s_%2.2d_nf.dat",path,fname,ibi);
    		f = fopen(filen, "w");
    		PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z,n_x,n_y,n_z,nt_x,nt_y,nt_z,ns_x,ns_y,ns_z\n");
    		PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-12]=CELLCENTERED)\n", n_v, n_elmt);
    		for (i=0; i<n_v; i++) {
      			PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->x_bp[i]);
    		}
    		for (i=0; i<n_v; i++) {
   	   		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->y_bp[i]);
	    	}
    		for (i=0; i<n_v; i++) {	
	  	      	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->z_bp[i]);
    		}
	        for (i=0; i<n_elmt; i++) {
		        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_x[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
		        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_y[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
		        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_z[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
	 	        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_x[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
		        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_y[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
		        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_z[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
		        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_x[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
	   	        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_y[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
	 	        PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_z[i]);
	        }
	        for (i=0; i<n_elmt; i++) {
	 	        PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
	        }
    
    		fclose(f);

  	}
  	else if (rank) {
    		MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
	        ibm->n_v = n_v;
    
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
	  
		x_bp = ibm->x_bp;	// added by seokkoo 03.04.2009
		y_bp = ibm->y_bp;
		z_bp = ibm->z_bp;
	  
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_i));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_i));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_i));
    
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));
 
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
	        PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
    
    
	        PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
	        PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
	        PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

	        for (i=0; i<n_v; i++) {
	  		ibm->u[i].x = 0.;
  		        ibm->u[i].y = 0.;
	    	        ibm->u[i].z = 0.;

		        ibm->uold[i].x = 0.;
		        ibm->uold[i].y = 0.;
	   	        ibm->uold[i].z = 0.;
      
	  	        ibm->urm1[i].x = 0.;
		        ibm->urm1[i].y = 0.;
	 	        ibm->urm1[i].z = 0.;      
	        }

	        MPI_Bcast(ibm->x_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->y_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->z_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
        
	        MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);    
	        MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

	        MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

 	        MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	       	MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
	       	MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    

	       	MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);
	       	n_elmt = ibm->n_elmt;

	       	PetscMalloc(n_elmt*sizeof(int), &nv1);
	       	PetscMalloc(n_elmt*sizeof(int), &nv2);
	       	PetscMalloc(n_elmt*sizeof(int), &nv3);

	       	PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);

	       	PetscMalloc(n_elmt*sizeof(PetscReal), &dA);

   	       	PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

	       	PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);

	       	ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;
	       	ibm->nf_x = nf_x; ibm->nf_y = nf_y; ibm->nf_z = nf_z;
    
	       	ibm->dA = dA;
	       	ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
	       	ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    

	       	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));

	       	// rotor model
	       	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_x));
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_y));
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_z));

	        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_x));
	        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_y));
	        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_z));

	        PetscMalloc(n_elmt*sizeof(int), &(ibm->i_min));
	        PetscMalloc(n_elmt*sizeof(int), &(ibm->j_min));
	        PetscMalloc(n_elmt*sizeof(int), &(ibm->k_min));

	        PetscMalloc(n_elmt*sizeof(int), &(ibm->i_max));
	        PetscMalloc(n_elmt*sizeof(int), &(ibm->j_max));
	        PetscMalloc(n_elmt*sizeof(int), &(ibm->k_max));

		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhx));
          	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhy));
              	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhz));

        	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_attack));
        	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_pitch));
        	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->chord_blade));

                PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CD));
                PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CL));


	    	MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

	        MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
	        MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

	        MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
	        MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

	        MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	        MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	}
	PetscPrintf(PETSC_COMM_WORLD, "Read %s !\n", fname);

  	return(0);
}

/*
// called in main.c after reading IB grid
// for actuator disc model

PetscErrorCode ACD_read(IBMNodes *ibm, int ibi, FSInfo *fsi, int OneDmZ)
{
  	int	rank;
  	int	n_v , n_elmt ;
  	PetscReal	*x_bp , *y_bp , *z_bp ;
  int	*nv1 , *nv2 , *nv3 ;
  PetscReal	*nf_x, *nf_y, *nf_z;
  int	i,ii;
  int	n1e, n2e, n3e;
  PetscReal	dx12, dy12, dz12, dx13, dy13, dz13;
  PetscReal     dr;
  //Added 4/1/06 iman
  PetscReal     *dA ;//area
  PetscReal	*nt_x, *nt_y, *nt_z;
  PetscReal	*ns_x, *ns_y, *ns_z;

  char   ss[20];
  //double xt;
  char string[128];


//  	PetscReal cl = 1.;
//	PetscOptionsGetReal(PETSC_NULL, "-chact_leng", &cl, PETSC_NULL);


  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if(!rank) { // root processor read in the data
    FILE *fd;
    PetscPrintf(PETSC_COMM_SELF, "READ acddata\n");
    char filen[80];  
    sprintf(filen,"%s/acddata%3.3d" , path, 0);
 
    fd = fopen(filen, "r"); 
    if (!fd) printf("Cannot open %s !!\n", filen),exit(0);
    else printf("Opened %s !\n", filen);
    n_v =0;

    if (fd) {
      fgets(string, 128, fd);
      fgets(string, 128, fd);
      fgets(string, 128, fd);
      
      fscanf(fd, "%i %i %i %i %i",&n_v,&n_elmt,&ii,&ii,&ii);
      PetscPrintf(PETSC_COMM_SELF, "number of nodes & elements %d %d\n",n_v, n_elmt);
      
      ibm->n_v = n_v;
      ibm->n_elmt = n_elmt;      
      
      MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
      //    PetscPrintf(PETSC_COMM_WORLD, "nv, %d %e \n", n_v, xt);
	    
	    
//      PetscMalloc(n_v*sizeof(PetscReal), &x_bp);	// removed by seokkoo 03.04.2009
//      PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
//      PetscMalloc(n_v*sizeof(PetscReal), &z_bp);
      
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));	// added by seokkoo 03.04.2009
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
      
      x_bp = ibm->x_bp;	// seokkoo
      y_bp = ibm->y_bp;
      z_bp = ibm->z_bp;

      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_i));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_i));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_i));
      
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
 
      PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
      PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
      PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

      for (i=0; i<n_v; i++) {
	fscanf(fd, "%i %le %le %le", &ii, &x_bp[i], &y_bp[i], &z_bp[i]);//, &t, &t, &t);
	
//	x_bp[i] = x_bp[i]*L_dim + CMx_c;//0.25 ;// 24.;	
//	y_bp[i] = y_bp[i]*L_dim + CMy_c ;//+ ibi*2.;//5.;//2.;//8.;//6.;//2.   ;// 24.;
//	z_bp[i] = z_bp[i]*L_dim + CMz_c ;//+ ibi*1.5;//2.;//8.;//15.;//2.   ;// 24.;


	x_bp[i]=x_bp[i]/reflength_wt;
	y_bp[i]=y_bp[i]/reflength_wt;
	z_bp[i]=z_bp[i]/reflength_wt;

	// rotate the axis. Assume the one from gridgen is (0,0,1)
	
	double n_rot[3];
	n_rot[0]=-fsi->ny_tb; n_rot[1]=fsi->nx_tb; n_rot[2]=0.0;
	double ux=n_rot[0], uy=n_rot[1], uz=n_rot[2];

	double ang_rot;
	double rrr=fsi->nz_tb;
	ang_rot=acos(rrr);
	double cos_=cos(ang_rot), sin_=sin(ang_rot);

	
	//double mat_rot[3][3];

	//mat_rot[0][0]=cos_+pow(ux,2)*(1.0-cos_);
	//mat_rot[0][1]=ux*uy*(1.0-cos_)-uz*sin_;
	//mat_rot[0][2]=ux*uz*(1.0-cos_)+uy*sin_;

	//mat_rot[1][0]=uy*ux*(1.0-cos_)+uz*sin_;
	//mat_rot[1][1]=cos_+pow(uy,2)*(1.0-cos_);
	//mat_rot[1][2]=uy*uz*(1.0-cos_)-ux*sin_;

	//mat_rot[2][0]=uz*ux*(1.0-cos_)-uy*sin_;
	//mat_rot[2][1]=uz*uy*(1.0-cos_)+ux*sin_;
	//mat_rot[2][2]=cos_+pow(uz,2)*(1.0-cos_);
	

	double xb=x_bp[i], yb=y_bp[i], zb=z_bp[i];

	
	//x_bp[i]=mat_rot[0][0]*xb+mat_rot[0][1]*yb+mat_rot[0][2]*zb;
	//y_bp[i]=mat_rot[1][0]*xb+mat_rot[1][1]*yb+mat_rot[1][2]*zb;
	//z_bp[i]=mat_rot[2][0]*xb+mat_rot[2][1]*yb+mat_rot[2][2]*zb;
	

	double wCv[3];
	wCv[0]=uy*zb-uz*yb; wCv[1]=uz*xb-ux*zb; wCv[2]=ux*yb-uy*xb;
	double wPv=ux*xb+uy*yb+uz*zb;
	x_bp[i]=xb*cos_+wCv[0]*sin_+wPv*ux*(1.0-cos_);
	y_bp[i]=yb*cos_+wCv[1]*sin_+wPv*uy*(1.0-cos_);
	z_bp[i]=zb*cos_+wCv[2]*sin_+wPv*uz*(1.0-cos_);


        ibm->x_bp_i[i] = x_bp[i];
        ibm->y_bp_i[i] = y_bp[i];
        ibm->z_bp_i[i] = z_bp[i];

        x_bp[i] += fsi->x_c;
        y_bp[i] += fsi->y_c;
        z_bp[i] += fsi->z_c;

        ibm->x_bp0[i] = x_bp[i];
        ibm->y_bp0[i] = y_bp[i];
        ibm->z_bp0[i] = z_bp[i];

//        ibm->x_bp0[i] = x_bp[i];
//        ibm->y_bp0[i] = y_bp[i];
//        ibm->z_bp0[i] = z_bp[i];


	if (OneDmZ) z_bp[i]-=2.0*r_rotor/reflength_wt; // flow is in the z-direction
	
	ibm->x_bp[i] = x_bp[i];
	ibm->y_bp[i] = y_bp[i];
	ibm->z_bp[i] = z_bp[i];

	ibm->x_bp_o[i] = x_bp[i];
	ibm->y_bp_o[i] = y_bp[i];
	ibm->z_bp_o[i] = z_bp[i];

	ibm->u[i].x = 0.;
	ibm->u[i].y = 0.;
	ibm->u[i].z = 0.;

	ibm->uold[i].x = 0.;
	ibm->uold[i].y = 0.;
	ibm->uold[i].z = 0.;

	ibm->urm1[i].x = 0.;
	ibm->urm1[i].y = 0.;
	ibm->urm1[i].z = 0.;
      }
      i=0;
      PetscPrintf(PETSC_COMM_WORLD, "xyz_bp %le %le %le\n", x_bp[i], y_bp[i], z_bp[i]);

//       ibm->x_bp0 = x_bp; ibm->y_bp0 = y_bp; ibm->z_bp0 = z_bp; 
//       ibm->x_bp_o = x_bp; ibm->y_bp_o = y_bp; ibm->z_bp_o = z_bp; 

      MPI_Bcast(ibm->x_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->y_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->z_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
      MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);


      PetscMalloc(n_elmt*sizeof(int), &nv1);
      PetscMalloc(n_elmt*sizeof(int), &nv2);
      PetscMalloc(n_elmt*sizeof(int), &nv3);
      

      PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);
      
      // Added 4/1/06 iman
      PetscMalloc(n_elmt*sizeof(PetscReal), &dA); //Area

      PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
      PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

      PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);
      
      // Added 6/4/06 iman
      //PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->cent));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));
      // end added

      // added 12-7-2010 xyang
      if (rotor_model) {
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_x));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_y));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_z));

        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_x));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_y));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_z));


        PetscMalloc(n_elmt*sizeof(int), &(ibm->i_min));
        PetscMalloc(n_elmt*sizeof(int), &(ibm->j_min));
        PetscMalloc(n_elmt*sizeof(int), &(ibm->k_min));

        PetscMalloc(n_elmt*sizeof(int), &(ibm->i_max));
        PetscMalloc(n_elmt*sizeof(int), &(ibm->j_max));
        PetscMalloc(n_elmt*sizeof(int), &(ibm->k_max));

		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhx));
                PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhy));
                PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhz));

      }


      			if (rotor_model == 2) {
        			PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_attack));
        			PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_pitch));
        			PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->chord_blade));

                        	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CD));
                        	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CL));


      			}


      for (i=0; i<n_elmt; i++) {

	fscanf(fd, "%i %i %s %i %i %i\n", &ii,&ii, &ss, nv1+i, nv2+i, nv3+i);
	nv1[i] = nv1[i] - 1; nv2[i] = nv2[i]-1; nv3[i] = nv3[i] - 1;
	      
      }
      ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;

      i=0;
      PetscPrintf(PETSC_COMM_WORLD, "nv %d %d %d\n", nv1[i], nv2[i], nv3[i]);

      fclose(fd);
    }
      
    for (i=0; i<n_elmt; i++) {
      
      n1e = nv1[i]; n2e =nv2[i]; n3e = nv3[i];
      dx12 = x_bp[n2e] - x_bp[n1e];
      dy12 = y_bp[n2e] - y_bp[n1e];
      dz12 = z_bp[n2e] - z_bp[n1e];
      
      dx13 = x_bp[n3e] - x_bp[n1e];
      dy13 = y_bp[n3e] - y_bp[n1e];
      dz13 = z_bp[n3e] - z_bp[n1e];
      
      nf_x[i] = dy12 * dz13 - dz12 * dy13;
      nf_y[i] = -dx12 * dz13 + dz12 * dx13;
      nf_z[i] = dx12 * dy13 - dy12 * dx13;
      
      dr = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i] + nf_z[i]*nf_z[i]);
      
      nf_x[i] /=dr; nf_y[i]/=dr; nf_z[i]/=dr;
      
      // Temp sol. 2D
//       if (fabs(nf_x[i])<.5) 
// 	nf_x[i]=0.; 

      // Addedd 4/2/06 iman
      if ((((1.-nf_z[i])<=1e-6 )&((-1.+nf_z[i])<1e-6))|
	  (((nf_z[i]+1.)<=1e-6 )&((-1.-nf_z[i])<1e-6))) {
	ns_x[i] = 1.;     
	ns_y[i] = 0.;     
	ns_z[i] = 0. ;
	
	nt_x[i] = 0.;
	nt_y[i] = 1.;
	nt_z[i] = 0.;
      } else {
	ns_x[i] =  nf_y[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);      
	ns_y[i] = -nf_x[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);     
	ns_z[i] = 0. ;
	
	nt_x[i] = -nf_x[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
	nt_y[i] = -nf_y[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
	nt_z[i] = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
      }
      
      //Added 4/1/06 iman
      dA[i] = dr/2.; 
      
      // Added 6/4/06 iman
      // Calc the center of the element
      ibm->cent_x[i]= (x_bp[n1e]+x_bp[n2e]+x_bp[n3e])/3.;
      ibm->cent_y[i]= (y_bp[n1e]+y_bp[n2e]+y_bp[n3e])/3.;
      ibm->cent_z[i]= (z_bp[n1e]+z_bp[n2e]+z_bp[n3e])/3.;	
    }
    
    
    ibm->nf_x = nf_x; ibm->nf_y = nf_y;  ibm->nf_z = nf_z;
    
    //Added 4/1/06 iman
    ibm->dA = dA;
    ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
    ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    
    
    MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    
    MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
    // Added 4/1/06 iman
    MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    // Added 4/2/06 iman
    MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    // Added 6/4/06 iman
    MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

 //    PetscFree(dA); 
//     PetscFree(nf_x);PetscFree(nf_y);PetscFree(nf_z); 
//     PetscFree(nt_x);PetscFree(nt_y);PetscFree(nt_z); 
//     PetscFree(ns_x);PetscFree(ns_y);PetscFree(ns_z); 
//     PetscFree(nv1);PetscFree(nv2);PetscFree(nv3);
//     PetscFree(x_bp);PetscFree(y_bp);PetscFree(z_bp);
    int ti=0;
    FILE *f;
    //char filen[80];
    sprintf(filen, "%s/ACDsurface%3.3d_%2.2d_nf.dat",path,ti,ibi);
    f = fopen(filen, "w");
    PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z,n_x,n_y,n_z,nt_x,nt_y,nt_z,ns_x,ns_y,ns_z\n");
    PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-12]=CELLCENTERED)\n", n_v, n_elmt);
    for (i=0; i<n_v; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->x_bp[i]);
    }
    for (i=0; i<n_v; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->y_bp[i]);
    }
    for (i=0; i<n_v; i++) {	
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->z_bp[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_x[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_y[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_z[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_x[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_y[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_z[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_x[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_y[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_z[i]);
    }
    for (i=0; i<n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
    }
    
//     PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z\n");
//     PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEPOINT, ET=TRIANGLE\n", n_v, n_elmt); 
//     for (i=0; i<n_v; i++) { 
      
//       PetscFPrintf(PETSC_COMM_WORLD, f, "%e %e %e\n", ibm->x_bp[i], ibm->y_bp[i], ibm->z_bp[i]); 
//     } 
//     for (i=0; i<n_elmt; i++) { 
//       PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1); 
//     } 
    fclose(f);

  }
  else if (rank) {
    MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    ibm->n_v = n_v;
    
//    PetscMalloc(n_v*sizeof(PetscReal), &x_bp);	// removed by seokkoo 03.04.2009
//    PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
//    PetscMalloc(n_v*sizeof(PetscReal), &z_bp);
    
	
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
	  
	x_bp = ibm->x_bp;	// added by seokkoo 03.04.2009
	y_bp = ibm->y_bp;
	z_bp = ibm->z_bp;
	  
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_i));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_i));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_i));
    
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));
    
 
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
    
    
    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
    PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

    for (i=0; i<n_v; i++) {
      ibm->u[i].x = 0.;
      ibm->u[i].y = 0.;
      ibm->u[i].z = 0.;

      ibm->uold[i].x = 0.;
      ibm->uold[i].y = 0.;
      ibm->uold[i].z = 0.;
      
      ibm->urm1[i].x = 0.;
      ibm->urm1[i].y = 0.;
      ibm->urm1[i].z = 0.;      
    }

    MPI_Bcast(ibm->x_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->y_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->z_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
        
    MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);    
    MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    

    MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    n_elmt = ibm->n_elmt;

    PetscMalloc(n_elmt*sizeof(int), &nv1);
    PetscMalloc(n_elmt*sizeof(int), &nv2);
    PetscMalloc(n_elmt*sizeof(int), &nv3);

    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);

    //Added 4/1/06 iman
    PetscMalloc(n_elmt*sizeof(PetscReal), &dA);

    //Added 4/2/06 iman
    PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
    PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

    PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
    PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
    PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);

    ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;
    ibm->nf_x = nf_x; ibm->nf_y = nf_y; ibm->nf_z = nf_z;
    
    // Added 4/2/06 iman
    ibm->dA = dA;
    ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
    ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    

    // Added 6/4/06
    //PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->cent));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));

    // added 12-7-2010 xyang
    if (rotor_model) {
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_x));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_y));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_z));

      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_x));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_y));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_z));

        PetscMalloc(n_elmt*sizeof(int), &(ibm->i_min));
        PetscMalloc(n_elmt*sizeof(int), &(ibm->j_min));
        PetscMalloc(n_elmt*sizeof(int), &(ibm->k_min));

        PetscMalloc(n_elmt*sizeof(int), &(ibm->i_max));
        PetscMalloc(n_elmt*sizeof(int), &(ibm->j_max));
        PetscMalloc(n_elmt*sizeof(int), &(ibm->k_max));

		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhx));
          	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhy));
              	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhz));


    }


      		if (rotor_model == 2) {
        		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_attack));
        		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_pitch));
        		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->chord_blade));

                        PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CD));
                        PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CL));

      		}


    // end add

    MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    //Added 4/2/06 iman
    MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

    MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
  }
	PetscPrintf(PETSC_COMM_WORLD, "Read ACD file !\n");


  return(0);
}
*/

PetscErrorCode ACL_read_ucd(IBMNodes *ibm, int ibi, FSInfo *fsi)
{
  	int	rank;
  	int	n_v , n_elmt ;
  	PetscReal	*x_bp , *y_bp , *z_bp ;
  	int	*nv1 , *nv2 , *nv3 ;
  	PetscReal	*nf_x, *nf_y, *nf_z;
  	int	i,ii;
  	int	n1e, n2e, n3e;
  	PetscReal	dx12, dy12, dz12, dx13, dy13, dz13;
  	PetscReal     dr;
  	//Added 4/1/06 iman
  	PetscReal     *dA ;//area
  	PetscReal	*nt_x, *nt_y, *nt_z;
  	PetscReal	*ns_x, *ns_y, *ns_z;

	int nv_blade, nelmt_blade;

  	char   ss[20];
  	//double xt;
  	char string[128];

  	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  	if(!rank) { // root processor read in the data
    		FILE *fd;
    		PetscPrintf(PETSC_COMM_SELF, "READ acldata\n");
    		char filen[80];  
    		sprintf(filen,"%s/acldata%3.3d" , path, 0);

    		fd = fopen(filen, "r"); 
    		if (!fd) printf("Cannot open %s !!\n", filen),exit(1);
    		else printf("Opened %s !\n", filen);

    		n_v =0;

    		if (fd) {

      			fscanf(fd, "%i ",&n_v);
      			n_elmt = n_v - 1;
      			PetscPrintf(PETSC_COMM_SELF, "number of nodes & elements & blades %d %d %d \n",n_v, n_elmt, num_blade);
        		nv_blade=n_v;
			nelmt_blade=n_elmt; 

      			ibm->n_v = n_v * num_blade;
      			ibm->n_elmt = n_elmt * num_blade;      
 
      			n_v=ibm->n_v;
      			n_elmt=ibm->n_elmt;      
 

      			MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
	    
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));	// added by seokkoo 03.04.2009
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));

      			x_bp = ibm->x_bp; // seokkoo
      			y_bp = ibm->y_bp;
      			z_bp = ibm->z_bp;

        		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhx));
        		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhy));
        		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhz));

      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_i));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_i));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_i));
      
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
      
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

			int nb,j;
      			for (nb=0; nb<num_blade; nb++) {
        			if (nb != 0) fscanf(fd, "%i ", &ii);
        			for (j=0; j<nv_blade; j++) {
          				i = nb*nv_blade + j;
          
	  				fscanf(fd, "%le %le %le", &x_bp[i], &y_bp[i], &z_bp[i]);//, &t, &t, &t);
				
					x_bp[i]=x_bp[i]/reflength_wt;
					y_bp[i]=y_bp[i]/reflength_wt;
					z_bp[i]=z_bp[i]/reflength_wt;


					// rotate the axis. Assume the one from gridgen is (0,0,1)
	

					// not work for some cases .................
					/*

					Cmpnts p, r, q;
					p.x=x_bp[i]; p.y=y_bp[i]; p.z=z_bp[i];
					r.x=-fsi->ny_tb; r.y=fsi->nx_tb; r.z=0.0;
					double theta=acos(fsi->nz_tb);
					q=ArbitraryRotate(p,theta,r);
					x_bp[i]=q.x;	
					y_bp[i]=q.y;	
					z_bp[i]=q.z;	
					*/
					// ....................
	
					//double n_rot[3];
					//n_rot[0]=-fsi->ny_tb; n_rot[1]=fsi->nx_tb; n_rot[2]=0.0;
					//double ux=n_rot[0], uy=n_rot[1], uz=n_rot[2];

					//double ang_rot;
					//double rrr=fsi->nz_tb;
					//ang_rot=acos(rrr);
					//double cos_=cos(ang_rot), sin_=sin(ang_rot);

					/*
					double mat_rot[3][3];

					mat_rot[0][0]=cos_+pow(ux,2)*(1.0-cos_);
					mat_rot[0][1]=ux*uy*(1.0-cos_)-uz*sin_;
					mat_rot[0][2]=ux*uz*(1.0-cos_)+uy*sin_;

					mat_rot[1][0]=uy*ux*(1.0-cos_)+uz*sin_;
					mat_rot[1][1]=cos_+pow(uy,2)*(1.0-cos_);
					mat_rot[1][2]=uy*uz*(1.0-cos_)-ux*sin_;

					mat_rot[2][0]=uz*ux*(1.0-cos_)-uy*sin_;
					mat_rot[2][1]=uz*uy*(1.0-cos_)+ux*sin_;
					mat_rot[2][2]=cos_+pow(uz,2)*(1.0-cos_);
					*/

					//double xb=x_bp[i], yb=y_bp[i], zb=z_bp[i];

					/*
					x_bp[i]=mat_rot[0][0]*xb+mat_rot[0][1]*yb+mat_rot[0][2]*zb;
					y_bp[i]=mat_rot[1][0]*xb+mat_rot[1][1]*yb+mat_rot[1][2]*zb;
					z_bp[i]=mat_rot[2][0]*xb+mat_rot[2][1]*yb+mat_rot[2][2]*zb;
					*/

					//double wCv[3];
					//wCv[0]=uy*zb-uz*yb; wCv[1]=uz*xb-ux*zb; wCv[2]=ux*yb-uy*xb;
					//double wPv=ux*xb+uy*yb+uz*zb;
					//x_bp[i]=xb*cos_+wCv[0]*sin_+wPv*ux*(1.0-cos_);
					//y_bp[i]=yb*cos_+wCv[1]*sin_+wPv*uy*(1.0-cos_);
					//z_bp[i]=zb*cos_+wCv[2]*sin_+wPv*uz*(1.0-cos_);

	        			ibm->x_bp_i[i] = x_bp[i];
        				ibm->y_bp_i[i] = y_bp[i];
        				ibm->z_bp_i[i] = z_bp[i];

	        			x_bp[i] += fsi->x_c;
        				y_bp[i] += fsi->y_c;
        				z_bp[i] += fsi->z_c;

	        			ibm->x_bp0[i] = x_bp[i];
        				ibm->y_bp0[i] = y_bp[i];
        				ibm->z_bp0[i] = z_bp[i];

	
					ibm->x_bp[i] = x_bp[i];
					ibm->y_bp[i] = y_bp[i];
					ibm->z_bp[i] = z_bp[i];

					ibm->x_bp_o[i] = x_bp[i];
					ibm->y_bp_o[i] = y_bp[i];
					ibm->z_bp_o[i] = z_bp[i];

					ibm->u[i].x = 0.;
					ibm->u[i].y = 0.;
					ibm->u[i].z = 0.;

					ibm->uold[i].x = 0.;
					ibm->uold[i].y = 0.;
					ibm->uold[i].z = 0.;

					ibm->urm1[i].x = 0.;
					ibm->urm1[i].y = 0.;
					ibm->urm1[i].z = 0.;
	      			}
			}
      			i=0;
      			PetscPrintf(PETSC_COMM_WORLD, "xyz_bp %le %le %le\n", x_bp[i], y_bp[i], z_bp[i]);

      			MPI_Bcast(ibm->x_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

     	 		MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      			MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
      			MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      			MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);

      			PetscMalloc(n_elmt*sizeof(int), &nv1);
      			PetscMalloc(n_elmt*sizeof(int), &nv2);
      			PetscMalloc(n_elmt*sizeof(int), &nv3);
      
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);
      
      			// Added 4/1/06 iman
      			PetscMalloc(n_elmt*sizeof(PetscReal), &dA); //Area

      			PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

      			PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);
      
      			// Added 6/4/06 iman
      			//PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->cent));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));
      			// end added


      			// added 12-7-2010 xyang
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_x));
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_y));
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_z));

        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_x));
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_y));
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_z));

	        	PetscMalloc(n_elmt*sizeof(int), &(ibm->i_min));
        		PetscMalloc(n_elmt*sizeof(int), &(ibm->j_min));
        		PetscMalloc(n_elmt*sizeof(int), &(ibm->k_min));

        		PetscMalloc(n_elmt*sizeof(int), &(ibm->i_max));
        		PetscMalloc(n_elmt*sizeof(int), &(ibm->j_max));
        		PetscMalloc(n_elmt*sizeof(int), &(ibm->k_max));

       			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhx));
              		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhy));
              		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhz));

      			if (surface_p_out) {
	        		PetscMalloc(n_elmt*sizeof(int), &(ibm->ib_elmt));
	        		PetscMalloc(n_elmt*sizeof(int), &(ibm->jb_elmt));
        			PetscMalloc(n_elmt*sizeof(int), &(ibm->kb_elmt));

        			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->xb_elmt));
        			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->yb_elmt));
	        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->zb_elmt));

        			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->p_elmt));
        			PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->tau_elmt));
	      		}

        		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_attack));
        		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_pitch));
        		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->chord_blade));

                       	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CD));
                       	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CL));

			PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->shear);
			PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->mean_shear);
			PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress1);
			PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress2);
			PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress3);
			PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->pressure);

      			for (nb=0; nb<num_blade; nb++) {
        		for (j=0; j<nelmt_blade; j++) {
	  			i = nb*nelmt_blade + j;
				ii = nb*nv_blade + j;
	  			nv1[i] = ii; nv2[i] = ii + 1; nv3[i] = ii + 1;
        		}
      			}

	      		ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;

      			i=0;
	      		PetscPrintf(PETSC_COMM_WORLD, "nv %d %d %d\n", ibm->nv1[i], ibm->nv2[i], ibm->nv3[i]);

      			fclose(fd);
	    	}
     
		int nb, j;
		// for ACL, the nf_x, nf_y, nf_z denote the direction of the actuator line. and dA denotes the length of each element.
      		for (nb=0; nb<num_blade; nb++) {
      
    		for (j=0; j<nelmt_blade; j++) {
      			i = nb*nelmt_blade + j;

      			n1e = ibm->nv1[i]; n2e =ibm->nv2[i]; 
      			dx12 = ibm->x_bp[n2e] - ibm->x_bp[n1e];
      			dy12 = ibm->y_bp[n2e] - ibm->y_bp[n1e];
      			dz12 = ibm->z_bp[n2e] - ibm->z_bp[n1e];
      
      
      			nf_x[i] = dx12;
      			nf_y[i] = dy12;
      			nf_z[i] = dz12;
      
      			dr = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i] + nf_z[i]*nf_z[i]);
      
      			nf_x[i] /=dr; nf_y[i]/=dr; nf_z[i]/=dr;
      
			ns_x[i] = 0.;     
			ns_y[i] = 0.;     
			ns_z[i] = 0. ;
	
			nt_x[i] = 0.;
			nt_y[i] = 0.;
			nt_z[i] = 0.;
     
			dA[i] = dr;

      			// Calc the center of the element
      			ibm->cent_x[i]= (ibm->x_bp[n1e]+ibm->x_bp[n2e])/2.;
      			ibm->cent_y[i]= (ibm->y_bp[n1e]+ibm->y_bp[n2e])/2.;
      			ibm->cent_z[i]= (ibm->z_bp[n1e]+ibm->z_bp[n2e])/2.;
	
      		}
      
    		}
     
    		ibm->nf_x = nf_x; ibm->nf_y = nf_y;  ibm->nf_z = nf_z;
    
	    	//Added 4/1/06 iman
	    	ibm->dA = dA;
	    	ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
	    	ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    
    
    
	    	MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    
	    	MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
	    	// Added 4/1/06 iman
	    	MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
	    	// Added 4/2/06 iman
	    	MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
	    	MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
	    	// Added 6/4/06 iman
	    	MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

	 	/*    PetscFree(dA); */
		/*     PetscFree(nf_x);PetscFree(nf_y);PetscFree(nf_z); */
		/*     PetscFree(nt_x);PetscFree(nt_y);PetscFree(nt_z); */
		/*     PetscFree(ns_x);PetscFree(ns_y);PetscFree(ns_z); */
		/*     PetscFree(nv1);PetscFree(nv2);PetscFree(nv3); */
		/*     PetscFree(x_bp);PetscFree(y_bp);PetscFree(z_bp); */


    		int ti=0;
    		FILE *f;
    		sprintf(filen, "%s/line%3.3d_%2.2d_nf.dat",path,ti,ibi);
    		f = fopen(filen, "w");
    		for (i=0; i<ibm->n_elmt; i++) {
      			PetscFPrintf(PETSC_COMM_WORLD, f, "%e %le %le \n", ibm->cent_x[i], ibm->cent_y[i], ibm->cent_z[i]);
    		}
    		fclose(f);


  	}
  	else if (rank) {
    		MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    		ibm->n_v = n_v;
	
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
	  
		x_bp = ibm->x_bp;	// added by seokkoo 03.04.2009
		y_bp = ibm->y_bp;
		z_bp = ibm->z_bp;

        	PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhx));
        	PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhy));
        	PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhz));
	  
      		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_i));
      		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_i));
      		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_i));
 
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
    
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));
    
    		PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
    		PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
    		PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

    		for (i=0; i<n_v; i++) {
      			ibm->u[i].x = 0.;
      			ibm->u[i].y = 0.;
      			ibm->u[i].z = 0.;

      			ibm->uold[i].x = 0.;
      			ibm->uold[i].y = 0.;
      			ibm->uold[i].z = 0.;
      
      			ibm->urm1[i].x = 0.;
      			ibm->urm1[i].y = 0.;
      			ibm->urm1[i].z = 0.;      
    		}
 
      		MPI_Bcast(ibm->x_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      		MPI_Bcast(ibm->y_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      		MPI_Bcast(ibm->z_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
       
    		MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);    
    		MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    		MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    		MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    		MPI_Bcast(&(n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    		ibm->n_elmt = n_elmt;

    		PetscMalloc(n_elmt*sizeof(int), &nv1);
    		PetscMalloc(n_elmt*sizeof(int), &nv2);
    		PetscMalloc(n_elmt*sizeof(int), &nv3);

    		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
    		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
    		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);

    		//Added 4/1/06 iman
    		PetscMalloc(n_elmt*sizeof(PetscReal), &dA);

    		//Added 4/2/06 iman
    		PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
    		PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
    		PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

    		PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
    		PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
    		PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);

    		ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;
    		ibm->nf_x = nf_x; ibm->nf_y = nf_y; ibm->nf_z = nf_z;
    
    		// Added 4/2/06 iman
    		ibm->dA = dA;
    		ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
    		ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    

    		// Added 6/4/06
    		//PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->cent));
    		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
    		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
    		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));


		// added 12-7-2010 xyang
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_x));
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_y));
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_z));

      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_x));
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_y));
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_z));

      		PetscMalloc(n_elmt*sizeof(int), &(ibm->i_min));
      		PetscMalloc(n_elmt*sizeof(int), &(ibm->j_min));
      		PetscMalloc(n_elmt*sizeof(int), &(ibm->k_min));

      		PetscMalloc(n_elmt*sizeof(int), &(ibm->i_max));
      		PetscMalloc(n_elmt*sizeof(int), &(ibm->j_max));
      		PetscMalloc(n_elmt*sizeof(int), &(ibm->k_max));

       		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhx));
              	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhy));
              	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhz));
                         

    		if (surface_p_out) {
 		 	PetscMalloc(n_elmt*sizeof(int), &(ibm->ib_elmt));
      			PetscMalloc(n_elmt*sizeof(int), &(ibm->jb_elmt));
      			PetscMalloc(n_elmt*sizeof(int), &(ibm->kb_elmt));

      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->xb_elmt));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->yb_elmt));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->zb_elmt));

      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->p_elmt));
      			PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->tau_elmt));
    		}


        	PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_attack));
       		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->angle_pitch));
       		PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->chord_blade));

                PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CD));
                PetscMalloc(ibm->n_elmt*sizeof(PetscReal), &(ibm->CL));

		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->shear);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->mean_shear);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress1);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress2);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress3);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->pressure);

    		MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

    		MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    		//Added 4/2/06 iman
    		MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

    		MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    		MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

    		MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
  	}
	PetscPrintf(PETSC_COMM_WORLD, "Finish Reading IBDelta file !\n");
  	return(0);
}

/*

PetscErrorCode ibmDelta_read_ucd(IBMNodes *ibm, int ibi, FSInfo *fsi, int iname)
{
  	int	rank;
  	int	n_v , n_elmt ;
  	PetscReal	*x_bp , *y_bp , *z_bp ;
  	int	*nv1 , *nv2 , *nv3 ;
  	PetscReal	*nf_x, *nf_y, *nf_z;
  	int	i,ii;
  	int	n1e, n2e, n3e;
  	PetscReal	dx12, dy12, dz12, dx13, dy13, dz13;
  	PetscReal     dr;
  	//Added 4/1/06 iman
  	PetscReal     *dA ;//area
  	PetscReal	*nt_x, *nt_y, *nt_z;
  	PetscReal	*ns_x, *ns_y, *ns_z;

  	char   ss[20];
  	//double xt;
  	char string[128];

  	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  	if(!rank) { // root processor read in the data
    		FILE *fd;
    		PetscPrintf(PETSC_COMM_SELF, "READ ibmdata\n");
    		char filen[80];  
    		sprintf(filen,"%s/ibmDelta%3.3d" , path, iname);

    		fd = fopen(filen, "r"); 
    		if (!fd) printf("Cannot open %s !!\n", filen),exit(0);
    		else printf("Opened %s !\n", filen);

    		n_v =0;

    		if (fd) {
      			fgets(string, 128, fd);
      			fgets(string, 128, fd);
      			fgets(string, 128, fd);
      
      			fscanf(fd, "%i %i %i %i %i",&n_v,&n_elmt,&ii,&ii,&ii);
      			PetscPrintf(PETSC_COMM_SELF, "number of nodes & elements %d %d\n",n_v, n_elmt);
      
      			ibm->n_v = n_v;
      			ibm->n_elmt = n_elmt;      
      
      			MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
	    
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));	// added by seokkoo 03.04.2009
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));

      			x_bp = ibm->x_bp; // seokkoo
      			y_bp = ibm->y_bp;
      			z_bp = ibm->z_bp;

        		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhx));
        		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhy));
        		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhz));

      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_i));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_i));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_i));
      
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
      
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

      			for (i=0; i<n_v; i++) {
				fscanf(fd, "%i %le %le %le", &ii, &x_bp[i], &y_bp[i], &z_bp[i]);//, &t, &t, &t);

				x_bp[i]=x_bp[i]/reflength_IBDelta;
				y_bp[i]=y_bp[i]/reflength_IBDelta;
				z_bp[i]=z_bp[i]/reflength_IBDelta;

        			ibm->x_bp_i[i] = x_bp[i];
        			ibm->y_bp_i[i] = y_bp[i];
        			ibm->z_bp_i[i] = z_bp[i];

        			x_bp[i] += fsi->x_c;
        			y_bp[i] += fsi->y_c;
        			z_bp[i] += fsi->z_c;

        			ibm->x_bp0[i] = x_bp[i];
        			ibm->y_bp0[i] = y_bp[i];
        			ibm->z_bp0[i] = z_bp[i];


				// xiaolei Deactivate
				//	x_bp[i] = x_bp[i]*L_dim + CMx_c;//0.25 ;// 24.;	
				//	y_bp[i] = y_bp[i]*L_dim + CMy_c ;//+ ibi*2.;//5.;//2.;//8.;//6.;//2.   ;// 24.;
				//	z_bp[i] = z_bp[i]*L_dim + CMz_c ;//+ ibi*1.5;//2.;//8.;//15.;//2.   ;// 24.;

	
				ibm->x_bp[i] = x_bp[i];
				ibm->y_bp[i] = y_bp[i];
				ibm->z_bp[i] = z_bp[i];

				ibm->x_bp_o[i] = x_bp[i];
				ibm->y_bp_o[i] = y_bp[i];
				ibm->z_bp_o[i] = z_bp[i];

				ibm->u[i].x = 0.;
				ibm->u[i].y = 0.;
				ibm->u[i].z = 0.;

				ibm->uold[i].x = 0.;
				ibm->uold[i].y = 0.;
				ibm->uold[i].z = 0.;

				ibm->urm1[i].x = 0.;
				ibm->urm1[i].y = 0.;
				ibm->urm1[i].z = 0.;
      			}
      			i=0;
      			PetscPrintf(PETSC_COMM_WORLD, "xyz_bp %le %le %le\n", x_bp[i], y_bp[i], z_bp[i]);

      			MPI_Bcast(ibm->x_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

     	 		MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      			MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
      			MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      			MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);

      			PetscMalloc(n_elmt*sizeof(int), &nv1);
      			PetscMalloc(n_elmt*sizeof(int), &nv2);
      			PetscMalloc(n_elmt*sizeof(int), &nv3);
      
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);
      
      			// Added 4/1/06 iman
      			PetscMalloc(n_elmt*sizeof(PetscReal), &dA); //Area

      			PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

      			PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
      			PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);
      
      			// Added 6/4/06 iman
      			//PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->cent));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));
      			// end added


      			// added 12-7-2010 xyang
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_x));
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_y));
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_z));

        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_x));
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_y));
        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_z));

	        	PetscMalloc(n_elmt*sizeof(int), &(ibm->i_min));
        		PetscMalloc(n_elmt*sizeof(int), &(ibm->j_min));
        		PetscMalloc(n_elmt*sizeof(int), &(ibm->k_min));

        		PetscMalloc(n_elmt*sizeof(int), &(ibm->i_max));
        		PetscMalloc(n_elmt*sizeof(int), &(ibm->j_max));
        		PetscMalloc(n_elmt*sizeof(int), &(ibm->k_max));

       			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhx));
              		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhy));
              		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhz));

      			if (surface_p_out) {
	        		PetscMalloc(n_elmt*sizeof(int), &(ibm->ib_elmt));
	        		PetscMalloc(n_elmt*sizeof(int), &(ibm->jb_elmt));
        			PetscMalloc(n_elmt*sizeof(int), &(ibm->kb_elmt));

        			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->xb_elmt));
        			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->yb_elmt));
	        		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->zb_elmt));

        			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->p_elmt));
        			PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->tau_elmt));
	      		}

			PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->shear);
			PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->mean_shear);
			PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress1);
			PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress2);
			PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress3);
			PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->pressure);

	      		for (i=0; i<n_elmt; i++) {
				fscanf(fd, "%i %i %s %i %i %i\n", &ii,&ii, &ss, nv1+i, nv2+i, nv3+i);
				nv1[i] = nv1[i] - 1; nv2[i] = nv2[i]-1; nv3[i] = nv3[i] - 1;
      			}
	      		ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;

      			i=0;
	      		PetscPrintf(PETSC_COMM_WORLD, "nv %d %d %d\n", nv1[i], nv2[i], nv3[i]);

      			fclose(fd);
	    	}
      
	    	for (i=0; i<n_elmt; i++) {
      
      			n1e = nv1[i]; n2e =nv2[i]; n3e = nv3[i];
	      		dx12 = x_bp[n2e] - x_bp[n1e];
	      		dy12 = y_bp[n2e] - y_bp[n1e];
	      		dz12 = z_bp[n2e] - z_bp[n1e];
      
      			dx13 = x_bp[n3e] - x_bp[n1e];
      			dy13 = y_bp[n3e] - y_bp[n1e];
	      		dz13 = z_bp[n3e] - z_bp[n1e];
      
	      		nf_x[i] = dy12 * dz13 - dz12 * dy13;
	      		nf_y[i] = -dx12 * dz13 + dz12 * dx13;
      			nf_z[i] = dx12 * dy13 - dy12 * dx13;
      
      			dr = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i] + nf_z[i]*nf_z[i]);
      
	      		nf_x[i] /=dr; nf_y[i]/=dr; nf_z[i]/=dr;
      

      			// Addedd 4/2/06 iman
	      		if ((((1.-nf_z[i])<=1e-6 )&((-1.+nf_z[i])<1e-6))|
		  		(((nf_z[i]+1.)<=1e-6 )&((-1.-nf_z[i])<1e-6))) {
				ns_x[i] = 1.;     
				ns_y[i] = 0.;     
				ns_z[i] = 0. ;
	
				nt_x[i] = 0.;
				nt_y[i] = 1.;
				nt_z[i] = 0.;
	      		} else {
				ns_x[i] =  nf_y[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);      
				ns_y[i] = -nf_x[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);     
				ns_z[i] = 0. ;
	
				nt_x[i] = -nf_x[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
				nt_y[i] = -nf_y[i]*nf_z[i]/ sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
				nt_z[i] = sqrt(nf_x[i]*nf_x[i] + nf_y[i]*nf_y[i]);
	      		}
      
      			//Added 4/1/06 iman
	      		dA[i] = dr/2.; 
     
 
	      		// Added 6/4/06 iman
      			// Calc the center of the element
	      		ibm->cent_x[i]= (x_bp[n1e]+x_bp[n2e]+x_bp[n3e])/3.;
	      		ibm->cent_y[i]= (y_bp[n1e]+y_bp[n2e]+y_bp[n3e])/3.;
	      		ibm->cent_z[i]= (z_bp[n1e]+z_bp[n2e]+z_bp[n3e])/3.;	
	    	}
    
    
    		ibm->nf_x = nf_x; ibm->nf_y = nf_y;  ibm->nf_z = nf_z;
    
	    	//Added 4/1/06 iman
	    	ibm->dA = dA;
	    	ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
	    	ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    
    
	    	MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    
	    	MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
	    	// Added 4/1/06 iman
	    	MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
	    	// Added 4/2/06 iman
	    	MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
	    	MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
	    	// Added 6/4/06 iman
	    	MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
	    	MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

	 	//    PetscFree(dA); 
		//     PetscFree(nf_x);PetscFree(nf_y);PetscFree(nf_z); 
		//     PetscFree(nt_x);PetscFree(nt_y);PetscFree(nt_z); 
		//     PetscFree(ns_x);PetscFree(ns_y);PetscFree(ns_z); 
		//     PetscFree(nv1);PetscFree(nv2);PetscFree(nv3); 
		//     PetscFree(x_bp);PetscFree(y_bp);PetscFree(z_bp); 
	    	int ti=0;
	    	FILE *f;
	    	//char filen[80];
	    	sprintf(filen, "%s/IB_Delta%3.3d_%2.2d_nf.dat",path,ti,ibi);
	    	f = fopen(filen, "w");
	    	PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z,n_x,n_y,n_z,nt_x,nt_y,nt_z,ns_x,ns_y,ns_z\n");
	    	PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-12]=CELLCENTERED)\n", n_v, n_elmt);
	    	for (i=0; i<n_v; i++) {
	      		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->x_bp[i]);
	    	}
	    	for (i=0; i<n_v; i++) {
	      		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->y_bp[i]);
	    	}
	    	for (i=0; i<n_v; i++) {	
	      		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->z_bp[i]);
	    	}
	    	for (i=0; i<n_elmt; i++) {
      			PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_x[i]);
	    	}
    		for (i=0; i<n_elmt; i++) {
	      		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_y[i]);
    		}
	    	for (i=0; i<n_elmt; i++) {
      			PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_z[i]);
	    	}
	    	for (i=0; i<n_elmt; i++) {
	      		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_x[i]);
	    	}
	    	for (i=0; i<n_elmt; i++) {
	      		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_y[i]);
	    	}
	    	for (i=0; i<n_elmt; i++) {
      			PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_z[i]);
	    	}
	    	for (i=0; i<n_elmt; i++) {
	      		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_x[i]);
	    	}
	    	for (i=0; i<n_elmt; i++) {
	      		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_y[i]);
	    	}
	    	for (i=0; i<n_elmt; i++) {
	      		PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_z[i]);
	    	}
	    	for (i=0; i<n_elmt; i++) {
	      		PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
	    	}
    
	//	     PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z\n"); 
		//     PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEPOINT, ET=TRIANGLE\n", n_v, n_elmt); 
		//     for (i=0; i<n_v; i++) { 
      
		//       PetscFPrintf(PETSC_COMM_WORLD, f, "%e %e %e\n", ibm->x_bp[i], ibm->y_bp[i], ibm->z_bp[i]); 
		//     } 
		//     for (i=0; i<n_elmt; i++) { 
		//       PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1); 
		//     } 
	    	fclose(f);
  	}
  	else if (rank) {
    		MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    		ibm->n_v = n_v;
	
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
	  
		x_bp = ibm->x_bp;	// added by seokkoo 03.04.2009
		y_bp = ibm->y_bp;
		z_bp = ibm->z_bp;

        	PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhx));
        	PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhy));
        	PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dhz));
	  
      		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_i));
      		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_i));
      		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_i));
 
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
    
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
    		PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));
    
    		PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
    		PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
    		PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

    		for (i=0; i<n_v; i++) {
      			ibm->u[i].x = 0.;
      			ibm->u[i].y = 0.;
      			ibm->u[i].z = 0.;

      			ibm->uold[i].x = 0.;
      			ibm->uold[i].y = 0.;
      			ibm->uold[i].z = 0.;
      
      			ibm->urm1[i].x = 0.;
      			ibm->urm1[i].y = 0.;
      			ibm->urm1[i].z = 0.;      
    		}
 
      		MPI_Bcast(ibm->x_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      		MPI_Bcast(ibm->y_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      		MPI_Bcast(ibm->z_bp_i, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
       
    		MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);    
    		MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    		MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

    		MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    		MPI_Bcast(&(n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    		ibm->n_elmt = n_elmt;

    		PetscMalloc(n_elmt*sizeof(int), &nv1);
    		PetscMalloc(n_elmt*sizeof(int), &nv2);
    		PetscMalloc(n_elmt*sizeof(int), &nv3);

    		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_x);
    		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_y);
    		PetscMalloc(n_elmt*sizeof(PetscReal), &nf_z);

    		//Added 4/1/06 iman
    		PetscMalloc(n_elmt*sizeof(PetscReal), &dA);

    		//Added 4/2/06 iman
    		PetscMalloc(n_elmt*sizeof(PetscReal), &nt_x);
    		PetscMalloc(n_elmt*sizeof(PetscReal), &nt_y);
    		PetscMalloc(n_elmt*sizeof(PetscReal), &nt_z);

    		PetscMalloc(n_elmt*sizeof(PetscReal), &ns_x);
    		PetscMalloc(n_elmt*sizeof(PetscReal), &ns_y);
    		PetscMalloc(n_elmt*sizeof(PetscReal), &ns_z);

    		ibm->nv1 = nv1; ibm->nv2 = nv2; ibm->nv3 = nv3;
    		ibm->nf_x = nf_x; ibm->nf_y = nf_y; ibm->nf_z = nf_z;
    
    		// Added 4/2/06 iman
    		ibm->dA = dA;
    		ibm->nt_x = nt_x; ibm->nt_y = nt_y;  ibm->nt_z = nt_z;
    		ibm->ns_x = ns_x; ibm->ns_y = ns_y;  ibm->ns_z = ns_z;    

    		// Added 6/4/06
    		//PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->cent));
    		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_x));
    		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_y));
    		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z));


		// added 12-7-2010 xyang
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_x));
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_y));
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_lagr_z));

      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_x));
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_y));
      		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->U_lagr_z));

      		PetscMalloc(n_elmt*sizeof(int), &(ibm->i_min));
      		PetscMalloc(n_elmt*sizeof(int), &(ibm->j_min));
      		PetscMalloc(n_elmt*sizeof(int), &(ibm->k_min));

      		PetscMalloc(n_elmt*sizeof(int), &(ibm->i_max));
      		PetscMalloc(n_elmt*sizeof(int), &(ibm->j_max));
      		PetscMalloc(n_elmt*sizeof(int), &(ibm->k_max));

       		PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhx));
              	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhy));
              	PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->dhz));
                         

    		if (surface_p_out) {
 		 	PetscMalloc(n_elmt*sizeof(int), &(ibm->ib_elmt));
      			PetscMalloc(n_elmt*sizeof(int), &(ibm->jb_elmt));
      			PetscMalloc(n_elmt*sizeof(int), &(ibm->kb_elmt));

      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->xb_elmt));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->yb_elmt));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->zb_elmt));

      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->p_elmt));
      			PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->tau_elmt));
    		}

		//seokkoo
		//seokkoo
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->shear);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->mean_shear);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress1);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress2);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress3);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->pressure);

    		MPI_Bcast(ibm->nv1, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nv2, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nv3, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

    		MPI_Bcast(ibm->nf_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nf_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nf_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    		//Added 4/2/06 iman
    		MPI_Bcast(ibm->dA, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

    		MPI_Bcast(ibm->nt_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nt_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->nt_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    
    		MPI_Bcast(ibm->ns_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->ns_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->ns_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);

    		MPI_Bcast(ibm->cent_x, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->cent_y, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    		MPI_Bcast(ibm->cent_z, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
  	}
	PetscPrintf(PETSC_COMM_WORLD, "Finish Reading IBDelta file !\n");
  	return(0);
}

*/

// Call in main.c, after reading the grid for turbines
PetscErrorCode Pre_process(UserCtx *user, IBMNodes *ibm, int NumberOfObjects)
{
  	DM              da = user->da, fda = user->fda;
  	DMDALocalInfo  	info;
  	PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  	PetscInt        mx, my, mz; // Dimensions in three directions
  	PetscInt        i, j, k, l, ibi;
  	PetscInt	lxs, lxe, lys, lye, lzs, lze;

  	Cmpnts		***coor, ***csi, ***eta, ***zet;

  	PetscReal 	***aj;

  	Vec		Coor;


  	DMDAGetLocalInfo(da, &info);
  	mx = info.mx; my = info.my; mz = info.mz;
  	xs = info.xs; xe = xs + info.xm;
  	ys = info.ys; ye = ys + info.ym;
  	zs = info.zs; ze = zs + info.zm;

  	lxs = xs; lxe = xe;
  	lys = ys; lye = ye;
  	lzs = zs; lze = ze;

  	if (xs==0) lxs = xs+1;
  	if (ys==0) lys = ys+1;
  	if (zs==0) lzs = zs+1;

  	if (xe==mx) lxe = xe-1;
  	if (ye==my) lye = ye-1;
  	if (ze==mz) lze = ze-1;


	double *dhx_, *dhy_, *dhz_;
	int *iclose, *jclose, *kclose;

	double *sum_dhx_, *sum_dhy_, *sum_dhz_;
	int *sum_iclose, *sum_jclose, *sum_kclose;


  	DMDAGetGhostedCoordinates(da, &Coor);
  	DMDAVecGetArray(fda, Coor, &coor);

  	DMDAVecGetArray(fda, user->lCsi,  &csi);
  	DMDAVecGetArray(fda, user->lEta,  &eta);
  	DMDAVecGetArray(fda, user->lZet,  &zet);
  	DMDAVecGetArray(da,  user->lAj,  &aj);


  	int n1e, n2e, n3e;

  	for (ibi=0; ibi<NumberOfObjects; ibi++) {

		dhx_= (double *) malloc(ibm[ibi].n_elmt*sizeof(double));
		dhy_= (double *) malloc(ibm[ibi].n_elmt*sizeof(double));
		dhz_= (double *) malloc(ibm[ibi].n_elmt*sizeof(double));

		iclose= (int *) malloc(ibm[ibi].n_elmt*sizeof(int));
		jclose= (int *) malloc(ibm[ibi].n_elmt*sizeof(int));
		kclose= (int *) malloc(ibm[ibi].n_elmt*sizeof(int));

		sum_dhx_= (double *) malloc(ibm[ibi].n_elmt*sizeof(double));
		sum_dhy_= (double *) malloc(ibm[ibi].n_elmt*sizeof(double));
		sum_dhz_= (double *) malloc(ibm[ibi].n_elmt*sizeof(double));

		sum_iclose= (int *) malloc(ibm[ibi].n_elmt*sizeof(int));
		sum_jclose= (int *) malloc(ibm[ibi].n_elmt*sizeof(int));
		sum_kclose= (int *) malloc(ibm[ibi].n_elmt*sizeof(int));

	        for (l=0; l<ibm[ibi].n_elmt; l++) {

	               	dhx_[l] = 0.0;
	                dhy_[l] = 0.0;
	                dhz_[l] = 0.0;

	               	sum_dhx_[l] = 0.0;
        	        sum_dhy_[l] = 0.0;
                	sum_dhz_[l] = 0.0;

			iclose[l]=0;
			jclose[l]=0;
			kclose[l]=0;

			sum_iclose[l]=0;
			sum_jclose[l]=0;
			sum_kclose[l]=0;

		}


	        for (l=0; l<ibm[ibi].n_elmt; l++) {
			int imark=0;
			double dmin=1.e6;
			int ic,jc,kc;
	                for (i=lxs; i<lxe; i++)
        	        for (j=lys; j<lye; j++) 
                	for (k=lzs; k<lze; k++){ 
			
				double r1=ibm[ibi].cent_x[l]-coor[k][j][i].x, r2=ibm[ibi].cent_y[l]-coor[k][j][i].y, r3=ibm[ibi].cent_z[l]-coor[k][j][i].z; 
				double d1=sqrt(r1*r1+r2*r2+r3*r3);
				if (d1<dmin) {
					dmin=d1;
					ic=i; jc=j; kc=k;
				}
                	}
	
			double dmin_global;
			MPI_Allreduce (&dmin, &dmin_global, 1, MPI_DOUBLE, MPI_MIN, PETSC_COMM_WORLD);
			double diff=fabs(dmin-dmin_global);
			if (diff>1.e-6) {
				ic=0; jc=0; kc=0;
				iclose[l]=0; jclose[l]=0; kclose[l]=0;
				dhx_[l]=0.0;
				dhy_[l]=0.0;
				dhz_[l]=0.0;
			} else {

				iclose[l]=ic; jclose[l]=jc; kclose[l]=kc;
				i=ic; j=jc; k=kc;
		                double area = sqrt( csi[k][j][i].x*csi[k][j][i].x + csi[k][j][i].y*csi[k][j][i].y + csi[k][j][i].z*csi[k][j][i].z );
				dhx_[l]=1.0/aj[k][j][i]/area;

			       	area = sqrt( eta[k][j][i].x*eta[k][j][i].x + eta[k][j][i].y*eta[k][j][i].y + eta[k][j][i].z*eta[k][j][i].z );
				dhy_[l]=1.0/aj[k][j][i]/area;

			       	area = sqrt( zet[k][j][i].x*zet[k][j][i].x + zet[k][j][i].y*zet[k][j][i].y + zet[k][j][i].z*zet[k][j][i].z );
				dhz_[l]=1.0/aj[k][j][i]/area;	
			}

	        }



	  	PetscBarrier(PETSC_NULL);

		MPI_Allreduce (&dhx_[0], &sum_dhx_[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&dhy_[0], &sum_dhy_[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&dhz_[0], &sum_dhz_[0], ibm[ibi].n_elmt, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

		MPI_Allreduce (&iclose[0], &sum_iclose[0], ibm[ibi].n_elmt, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&jclose[0], &sum_jclose[0], ibm[ibi].n_elmt, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce (&kclose[0], &sum_kclose[0], ibm[ibi].n_elmt, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);


	        for (l=0; l<ibm[ibi].n_elmt; l++) {

        	       	ibm[ibi].dhx[l]=sum_dhx_[l];
	               	ibm[ibi].dhy[l]=sum_dhy_[l];
        	       	ibm[ibi].dhz[l]=sum_dhz_[l];

	               	dhx_[l]=sum_dhx_[l];
	               	dhy_[l]=sum_dhy_[l];
	               	dhz_[l]=sum_dhz_[l];

			iclose[l]=sum_iclose[l];
			jclose[l]=sum_jclose[l];
			kclose[l]=sum_kclose[l];

		}


	        int iii1 = 0;
		int iii2 = 0;
	        int iii = 0;

	        for (l=0; l<ibm[ibi].n_elmt; l++) {

                        int n1e = ibm[ibi].nv1[i], n2e = ibm[ibi].nv2[i], n3e = ibm[ibi].nv3[i];

			/*
			double d12=sqrt(pow(ibm[ibi].x_bp[n1e]-ibm[ibi].x_bp[n2e],2)+pow(ibm[ibi].y_bp[n1e]-ibm[ibi].y_bp[n2e],2)+pow(ibm[ibi].z_bp[n1e]-ibm[ibi].z_bp[n2e],2));
			double d23=sqrt(pow(ibm[ibi].x_bp[n3e]-ibm[ibi].x_bp[n2e],2)+pow(ibm[ibi].y_bp[n3e]-ibm[ibi].y_bp[n2e],2)+pow(ibm[ibi].z_bp[n3e]-ibm[ibi].z_bp[n2e],2));
			double d31=sqrt(pow(ibm[ibi].x_bp[n1e]-ibm[ibi].x_bp[n3e],2)+pow(ibm[ibi].y_bp[n1e]-ibm[ibi].y_bp[n3e],2)+pow(ibm[ibi].z_bp[n1e]-ibm[ibi].z_bp[n3e],2));

			double d_max=max(d12,d23);
			d_max=max(d_max,d31);
			
			Itpwidth=(int)d_max/ibm[ibi].dhx[l]+4;
			*/

			int ic=iclose[l];
			int jc=jclose[l];
			int kc=kclose[l];

			int ic1=ic-Itpwidth, ic2=ic+Itpwidth; 
		
			if (ic1>lxe||ic2<lxs) {
				ibm[ibi].i_min[l]=lxs;
				ibm[ibi].i_max[l]=lxs;
			} else {
				ibm[ibi].i_min[l]=PetscMax(ic1, lxs);
				ibm[ibi].i_max[l]=PetscMin(ic2, lxe);
			}

			// Itpwidth=(int)d_max/ibm[ibi].dhy[l]+4;
			int jc1=jc-Itpwidth, jc2=jc+Itpwidth; 

	       		if (jc1>lye||jc2<lys) {
				ibm[ibi].j_min[l]=lys;
				ibm[ibi].j_max[l]=lys;
			} else {
				ibm[ibi].j_min[l]=PetscMax(jc1, lys);
				ibm[ibi].j_max[l]=PetscMin(jc2, lye);
			}

			// Itpwidth=(int)d_max/ibm[ibi].dhz[l]+4;
			int kc1=kc-Itpwidth, kc2=kc+Itpwidth; 

	       		if (kc1>lze||kc2<lzs) {
				ibm[ibi].k_min[l]=lzs;
				ibm[ibi].k_max[l]=lzs;
			} else {
				ibm[ibi].k_min[l]=PetscMax(kc1, lzs);
				ibm[ibi].k_max[l]=PetscMin(kc2, lze);
			}


		}

/*
	        for (l=0; l<ibm[ibi].n_elmt; l++) {

			ibm[ibi].i_min[l]=lxs;
			ibm[ibi].i_max[l]=lxe;

			ibm[ibi].j_min[l]=lys;
			ibm[ibi].j_max[l]=lye;


			ibm[ibi].k_min[l]=lzs;
			ibm[ibi].k_max[l]=lze;


		}	
*/

		/*
		if (rotor_model == 2) {

		int imin_g=1000000, imax_g=-1000000, jmin_g=1000000, jmax_g=-1000000, kmin_g=1000000, kmax_g=-1000000;

	        for (l=0; l<ibm[ibi].n_elmt; l++) {
			if (imin_g>ibm[ibi].i_min[l]) imin_g=ibm[ibi].i_min[l];	
			if (imax_g<ibm[ibi].i_max[l]) imax_g=ibm[ibi].i_max[l];	

			if (jmin_g>ibm[ibi].j_min[l]) jmin_g=ibm[ibi].j_min[l];	
			if (jmax_g<ibm[ibi].j_max[l]) jmax_g=ibm[ibi].j_max[l];	

			if (kmin_g>ibm[ibi].k_min[l]) kmin_g=ibm[ibi].k_min[l];	
			if (kmax_g<ibm[ibi].k_max[l]) kmax_g=ibm[ibi].k_max[l];	
		}

	        for (l=0; l<ibm[ibi].n_elmt; l++) {

			ibm[ibi].i_min[l]=imin_g;	
			ibm[ibi].j_min[l]=jmin_g;	
			ibm[ibi].k_min[l]=kmin_g;	

			ibm[ibi].i_max[l]=imax_g;	
			ibm[ibi].j_max[l]=jmax_g;	
			ibm[ibi].k_max[l]=kmax_g;	

		}	

		}
		*/

		free(dhx_);
		free(dhy_); 
		free(dhz_);
	        free(iclose);
		free(jclose);
		free(kclose);

		free(sum_dhx_);
		free(sum_dhy_); 
		free(sum_dhz_);
	        free(sum_iclose);
		free(sum_jclose);
		free(sum_kclose);

	}

  	DMDAVecRestoreArray(fda, Coor, &coor);

	DMDAVecRestoreArray(fda, user->lCsi,  &csi);
  	DMDAVecRestoreArray(fda, user->lEta,  &eta);
  	DMDAVecRestoreArray(fda, user->lZet,  &zet);
  	DMDAVecRestoreArray(da,  user->lAj,  &aj);

  	return(0);

}

PetscErrorCode Pre_process_wtm(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int NumberOfObjects)
{
  DM              da = user->da, fda = user->fda;
  DMDALocalInfo     info;
  PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  PetscInt        mx, my, mz; // Dimensions in three directions
  PetscInt        i, j, k, l, ibi;
  PetscInt	lxs, lxe, lys, lye, lzs, lze;
  PetscReal	xb_min, xb_max, yb_min, yb_max, zb_min, zb_max;

  Cmpnts	***coor;

  Vec		Coor;

  double hx, hy, hz;

  double xlag1, xlag2;
  double ylag1, ylag2;
  double zlag1, zlag2;

  double xlag, ylag, zlag;

  PetscReal dhx, dhy, dhz;

  DMDAGetLocalInfo(da, &info);
  mx = info.mx; my = info.my; mz = info.mz;
  xs = info.xs; xe = xs + info.xm;
  ys = info.ys; ye = ys + info.ym;
  zs = info.zs; ze = zs + info.zm;


  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;


  DMDAGetGhostedCoordinates(da, &Coor);
  DMDAVecGetArray(fda, Coor, &coor);
  PetscInt n1e, n2e, n3e;


  for (ibi=0; ibi<NumberOfObjects; ibi++) {


    	xb_min = 1.0e6; xb_max= -1.0e6;        
    	yb_min = 1.0e6; yb_max= -1.0e6;
    	zb_min = 1.0e6; zb_max= -1.0e6;

	xb_min=fsi[ibi].x_c-r_rotor/reflength_wt;;
	xb_max=fsi[ibi].x_c+r_rotor/reflength_wt;;
	yb_min=fsi[ibi].y_c-r_rotor/reflength_wt;;
	yb_max=fsi[ibi].y_c+r_rotor/reflength_wt;;
	zb_min=fsi[ibi].z_c;//-r_rotor/reflength_wt;;
	zb_max=fsi[ibi].z_c;//+r_rotor/reflength_wt;;

	xb_min-=3.0*(coor[lzs][lys][lxs+1].x-coor[lzs][lys][lxs].x);
	xb_max+=3.0*(coor[lzs][lys][lxs+1].x-coor[lzs][lys][lxs].x);
        yb_min-=3.0*(coor[lzs][lys+1][lxs].y-coor[lzs][lys][lxs].y);
        yb_max+=3.0*(coor[lzs][lys+1][lxs].y-coor[lzs][lys][lxs].y);
        zb_min-=3.0*(coor[lzs+1][lys][lxs].z-coor[lzs][lys][lxs].z);
        zb_max+=3.0*(coor[lzs+1][lys][lxs].z-coor[lzs][lys][lxs].z);


        int iii1 = 0;
	int iii2 = 0;
        int iii = 0;

//        for (l=0; l<ibm[ibi].n_elmt; l++) {

		iii1 = 0;
		iii2 = 0;
		iii  = 0;

		j = lys; k =lzs;
                for (i=lxs; i<lxe; i++){

                        if ((xb_min-coor[k][j][i-1].x)>-1.e-9 && (xb_min-coor[k][j][i].x)<1.e-9) { 
				int ii = i;
				if (ii<0) ii=1;	
        			for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].i_min[l] = ii;
                        	iii1 = 1;
                        }
                        if ((xb_max-coor[k][j][i-1].x)>-1.e-9 && (xb_max-coor[k][j][i].x)<1.e-9) {
				int ii=i;
				if (ii>mx-1) ii=mx-1;
                                for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].i_max[l] = ii;
                                iii2 = 1;
                        }

                }

                if (iii1 && !iii2) {
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].i_max[l] = lxe;
                }
                if (!iii1 && iii2) {
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].i_min[l] = lxs;
                }
                if (!iii1 && !iii2) {
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].i_min[l] = lxs;
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].i_max[l] = lxs;
                }


		if ((xb_min-coor[k][j][lxs].x)<1.e-9 &&  (xb_max-coor[k][j][lxe-1].x)>-1.e-9) {
			for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].i_min[l] = lxs;
			for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].i_max[l] = lxe;
		}


		iii1 = 0;
		iii2 = 0;
		iii  = 0;

		i = lxs; k = lzs;
                for (j=lys; j<lye; j++){ 

			if ((yb_min-coor[k][j-1][i].y)>-1.e-9 && (yb_min-coor[k][j][i].y)<1.e-9) { 
				int jj=j;
				if (jj<0) jj=1;
                        	for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].j_min[l] = jj;
                        	iii1 = 1;
                        }
                        if ((yb_max-coor[k][j-1][i].y)>-1.e-9 && (yb_max-coor[k][j][i].y)<1.e-9) {
				int jj=j;
				if (jj>my-1) jj=my-1;
                                for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].j_max[l] = jj;
                                iii2 = 1;
                        }
                }


                if (iii1 && !iii2) {
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].j_max[l] = lye;
                }
                if (!iii1 && iii2) {
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].j_min[l] = lys;
                }
                if (!iii1 && !iii2) {
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].j_min[l] = lys;
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].j_max[l] = lys;
                }

                if ((yb_min-coor[k][lys][i].y)<1.e-9 &&  (yb_max-coor[k][lye-1][i].y)>-1.e-9) {
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].j_min[l] = lys;
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].j_max[l] = lye;
                }

		
		iii1 = 0;
		iii2 = 0;
		iii  = 0;


		i = lxs; j = lys;
                for (k=lzs; k<lze; k++){ 

                        if ((zb_min-coor[k-1][j][i].z)>-1.0e-9 && (zb_min-coor[k][j][i].z)<1.0e-9) { 
				int kk=k;
				if (kk<0) kk=1;
                        	for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].k_min[l] = kk;
                        	iii1 = 1;
                        }
                        if ((zb_max-coor[k-1][j][i].z)>-1.0e-9 && (zb_max-coor[k][j][i].z)<1.0e-9) {
				int kk=k;
				if (kk>mz-1) kk=mz-1;
                                for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].k_max[l] = kk;
                                iii2 = 1;
                        }
                }


                if (iii1 && !iii2) {
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].k_max[l] = lze;
                }
                if (!iii1 && iii2) {
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].k_min[l] = lzs;
                }
                if (!iii1 && !iii2) {
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].k_min[l] = lzs;
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].k_max[l] = lzs;
                }


                if ((zb_min-coor[lzs][j][i].z)<1.0e-9 &&  (zb_max- coor[lze-1][j][i].z)>-1.0e-9) {
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].k_min[l] = lzs;
                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].k_max[l] = lze;
                }

//                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].k_min[l] = lzs;
//                        for (l=0; l<ibm[ibi].n_elmt; l++) ibm[ibi].k_max[l] = lze;


//        }
//      		l = 0;
//              	printf("lxs, lys %d %d \n", lxs, lys);
//                printf("zb_min, zb_max %le %le \n", zb_min, zb_max);
//                printf("coor_z %le %le %le %le %le \n", coor[97][1][26].z, coor[98][1][26].z, coor[99][1][26].z, coor[100][1][26].z, coor[101][1][26].z);

//              	printf("i_min, i_max %d %d \n", ibm[ibi].i_min[l], ibm[ibi].i_max[l]);
//              	printf("j_min, j_max %d %d \n", ibm[ibi].j_min[l], ibm[ibi].j_max[l]);
//              	printf("k_min, k_max %d %d \n", ibm[ibi].k_min[l], ibm[ibi].k_max[l]);


  }


  DMDAVecRestoreArray(fda, Coor, &coor);

  return(0);

}




/* Read information of ACL, like chord, twist angle, lift and drag coefficients */
// call in the main.c for actuator line simulation only
PetscErrorCode airfoil_ACL(ACL *acl, IBMNodes *ibm,  FSInfo *fsi)
{

	int	rank;
  	int	n_CD, n_CL;
	int	ifoil, i;

  	char string[128];

  	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  	if(!rank) { // root processor read in the data
    		FILE *fd;
    		PetscPrintf(PETSC_COMM_SELF, "READ airfoil data for ACL\n");
    		char filen[80]; 

		for(ifoil=0; ifoil<num_foiltype; ifoil++) {

    			sprintf(filen,"%s/FOIL%2.2d" , path, ifoil);
    			fd = fopen(filen, "r"); 
			if (!fd) PetscPrintf(PETSC_COMM_SELF, "Cannot open airfoil data file\n"), exit(1);
			if (fd) {
      				fgets(string, 128, fd);
      				fgets(string, 128, fd);

      				fscanf(fd, "%i ",&(acl[ifoil].num_AC));

      				MPI_Bcast(&(acl[ifoil].num_AC), 1, MPI_INT, 0, PETSC_COMM_WORLD);

                        	PetscMalloc((acl[ifoil].num_AC)*sizeof(PetscReal), &(acl[ifoil].r_AC));  
                        	PetscMalloc((acl[ifoil].num_AC)*sizeof(PetscReal), &(acl[ifoil].angle_pitchInp));  
                        	PetscMalloc((acl[ifoil].num_AC)*sizeof(PetscReal), &(acl[ifoil].chord_bladeInp)); 

			 	for(i=0; i<acl[ifoil].num_AC; i++) {
					fscanf(fd, "%le %le %le", &(acl[ifoil].r_AC[i]), &(acl[ifoil].angle_pitchInp[i]), &(acl[ifoil].chord_bladeInp[i]));

					acl[ifoil].r_AC[i] = acl[ifoil].r_AC[i] / reflength_wt;
					acl[ifoil].chord_bladeInp[i] = acl[ifoil].chord_bladeInp[i] / reflength_wt;

				} 

				acl[ifoil].r_beg = acl[ifoil].r_AC[0];
				acl[ifoil].r_end = acl[ifoil].r_AC[acl[ifoil].num_AC-1];

				MPI_Bcast(acl[ifoil].r_AC, acl[ifoil].num_AC, MPIU_REAL, 0, PETSC_COMM_WORLD); 
				MPI_Bcast(acl[ifoil].angle_pitchInp, acl[ifoil].num_AC, MPIU_REAL, 0, PETSC_COMM_WORLD); 
				MPI_Bcast(acl[ifoil].chord_bladeInp, acl[ifoil].num_AC, MPIU_REAL, 0, PETSC_COMM_WORLD); 
				MPI_Bcast(&(acl[ifoil].r_beg), 1, MPIU_REAL, 0, PETSC_COMM_WORLD); 
				MPI_Bcast(&(acl[ifoil].r_end), 1, MPIU_REAL, 0, PETSC_COMM_WORLD); 

			}

                        sprintf(filen,"%s/CD%2.2d" , path, ifoil);
                        fd = fopen(filen, "r"); 
			if (!fd) PetscPrintf(PETSC_COMM_SELF, "Cannot open CD file\n"), exit(1);
                        if (fd) {
                                fgets(string, 128, fd);
                                fgets(string, 128, fd);

                                fscanf(fd, "%i ",&(acl[ifoil].num_CD));

                                MPI_Bcast(&(acl[ifoil].num_CD), 1, MPI_INT, 0, PETSC_COMM_WORLD);

                                PetscMalloc((acl[ifoil].num_CD)*sizeof(PetscReal), &(acl[ifoil].ang_CD));
                                PetscMalloc((acl[ifoil].num_CD)*sizeof(PetscReal), &(acl[ifoil].CDInp));

                                for(i=0; i<acl[ifoil].num_CD; i++) {
                                        fscanf(fd, "%le %le", &(acl[ifoil].ang_CD[i]), &(acl[ifoil].CDInp[i]));
                                }

                                MPI_Bcast(acl[ifoil].ang_CD, acl[ifoil].num_CD, MPIU_REAL, 0, PETSC_COMM_WORLD);
                                MPI_Bcast(acl[ifoil].CDInp, acl[ifoil].num_CD, MPIU_REAL, 0, PETSC_COMM_WORLD);

                        }

                        sprintf(filen,"%s/CL%2.2d" , path, ifoil);
                        fd = fopen(filen, "r");
			if (!fd) PetscPrintf(PETSC_COMM_SELF, "Cannot open CL file\n"), exit(1);
                        if (fd) {
                                fgets(string, 128, fd);
                                fgets(string, 128, fd);

                                fscanf(fd, "%i ",&(acl[ifoil].num_CL));

                                MPI_Bcast(&(acl[ifoil].num_CL), 1, MPI_INT, 0, PETSC_COMM_WORLD);

                                PetscMalloc((acl[ifoil].num_CL)*sizeof(PetscReal), &(acl[ifoil].ang_CL));
                                PetscMalloc((acl[ifoil].num_CL)*sizeof(PetscReal), &(acl[ifoil].CLInp));

                                for(i=0; i<acl[ifoil].num_CL; i++) {
                                        fscanf(fd, "%le %le", &(acl[ifoil].ang_CL[i]), &(acl[ifoil].CLInp[i]));
                                }

                                MPI_Bcast(acl[ifoil].ang_CL, acl[ifoil].num_CL, MPIU_REAL, 0, PETSC_COMM_WORLD);
                                MPI_Bcast(acl[ifoil].CLInp, acl[ifoil].num_CL, MPIU_REAL, 0, PETSC_COMM_WORLD);

                        }



		}

 
 
	} else if(rank) {


		for(ifoil=0; ifoil<num_foiltype; ifoil++) {

			// angle of pitch , chord length of blade
      			MPI_Bcast(&(acl[ifoil].num_AC), 1, MPI_INT, 0, PETSC_COMM_WORLD);

                        PetscMalloc((acl[ifoil].num_AC)*sizeof(PetscReal), &(acl[ifoil].r_AC));  
                        PetscMalloc((acl[ifoil].num_AC)*sizeof(PetscReal), &(acl[ifoil].angle_pitchInp));  
                        PetscMalloc((acl[ifoil].num_AC)*sizeof(PetscReal), &(acl[ifoil].chord_bladeInp)); 

			
			MPI_Bcast(acl[ifoil].r_AC, acl[ifoil].num_AC, MPIU_REAL, 0, PETSC_COMM_WORLD); 
			MPI_Bcast(acl[ifoil].angle_pitchInp, acl[ifoil].num_AC, MPIU_REAL, 0, PETSC_COMM_WORLD); 
			MPI_Bcast(acl[ifoil].chord_bladeInp, acl[ifoil].num_AC, MPIU_REAL, 0, PETSC_COMM_WORLD); 
			MPI_Bcast(&(acl[ifoil].r_beg), 1, MPIU_REAL, 0, PETSC_COMM_WORLD); 
			MPI_Bcast(&(acl[ifoil].r_end), 1, MPIU_REAL, 0, PETSC_COMM_WORLD); 

			// drag coefficients
                        MPI_Bcast(&(acl[ifoil].num_CD), 1, MPI_INT, 0, PETSC_COMM_WORLD);

                        PetscMalloc((acl[ifoil].num_CD)*sizeof(PetscReal), &(acl[ifoil].ang_CD));
                        PetscMalloc((acl[ifoil].num_CD)*sizeof(PetscReal), &(acl[ifoil].CDInp));

                        MPI_Bcast(acl[ifoil].ang_CD, acl[ifoil].num_CD, MPIU_REAL, 0, PETSC_COMM_WORLD);
                        MPI_Bcast(acl[ifoil].CDInp, acl[ifoil].num_CD, MPIU_REAL, 0, PETSC_COMM_WORLD);

			// left coefficient
                        MPI_Bcast(&(acl[ifoil].num_CL), 1, MPI_INT, 0, PETSC_COMM_WORLD);

                        PetscMalloc((acl[ifoil].num_CL)*sizeof(PetscReal), &(acl[ifoil].ang_CL));
                        PetscMalloc((acl[ifoil].num_CL)*sizeof(PetscReal), &(acl[ifoil].CLInp));

                        MPI_Bcast(acl[ifoil].ang_CL, acl[ifoil].num_CL, MPIU_REAL, 0, PETSC_COMM_WORLD);
                        MPI_Bcast(acl[ifoil].CLInp, acl[ifoil].num_CL, MPIU_REAL, 0, PETSC_COMM_WORLD);

		}

	}

	int ibi, j;
	double r, fac1, fac2;
	for (ibi=0;ibi<NumberOfTurbines;ibi++) {
		for (i=0;i<ibm[ibi].n_elmt;i++) {
			r = sqrt( pow((ibm[ibi].cent_x[i]-fsi[ibi].x_c),2.0) + pow((ibm[ibi].cent_y[i]-fsi[ibi].y_c),2.0) + pow((ibm[ibi].cent_z[i]-fsi[ibi].z_c),2.0));	

//			if (!rank) printf("Turbine %d #### elmt %d cent_x %le x_c %le \n", ibi, i,  ibm[ibi].cent_x[i], fsi[ibi].x_c);
//			if (!rank) printf("Turbine %d #### elmt %d cent_y %le y_c %le \n", ibi, i,  ibm[ibi].cent_y[i], fsi[ibi].y_c);
//			if (!rank) printf("Turbine %d #### elmt %d cent_z %le z_c %le \n", ibi, i,  ibm[ibi].cent_z[i], fsi[ibi].z_c);
			for(ifoil=0; ifoil<num_foiltype; ifoil++) {

				if( r>=acl[ifoil].r_beg && r<=acl[ifoil].r_end ) {
					
					for(j=0; j<acl[ifoil].num_AC-1; j++) {
						if (r>=acl[ifoil].r_AC[j] && r<=acl[ifoil].r_AC[j+1]) {
							fac1 = (acl[ifoil].r_AC[j+1]-r)/(acl[ifoil].r_AC[j+1]-acl[ifoil].r_AC[j]);
							fac2 = (-acl[ifoil].r_AC[j]+r)/(acl[ifoil].r_AC[j+1]-acl[ifoil].r_AC[j]);
							ibm[ibi].angle_pitch[i]	= fac1*acl[ifoil].angle_pitchInp[j] + fac2*acl[ifoil].angle_pitchInp[j+1];
							ibm[ibi].chord_blade[i]	= fac1*acl[ifoil].chord_bladeInp[j] + fac2*acl[ifoil].chord_bladeInp[j+1];
						}
					}
				}
			}

			//if (!rank) printf("Turbine %d #### %d th elmt the chord %le and pitch angle %le \n", ibi, i,  ibm[ibi].chord_blade[i], ibm[ibi].angle_pitch[i]);
		}

	}

    	PetscPrintf(PETSC_COMM_WORLD, "Finish READing airfoil data for ACL\n");

	return(0);

}




// calculate the force on the actuator disk, called in the solvers.c beforce solving the momentum equation
PetscErrorCode Calc_F_lagr(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int NumberOfObjects)
{

  	PetscInt	l, ibi;
  	double	pi = 3.141592653589793, a = 0.25;
  	double 	C_T;  // = 4.0 / 3.0;
  	double 	U_ref, A_sum;
  	PetscReal	sign;
  	double 	Uref_x, Uref_y, Uref_z;


  	Calc_U_lagr(user, ibm, fsi, NumberOfObjects);
  
  	for (ibi=0; ibi<NumberOfObjects; ibi++) {
		
		double indf_ax=ibm[ibi].indf_axis;
  
	  	C_T = 4.0 * indf_ax * (1-indf_ax);

	  	C_T = C_T / ( (1.0 - indf_ax)* (1.0 - indf_ax) );


    		U_ref = 0.0;
    		A_sum = 0.0;
    		Uref_x = 0.0;
    		Uref_y = 0.0;
    		Uref_z = 0.0;

    		for (l=0; l<ibm[ibi].n_elmt; l++) {
      			Uref_x += ibm[ibi].U_lagr_x[l] * ibm[ibi].dA[l];
      			Uref_y += ibm[ibi].U_lagr_y[l] * ibm[ibi].dA[l];
      			Uref_z += ibm[ibi].U_lagr_z[l] * ibm[ibi].dA[l];
      			A_sum += ibm[ibi].dA[l];
    		}
    		Uref_x /= A_sum;
    		Uref_y /= A_sum;
    		Uref_z /= A_sum;

    		U_ref = Uref_x*fsi[ibi].nx_tb+Uref_y*fsi[ibi].ny_tb+Uref_z*fsi[ibi].nz_tb;

    		PetscPrintf(PETSC_COMM_WORLD, "**** The U_ref at %i th body: %le\n", ibi, U_ref);

    		sign = U_ref / fabs(U_ref+1.e-9);
    		for (l=0; l<ibm[ibi].n_elmt; l++) {
      			ibm[ibi].F_lagr_x[l] = -0.5 * C_T * (U_ref * U_ref) * sign * fsi[ibi].nx_tb;
      			ibm[ibi].F_lagr_y[l] = -0.5 * C_T * (U_ref * U_ref) * sign * fsi[ibi].ny_tb;
      			ibm[ibi].F_lagr_z[l] = -0.5 * C_T * (U_ref * U_ref) * sign * fsi[ibi].nz_tb;  
    		}


 	}

 	return(0);

}

// not for turbine simulations
PetscErrorCode Calc_F_lagr_noslip(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int NumberOfObjects)
{

  	int	l, ibi;
  	int 		nv1, nv2, nv3;
	double 		r1, r2, r3, rr, rx, ry, rz;


        int rank=0;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);


  	Calc_U_lagr(user, ibm, fsi, NumberOfObjects);
 
  	for (ibi=0; ibi<NumberOfObjects; ibi++) {
  	for (l=0; l<ibm[ibi].n_elmt; l++) {
             	nv1 = ibm[ibi].nv1[l];
            	nv2 = ibm[ibi].nv2[l];
              	nv3 = ibm[ibi].nv3[l];

               	rx = ibm[ibi].x_bp[nv2] - ibm[ibi].x_bp[nv1];
             	ry = ibm[ibi].y_bp[nv2] - ibm[ibi].y_bp[nv1];
             	rz = ibm[ibi].z_bp[nv2] - ibm[ibi].z_bp[nv1];

              	r1 = sqrt(rx*rx + ry*ry + rz*rz );
		
                rx = ibm[ibi].x_bp[nv3] - ibm[ibi].x_bp[nv2];
                ry = ibm[ibi].y_bp[nv3] - ibm[ibi].y_bp[nv2];
                rz = ibm[ibi].z_bp[nv3] - ibm[ibi].z_bp[nv2];

                r2 = sqrt(rx*rx + ry*ry + rz*rz );

                rx = ibm[ibi].x_bp[nv1] - ibm[ibi].x_bp[nv3];
                ry = ibm[ibi].y_bp[nv1] - ibm[ibi].y_bp[nv3];
                rz = ibm[ibi].z_bp[nv1] - ibm[ibi].z_bp[nv3];

                r3 = sqrt(rx*rx + ry*ry + rz*rz );

		rr = (r1 + r2 + r3)/3.0;
	
		if (rotor_model) {
			double U_ref=ibm[ibi].U_ref;
			double CD=ibm[ibi].CD_bluff;
    			double sign = U_ref / (fabs(U_ref)+1.e-20);
      			ibm[ibi].F_lagr_x[l]=-0.5*CD*(U_ref*U_ref)*sign*fsi[ibi].nx_tb;
      			ibm[ibi].F_lagr_y[l]=-0.5*CD*(U_ref*U_ref)*sign*fsi[ibi].ny_tb;
      			ibm[ibi].F_lagr_z[l]=-0.5*CD*(U_ref*U_ref)*sign*fsi[ibi].nz_tb; 

/*
			ibm[ibi].F_lagr_x[l] = rr * (ibm[ibi].u[l].x - ibm[ibi].U_lagr_x[l])/user->dt;
	      		ibm[ibi].F_lagr_y[l] = rr * (ibm[ibi].u[l].y - ibm[ibi].U_lagr_y[l])/user->dt;
      			ibm[ibi].F_lagr_z[l] = rr * (ibm[ibi].u[l].z - ibm[ibi].U_lagr_z[l])/user->dt;    
*/
		} else {
      			ibm[ibi].F_lagr_x[l] = rr * (ibm[ibi].u[l].x - ibm[ibi].U_lagr_x[l])/user->dt;
	      		ibm[ibi].F_lagr_y[l] = rr * (ibm[ibi].u[l].y - ibm[ibi].U_lagr_y[l])/user->dt;
      			ibm[ibi].F_lagr_z[l] = rr * (ibm[ibi].u[l].z - ibm[ibi].U_lagr_z[l])/user->dt;    
		}

    	}
  	}
	
 return(0);

}



// calculate force at largrangian points using actuator line model
//
// called in solvers.c before solving the momentum equation
PetscErrorCode Calc_F_lagr_ACL(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, ACL *acl, int NumberOfObjects)
{

  	PetscInt      	l, ibi, j;
  	double		pi = 3.141592653589793;
  	PetscReal		A_sum, U_ref, rr, rx, ry, rz, tmp, u_tangent;
	Cmpnts	      	U_b, n_relV, n_L, n_blade, n_rot;	
	int		nv1, nv2, nv3, ifoil;
	PetscReal 	fac1, fac2, r;
	// n_relV: the direction of relative velocity
	// n_L: the direction of Lift

	int istall;

  	int rank=0;
  	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);


	Calc_U_lagr(user, ibm, fsi, NumberOfObjects);


	double sign_L;


  	for (ibi=0; ibi<NumberOfObjects; ibi++) {


    		for (l=0; l<ibm[ibi].n_elmt; l++) {

			nv1 = ibm[ibi].nv1[l];
			nv2 = ibm[ibi].nv2[l];
			nv3 = ibm[ibi].nv3[l];

			rx = ibm[ibi].cent_x[l]-fsi[ibi].x_c;
			ry = ibm[ibi].cent_y[l]-fsi[ibi].y_c;
			rz = ibm[ibi].cent_z[l]-fsi[ibi].z_c;
	

//                        rx = ibm[ibi].x_bp[nv2] - ibm[ibi].x_bp[nv1];
//                        ry = ibm[ibi].y_bp[nv2] - ibm[ibi].y_bp[nv1];
//                        rz = ibm[ibi].z_bp[nv2] - ibm[ibi].z_bp[nv1];

//			double r1 = sqrt(pow( (ibm[ibi].x_bp[nv1] - fsi[ibi].x_c), 2 ) + pow( (ibm[ibi].y_bp[nv1] - fsi[ibi].y_c), 2 ) + pow( (ibm[ibi].z_bp[nv1] - fsi[ibi].z_c), 2 ));   
//			double r2 = sqrt(pow( (ibm[ibi].x_bp[nv2] - fsi[ibi].x_c), 2 ) + pow( (ibm[ibi].y_bp[nv2] - fsi[ibi].y_c), 2 ) + pow( (ibm[ibi].z_bp[nv2] - fsi[ibi].z_c), 2 ));   
			double r = sqrt(rx*rx+ry*ry+rz*rz);
                        n_blade.x = rx/r; n_blade.y = ry/r; n_blade.z = rz/r;

//			n_rot.x=fsi[ibi].ny_tb*n_blade.z-fsi[ibi].nz_tb*n_blade.y;
//			n_rot.y=-fsi[ibi].nx_tb*n_blade.z+fsi[ibi].nz_tb*n_blade.x;
//			n_rot.z=fsi[ibi].nx_tb*n_blade.y-fsi[ibi].ny_tb*n_blade.x;

			double ux, uy, uz;

			if (rotor_model == 3) {
				ux=0.5*(ibm[ibi].u[nv1].x+ibm[ibi].u[nv2].x); 
				uy=0.5*(ibm[ibi].u[nv1].y+ibm[ibi].u[nv2].y); 
				uz=0.5*(ibm[ibi].u[nv1].z+ibm[ibi].u[nv2].z); 
			 } else if (rotor_model == 2) {
				double fac=1.0/3.0;
				ux=fac*(ibm[ibi].u[nv1].x+ibm[ibi].u[nv2].x+ibm[ibi].u[nv3].x); 
				uy=fac*(ibm[ibi].u[nv1].y+ibm[ibi].u[nv2].y+ibm[ibi].u[nv3].y); 
				uz=fac*(ibm[ibi].u[nv1].z+ibm[ibi].u[nv2].z+ibm[ibi].u[nv3].z); 
			}			


			double Ublade=sqrt(pow(ux,2)+pow(uy,2)+pow(uz,2));

			n_rot.x=ux/(Ublade+1.e-20);
			n_rot.y=uy/(Ublade+1.e-20);
			n_rot.z=uz/(Ublade+1.e-20);

//          		PetscPrintf(PETSC_COMM_WORLD, "Nv1 n_rot.x %le n_rot.y %le n_rot.z %le\n",ibm[ibi].u[nv1].x,ibm[ibi].u[nv1].y,ibm[ibi].u[nv1].z); 
//          		PetscPrintf(PETSC_COMM_WORLD, "ibi %d Elmt %d nv1 %d nv2 %d \n",ibi, l, nv1, nv2); 

			double Uaxis=ibm[ibi].U_lagr_x[l]*fsi[ibi].nx_tb+ibm[ibi].U_lagr_y[l]*fsi[ibi].ny_tb+ibm[ibi].U_lagr_z[l]*fsi[ibi].nz_tb;

//			Uaxis=Uaxis/(1.0-indf_ax);

			double Urot=ibm[ibi].U_lagr_x[l]*n_rot.x+ibm[ibi].U_lagr_y[l]*n_rot.y+ibm[ibi].U_lagr_z[l]*n_rot.z;

//          		PetscPrintf(PETSC_COMM_WORLD, "r %le Urot %le Ublade %le \n", 0.5*(r1+r2), Urot, Ublade);
 
			U_b.x=Uaxis*fsi[ibi].nx_tb+Urot*n_rot.x;
			U_b.y=Uaxis*fsi[ibi].ny_tb+Urot*n_rot.y;
			U_b.z=Uaxis*fsi[ibi].nz_tb+Urot*n_rot.z;

//			U_b.x=Uaxis*fsi[ibi].nx_tb;
//			U_b.y=Uaxis*fsi[ibi].ny_tb;
//			U_b.z=Uaxis*fsi[ibi].nz_tb;


			U_b.x=U_b.x-0.5*(ibm[ibi].u[nv1].x+ibm[ibi].u[nv2].x);
			U_b.y=U_b.y-0.5*(ibm[ibi].u[nv1].y+ibm[ibi].u[nv2].y);
			U_b.z=U_b.z-0.5*(ibm[ibi].u[nv1].z+ibm[ibi].u[nv2].z);

//			double u_r=U_b.x*n_blade.x+U_b.y*n_blade.y+U_b.z*n_blade.z;	

//			U_b.x=U_b.x-u_r*n_blade.x; 
//			U_b.y=U_b.y-u_r*n_blade.y; 
//			U_b.z=U_b.z-u_r*n_blade.z; 


//			double Uaxis=ibm[ibi].U_lagr_x[l]*fsi[ibi].nx_tb+ibm[ibi].U_lagr_y[l]*fsi[ibi].ny_tb+ibm[ibi].U_lagr_z[l]*fsi[ibi].nz_tb;

//			Uaxis=Uaxis/(1.0-indf_ax);

			U_ref=sqrt(pow(U_b.x,2)+pow(U_b.y,2)+pow(U_b.z,2));
//          		PetscPrintf(PETSC_COMM_WORLD, "Turbine %d Nelmt %d Ux %le Uy %le Uz %le\n", ibi,l, U_b.x, U_b.y, U_b.z); 

			n_relV.x = U_b.x/(U_ref+1.e-20); n_relV.y = U_b.y/(U_ref+1.e-20); n_relV.z = U_b.z/(U_ref+1.e-20); 

			tmp = n_relV.x*n_rot.x+n_relV.y*n_rot.y+n_relV.z*n_rot.z;
			
//			if (!rank) printf("**** tmp for AOA %le AOA %le %le \n", tmp, acos(fabs(tmp)-1.e-9)*180.0/pi, acos(1.0)*180.0/pi);


//			if (tmp>0.0) n_rot.x=-n_rot.x, n_rot.y=-n_rot.y, n_rot.z=-n_rot.z;

			ibm[ibi].angle_attack[l] = fabs(acos(fabs(tmp)-1.e-9)) * 180.0 / pi - ibm[ibi].angle_pitch[l];

//			if (!rank) printf(" Turbine %d Nelmt %d AOA %le Pitch %le \n", ibi, l, fabs(acos(fabs(tmp)-1.e-9)) * 180.0 / pi , ibm[ibi].angle_pitch[l]);
        		n_L.x = n_relV.y*n_blade.z - n_relV.z*n_blade.y; 
			n_L.y = n_relV.z*n_blade.x - n_relV.x*n_blade.z; 
			n_L.z = n_relV.x*n_blade.y - n_relV.y*n_blade.x;
		
			tmp = n_rot.x*n_L.x + n_rot.y*n_L.y + n_rot.z*n_L.z;
			
			if (tmp < 0.0) n_L.x = -n_L.x, n_L.y = -n_L.y, n_L.z = -n_L.z; 				 

                        r = sqrt( pow((ibm[ibi].cent_x[l]-fsi[ibi].x_c),2) + pow((ibm[ibi].cent_y[l]-fsi[ibi].y_c),2) + pow((ibm[ibi].cent_z[l]-fsi[ibi].z_c),2));

                        for(ifoil=0; ifoil<num_foiltype; ifoil++) {
				
                                if( r>=acl[ifoil].r_beg && r<=acl[ifoil].r_end ) {

					int inrange=0;

                                        for(j=0; j<acl[ifoil].num_CD-1; j++) {
                                                if (ibm[ibi].angle_attack[l]>=acl[ifoil].ang_CD[j] && ibm[ibi].angle_attack[l]<=acl[ifoil].ang_CD[j+1]) {
                                                        fac1 = (acl[ifoil].ang_CD[j+1]-ibm[ibi].angle_attack[l])/(acl[ifoil].ang_CD[j+1]-acl[ifoil].ang_CD[j]);
                                                        fac2 = (-acl[ifoil].ang_CD[j]+ibm[ibi].angle_attack[l])/(acl[ifoil].ang_CD[j+1]-acl[ifoil].ang_CD[j]);
                                                        ibm[ibi].CD[l] = fac1*acl[ifoil].CDInp[j] + fac2*acl[ifoil].CDInp[j+1];
							inrange=1;
                                                }
                                        }
					

					if (!inrange) {
                                	        if (ibm[ibi].angle_attack[l]<acl[ifoil].ang_CD[0] && ibm[ibi].angle_attack[l]>-45.0) ibm[ibi].CD[l] = (-1.2 - acl[ifoil].CDInp[0]) * (ibm[ibi].angle_attack[l] - acl[ifoil].ang_CD[0]) / (-45.0-acl[ifoil].ang_CD[0]) + acl[ifoil].CDInp[0];
                        	                if ( ibm[ibi].angle_attack[l]<=-45.0 && ibm[ibi].angle_attack[l]>=-90.0 ) ibm[ibi].CD[l] = -1.2;

						if ( ibm[ibi].angle_attack[l] < -90.0 ) ibm[ibi].CD[l] = -1.2;

 
						if (ibm[ibi].angle_attack[l]>acl[ifoil].ang_CD[acl[ifoil].num_CD-1] && ibm[ibi].angle_attack[l]<45.0) ibm[ibi].CD[l] = (1.2 - acl[ifoil].CDInp[acl[ifoil].num_CD-1]) * (ibm[ibi].angle_attack[l] - acl[ifoil].ang_CD[acl[ifoil].num_CD-1]) / (45.0-acl[ifoil].ang_CD[acl[ifoil].num_CD-1]) + acl[ifoil].CDInp[acl[ifoil].num_CD-1];
						if ( ibm[ibi].angle_attack[l]>=45.0 && ibm[ibi].angle_attack[l]<=90.0 ) ibm[ibi].CD[l] = 1.2;


					}

					inrange=0;
                                        for(j=0; j<acl[ifoil].num_CL-1; j++) {
                                                if (ibm[ibi].angle_attack[l]>=acl[ifoil].ang_CL[j] && ibm[ibi].angle_attack[l]<=acl[ifoil].ang_CL[j+1]) {
                                                        fac1 = (acl[ifoil].ang_CL[j+1]-ibm[ibi].angle_attack[l])/(acl[ifoil].ang_CL[j+1]-acl[ifoil].ang_CL[j]);
                                                        fac2 = (-acl[ifoil].ang_CL[j]+ibm[ibi].angle_attack[l])/(acl[ifoil].ang_CL[j+1]-acl[ifoil].ang_CL[j]);
                                                        ibm[ibi].CL[l] = fac1*acl[ifoil].CLInp[j] + fac2*acl[ifoil].CLInp[j+1];
							inrange=1;
                                                }

                                        }

					if (!inrange) {

                        	                if (ibm[ibi].angle_attack[l]<acl[ifoil].ang_CL[0] && ibm[ibi].angle_attack[l]>-45.0) ibm[ibi].CL[l] = (-1.05 - acl[ifoil].CLInp[0] ) * (ibm[ibi].angle_attack[l] - acl[ifoil].ang_CL[0] ) / (-45.0-acl[ifoil].ang_CL[0] ) + acl[ifoil].CLInp[0] ;
                	                        if ( ibm[ibi].angle_attack[l]<=-45.0 && ibm[ibi].angle_attack[l]>=-90.0 ) ibm[ibi].CL[l] = 1.05*sin(2.0*ibm[ibi].angle_attack[l]*pi/180.0);
						if ( ibm[ibi].angle_attack[l] < -90.0 ) ibm[ibi].CL[l] = 0.0;


	                                        if (ibm[ibi].angle_attack[l]>acl[ifoil].ang_CL[acl[ifoil].num_CL-1] && ibm[ibi].angle_attack[l]<45.0) ibm[ibi].CL[l] = (1.05 - acl[ifoil].CLInp[acl[ifoil].num_CL-1] ) * (ibm[ibi].angle_attack[l] - acl[ifoil].ang_CL[acl[ifoil].num_CL-1] ) / (45.0-acl[ifoil].ang_CL[acl[ifoil].num_CL-1] ) + acl[ifoil].CLInp[acl[ifoil].num_CL-1] ;
						if ( ibm[ibi].angle_attack[l]>=45.0 && ibm[ibi].angle_attack[l]<=90.0 ) ibm[ibi].CL[l] = 1.05*sin(2.0*ibm[ibi].angle_attack[l]*pi/180.0);

					}
// L=1.05*sin(2*a)


                        // add 3D and rotational effects

/*
					if (ibm[ibi].angle_attack[l]>0.0 && ibm[ibi].angle_attack[l]<20.0)
					{
                        			ibm[ibi].CD[l] += coef_cr1 * pow( ibm[ibi].chord_blade[l]/r, coef_cr2 ) * ( 0.0 - 0.0 );
                        			ibm[ibi].CL[l] += coef_cr1 * pow( ibm[ibi].chord_blade[l]/r, coef_cr2 ) * ( 2.0*pi*pi*ibm[ibi].angle_attack[l]/180.0 - ibm[ibi].CL[l]);
					} 
*/					
//					if ( ibm[ibi].CD[l] < 0.0) ibm[ibi].CD[l] = 0.0;
//					ibm[ibi].CD[l] = 0.0;
                                }



                        }

			// add 3D and rotational effects
//			ibm[ibi].CD[l] += coef_cr1 * pow( ibm[ibi].chord_blade[l]/r, coef_cr2 ) * ( ibm[ibi].CD[l] - acl[1].CDInp[0] );
//			ibm[ibi].CL[l] += coef_cr1 * pow( ibm[ibi].chord_blade[l]/r, coef_cr2 ) * ( 2.0*pi*pi*fabs(ibm[ibi].angle_attack[l])/180.0 - ibm[ibi].CL[l]);

//			ibm[ibi].CL[l] += coef_cr1 * pow( ibm[ibi].chord_blade[l]/r, coef_cr2 ) * ( 2.0*pi*pi*fabs(ibm[ibi].angle_attack[l])/180.0 - ibm[ibi].CL[l]);

			double factor;

			if (rotor_model==2) {
				factor=1.0;
			} else if (rotor_model == 3) {
				factor=ibm[ibi].chord_blade[l];
			}

			// force from C_D
      			ibm[ibi].F_lagr_x[l] = -0.5 * U_ref * U_ref * fabs(ibm[ibi].CD[l]) * n_relV.x * factor;
      			ibm[ibi].F_lagr_y[l] = -0.5 * U_ref * U_ref * fabs(ibm[ibi].CD[l]) * n_relV.y * factor;
      			ibm[ibi].F_lagr_z[l] = -0.5 * U_ref * U_ref * fabs(ibm[ibi].CD[l]) * n_relV.z * factor; 
			
          		//PetscPrintf(PETSC_COMM_WORLD, "%d Add Fz from CD %le, Uref %le CD %le n_rel %le\n", l, ibm[ibi].F_lagr_z[l], U_ref, fabs(ibm[ibi].CD[l]), n_relV.z);
			// add force from C_L
                        ibm[ibi].F_lagr_x[l] += -0.5 * U_ref * U_ref * fabs(ibm[ibi].CL[l]) * n_L.x * factor;
                        ibm[ibi].F_lagr_y[l] += -0.5 * U_ref * U_ref * fabs(ibm[ibi].CL[l]) * n_L.y * factor;
                        ibm[ibi].F_lagr_z[l] += -0.5 * U_ref * U_ref * fabs(ibm[ibi].CL[l]) * n_L.z * factor;

          		//PetscPrintf(PETSC_COMM_WORLD, "%d Add Fz from CL %le, Uref %le CL %le n_L.z %le\n", l, ibm[ibi].F_lagr_z[l], U_ref, fabs(ibm[ibi].CL[l]), n_L.z);

			// Modification to 2D force

			double pi = 3.1415926;
			double phi1 = (ibm[ibi].angle_attack[l] + ibm[ibi].angle_pitch[l])*pi/180.0;
			double g1 = exp(-0.125*(num_blade*ibm[ibi].Tipspeedratio-21.0)) + 0.1; 
			double F1 = 2.0*acos(exp(-g1*num_blade*(max(r_rotor/reflength_wt-r,0.0))/2.0/r/fabs(sin(phi1))))/pi;

                        ibm[ibi].F_lagr_x[l] *= F1;
                        ibm[ibi].F_lagr_y[l] *= F1;
                        ibm[ibi].F_lagr_z[l] *= F1;

//                        ibm[ibi].F_lagr_x[l] = 0.0;
//                        ibm[ibi].F_lagr_y[l] = 0.0;
//                        ibm[ibi].F_lagr_z[l] = 0.0;



			if (AL_Noslip) {
	                        rx = ibm[ibi].x_bp[nv2] - ibm[ibi].x_bp[nv1];
        	                ry = ibm[ibi].y_bp[nv2] - ibm[ibi].y_bp[nv1];
                	        rz = ibm[ibi].z_bp[nv2] - ibm[ibi].z_bp[nv1];

                        	rr = sqrt(rx*rx + ry*ry + rz*rz );

				double U_axis=U_b.x*fsi[ibi].nx_tb+U_b.y*fsi[ibi].ny_tb+U_b.z*fsi[ibi].nz_tb;
				double U_rot=U_b.x*n_rot.x+U_b.y*n_rot.y+U_b.z*n_rot.z;

                        	ibm[ibi].F_lagr_x[l]=factor*rr*ibm[ibi].chord_blade[l]*(0.0-U_axis)*fsi[ibi].nx_tb/user->dt;
	                        ibm[ibi].F_lagr_y[l]=factor*rr*ibm[ibi].chord_blade[l]*(0.0-U_axis)*fsi[ibi].ny_tb/user->dt;
        	                ibm[ibi].F_lagr_z[l]=factor*rr*ibm[ibi].chord_blade[l]*(0.0-U_axis)*fsi[ibi].nz_tb/user->dt;

                        	ibm[ibi].F_lagr_x[l]+=factor*rr*0.2*ibm[ibi].chord_blade[l]*(U_rot)*n_rot.x/user->dt;
	                        ibm[ibi].F_lagr_y[l]+=factor*rr*0.2*ibm[ibi].chord_blade[l]*(U_rot)*n_rot.y/user->dt;
        	                ibm[ibi].F_lagr_z[l]+=factor*rr*0.2*ibm[ibi].chord_blade[l]*(U_rot)*n_rot.z/user->dt;
			}



			if (r<r_nacelle/reflength_wt) {

				if (!IB_delta) {
				double nx=fsi[ibi].nx_tb, ny=fsi[ibi].ny_tb, nz=fsi[ibi].nz_tb;
				double F_axis=ibm[ibi].F_lagr_x[l]*nx+ibm[ibi].F_lagr_y[l]*ny+ibm[ibi].F_lagr_z[l]*nz;

				ibm[ibi].F_lagr_x[l]=F_axis*nx;	
				ibm[ibi].F_lagr_y[l]=F_axis*ny;	
				ibm[ibi].F_lagr_z[l]=F_axis*nz;	

				} else 	{

				ibm[ibi].F_lagr_x[l]=0.0;	
				ibm[ibi].F_lagr_y[l]=0.0;	
				ibm[ibi].F_lagr_z[l]=0.0;	

				}

			}

    		}
  	}

//    PetscInt rank=0;

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (!rank) {

      	FILE *f;
      	char filen[80];


	for (ibi=0; ibi<NumberOfObjects; ibi++) {
      		sprintf(filen, "Force@ACL%2.2d", ibi);
	      	f = fopen(filen, "w");

		for (l=0; l<ibm[0].n_elmt; l++) {
			double rr = sqrt(pow( (ibm[ibi].cent_x[l] - fsi[ibi].x_c), 2 ) + pow( (ibm[ibi].cent_y[l] - fsi[ibi].y_c), 2 ) + pow( (ibm[ibi].cent_z[l] - fsi[ibi].z_c), 2 ));   
      			PetscFPrintf(PETSC_COMM_WORLD, f, "%le %le %le %le \n", rr, ibm[ibi].F_lagr_x[l], ibm[ibi].F_lagr_y[l], ibm[ibi].F_lagr_z[l]);
      		}

      		fclose(f);

                sprintf(filen, "AOA@ACL%2.2d", ibi);
                f = fopen(filen, "w");

                for (l=0; l<ibm[0].n_elmt; l++) {
			double rr = sqrt(pow( (ibm[ibi].cent_x[l] - fsi[ibi].x_c), 2 ) + pow( (ibm[ibi].cent_y[l] - fsi[ibi].y_c), 2 ) + pow( (ibm[ibi].cent_z[l] - fsi[ibi].z_c), 2 ));   
                        PetscFPrintf(PETSC_COMM_WORLD, f, "%le %le  \n", rr, ibm[ibi].angle_attack[l]);
                }

                fclose(f);


	}

    }

//    exit(0);
    return false;
}



/* Interpolate the velocity at the Lagrangian points */
// subroutine for Calc_F_lagr_ACL and Calc_F_lagr
PetscErrorCode Calc_U_lagr(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int NumberOfObjects)
{

  	DM              da = user->da, fda = user->fda;
  	DMDALocalInfo     info;
  	PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  	PetscInt        mx, my, mz; // Dimensions in three directions
  	PetscInt        i, j, k, l, ibi;
  	PetscInt	lxs, lxe, lys, lye, lzs, lze;

  	Cmpnts	***ucat, ***coor, ***csi, ***eta, ***zet;

  	PetscReal 	***aj;

  	Vec		Coor;

  	PetscReal	dfunc;

  	PetscReal	xc, yc, zc, hx, hy, hz, vol_eul, vol_lagr;

  	double r1, r2, r3;

  	DMDAGetLocalInfo(da, &info);
  	mx = info.mx; my = info.my; mz = info.mz;
  	xs = info.xs; xe = xs + info.xm;
  	ys = info.ys; ye = ys + info.ym;
  	zs = info.zs; ze = zs + info.zm;

  	lxs = xs; lxe = xe;
  	lys = ys; lye = ye;
  	lzs = zs; lze = ze;

  	if (xs==0) lxs = xs+1;
  	if (ys==0) lys = ys+1;
  	if (zs==0) lzs = zs+1;

  	if (xe==mx) lxe = xe-1;
  	if (ye==my) lye = ye-1;
  	if (ze==mz) lze = ze-1;


  	DMDAGetGhostedCoordinates(da, &Coor);
  	DMDAVecGetArray(fda, Coor, &coor);
  	DMDAVecGetArray(fda, user->lUcat, &ucat);
  	DMDAVecGetArray(fda, user->lCsi,  &csi);
  	DMDAVecGetArray(fda, user->lEta,  &eta);
  	DMDAVecGetArray(fda, user->lZet,  &zet);
  	DMDAVecGetArray(da,  user->lAj,  &aj);

  	for (ibi=0; ibi<NumberOfObjects; ibi++) {
    	for (l=0; l<ibm[ibi].n_elmt; l++) {
      		ibm[ibi].U_lagr_x[l] = 0.0;
      		ibm[ibi].U_lagr_y[l] = 0.0;
      		ibm[ibi].U_lagr_z[l] = 0.0;
    	}
  	}


        clock_t start, end;
        double elapsed;

        start = clock();
	double ni[3], nj[3], nk[3];

	for (ibi=0; ibi<NumberOfObjects; ibi++) {
  	for (l=0; l<ibm[ibi].n_elmt; l++) {

		double dhx_=ibm[ibi].dhx[l];
		double dhy_=ibm[ibi].dhy[l];
		double dhz_=ibm[ibi].dhz[l];

	    	for (k = ibm[ibi].k_min[l]; k<ibm[ibi].k_max[l]; k++) 
	      	for (j = ibm[ibi].j_min[l]; j<ibm[ibi].j_max[l]; j++) 
	        for (i = ibm[ibi].i_min[l]; i<ibm[ibi].i_max[l]; i++) {

			xc=coor[k][j][i].x;
			yc=coor[k][j][i].y;
			zc=coor[k][j][i].z;

	            	double rx= (xc - ibm[ibi].cent_x[l]), ry = (yc - ibm[ibi].cent_y[l]), rz = (zc - ibm[ibi].cent_z[l]);

			Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);

			r1=(rx*ni[0]+ry*ni[1]+rz*ni[2])/dhx_; 
			r2=(rx*nj[0]+ry*nj[1]+rz*nj[2])/dhy_; 
			r3=(rx*nk[0]+ry*nk[1]+rz*nk[2])/dhz_; 

			/*
			if (deltafunc == 1) dfunc =  dfunc_2h(r1) * dfunc_2h(r2) * dfunc_2h(r3);
			if (deltafunc == 2) dfunc =  dfunc_4h(r1) * dfunc_4h(r2) * dfunc_4h(r3);
	                if (deltafunc == 3) dfunc =  dfunc_s3h(r1) * dfunc_s3h(r2) * dfunc_s3h(r3);
	                if (deltafunc == 4) dfunc =  dfunc_s4h(r1) * dfunc_s4h(r2) * dfunc_s4h(r3);
			*/

	                //dfunc =  dfunc_nh(r1,3) * dfunc_nh(r2,3) * dfunc_nh(r3,3);
	                dfunc =  dfunc_s3h(r1) * dfunc_s3h(r2) * dfunc_s3h(r3);
	            	ibm[ibi].U_lagr_x[l] += ucat[k][j][i].x * dfunc;
	            	ibm[ibi].U_lagr_y[l] += ucat[k][j][i].y * dfunc;
	            	ibm[ibi].U_lagr_z[l] += ucat[k][j][i].z * dfunc;          
   
      		}

  	}
  	}

        end = clock();
        elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;

        PetscPrintf(PETSC_COMM_WORLD, "Time for U_larg local %le \n", elapsed);



  	//PetscBarrier(PETSC_NULL);

        start = clock();
	// the n_elmt for each turbine is assumed to be the same
  	int nelmt_Max=-1000;
  	for (ibi=0; ibi<NumberOfObjects; ibi++)  {
		if (ibm[ibi].n_elmt>nelmt_Max) nelmt_Max=ibm[ibi].n_elmt;	
  	}


	/*
  	double    u_local[NumberOfObjects][nelmt_Max][3] ,u_sum[NumberOfObjects][nelmt_Max][3];
  	int totNum = NumberOfObjects*nelmt_Max*3;

  	for (ibi=0; ibi<NumberOfObjects; ibi++)  
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
		u_local[ibi][i][0] = ibm[ibi].U_lagr_x[i];
		u_local[ibi][i][1] = ibm[ibi].U_lagr_y[i];
		u_local[ibi][i][2] = ibm[ibi].U_lagr_z[i];
  	}

  	MPI_Allreduce( &(u_local[0][0][0]), &(u_sum[0][0][0]), totNum, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  	for (ibi=0; ibi<NumberOfObjects; ibi++) 
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        	ibm[ibi].U_lagr_x[i] = u_sum[ibi][i][0];
        	ibm[ibi].U_lagr_y[i] = u_sum[ibi][i][1];
        	ibm[ibi].U_lagr_z[i] = u_sum[ibi][i][2];
  	}
	*/


  	double    u_local[NumberOfObjects][nelmt_Max] ,u_sum[NumberOfObjects][nelmt_Max];
  	int totNum = NumberOfObjects*nelmt_Max;

	// x
  	for (ibi=0; ibi<NumberOfObjects; ibi++)  
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
		u_local[ibi][i] = ibm[ibi].U_lagr_x[i];
  	}

  	MPI_Allreduce( &(u_local[0][0]), &(u_sum[0][0]), totNum, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  	for (ibi=0; ibi<NumberOfObjects; ibi++) 
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        	ibm[ibi].U_lagr_x[i] = u_sum[ibi][i];
  	}

  	//PetscBarrier(PETSC_NULL);
	// y
  	for (ibi=0; ibi<NumberOfObjects; ibi++) 
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
  	      u_local[ibi][i] = ibm[ibi].U_lagr_y[i];
  	}

  	MPI_Allreduce( &(u_local[0][0]), &(u_sum[0][0]), totNum, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  	for (ibi=0; ibi<NumberOfObjects; ibi++)
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        	ibm[ibi].U_lagr_y[i] = u_sum[ibi][i];
  	}

  	//PetscBarrier(PETSC_NULL);
// z
  	for (ibi=0; ibi<NumberOfObjects; ibi++) 
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        	u_local[ibi][i] = ibm[ibi].U_lagr_z[i];
  	}

  	MPI_Allreduce( &(u_local[0][0]), &(u_sum[0][0]), totNum, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  	for (ibi=0; ibi<NumberOfObjects; ibi++)
  	for (i=0; i<ibm[ibi].n_elmt; i++ ) {
        	ibm[ibi].U_lagr_z[i] = u_sum[ibi][i];
  	}

        end = clock();
        elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;

        PetscPrintf(PETSC_COMM_WORLD, "Time for U_larg global sum %le \n", elapsed);

  	//PetscBarrier(PETSC_NULL);

  	if (ii_periodicWT || jj_periodicWT || kk_periodicWT) {

    		double xc_min = 1.0e6;        
 	   	double yc_min = 1.0e6;
	    	double zc_min = 1.0e6;

  		for (ibi=0; ibi<NumberOfObjects; ibi++) {
      			xc_min = PetscMin(xc_min, fsi[ibi].x_c);
	      		yc_min = PetscMin(yc_min, fsi[ibi].y_c);
      			zc_min = PetscMin(zc_min, fsi[ibi].z_c);
	    	}

//        	PetscPrintf(PETSC_COMM_WORLD, "HERE 1 \n");
		int IndWT[Nz_WT][Ny_WT][Nx_WT];

		double fac_x=1.0/Sx_WT, fac_y=1.0/Sy_WT, fac_z=1.0/Sz_WT;
        	for (ibi=0; ibi<NumberOfObjects; ibi++) {
			int ii=(int)((fsi[ibi].x_c-xc_min+1.e-9)*fac_x);
			int jj=(int)((fsi[ibi].y_c-yc_min+1.e-9)*fac_y);
			int kk=(int)((fsi[ibi].z_c-zc_min+1.e-9)*fac_z);

        		PetscPrintf(PETSC_COMM_WORLD, "ibi ii jj kk %d %d %d %d \n", ibi, ii, jj, kk);
			IndWT[kk][jj][ii]=ibi;
		}

		int i, j, k;

//        	PetscPrintf(PETSC_COMM_WORLD, "HERE 2 \n");
		for (k=0;k<Nz_WT;k++)
		for (j=0;j<Ny_WT;j++)
		for (i=0;i<Nx_WT;i++){

			if (ii_periodicWT && i==0) {
				ibi=IndWT[k][j][i];
				int ibi1=IndWT[k][j][Nx_WT-1];
				for (l=0; l<ibm[ibi].n_elmt; l++) {
					ibm[ibi].U_lagr_x[l]=ibm[ibi].U_lagr_x[l]+ibm[ibi1].U_lagr_x[l];
					ibm[ibi].U_lagr_y[l]=ibm[ibi].U_lagr_y[l]+ibm[ibi1].U_lagr_y[l];
					ibm[ibi].U_lagr_z[l]=ibm[ibi].U_lagr_z[l]+ibm[ibi1].U_lagr_z[l];
				}
			}

                        if (jj_periodicWT && j==0) {
                                ibi=IndWT[k][j][i];
                                int ibi1=IndWT[k][Ny_WT-1][i];
                                for (l=0; l<ibm[ibi].n_elmt; l++) {
                                        ibm[ibi].U_lagr_x[l]=ibm[ibi].U_lagr_x[l]+ibm[ibi1].U_lagr_x[l];
                                        ibm[ibi].U_lagr_y[l]=ibm[ibi].U_lagr_y[l]+ibm[ibi1].U_lagr_y[l];
                                        ibm[ibi].U_lagr_z[l]=ibm[ibi].U_lagr_z[l]+ibm[ibi1].U_lagr_z[l];
                                }
                        }

                        if (kk_periodicWT && k==0) {
                                ibi=IndWT[k][j][i];
                                int ibi1=IndWT[Nz_WT-1][j][i];
                                for (l=0; l<ibm[ibi].n_elmt; l++) {
                                        ibm[ibi].U_lagr_x[l]=ibm[ibi].U_lagr_x[l]+ibm[ibi1].U_lagr_x[l];
                                        ibm[ibi].U_lagr_y[l]=ibm[ibi].U_lagr_y[l]+ibm[ibi1].U_lagr_y[l];
                                        ibm[ibi].U_lagr_z[l]=ibm[ibi].U_lagr_z[l]+ibm[ibi1].U_lagr_z[l];
                                }

//				PetscPrintf(PETSC_COMM_WORLD, "U_lagr_z %le \n",ibm[ibi].U_lagr_z[0] );
                        }
		}


//        	PetscPrintf(PETSC_COMM_WORLD, "HERE 3 \n");
                for (k=0;k<Nz_WT;k++)
                for (j=0;j<Ny_WT;j++)
                for (i=0;i<Nx_WT;i++){

                        if (ii_periodicWT && i==Nx_WT-1) {
                                ibi=IndWT[k][j][i];
                                int ibi1=IndWT[k][j][0];
                                for (l=0; l<ibm[ibi].n_elmt; l++) {
                                        ibm[ibi].U_lagr_x[l]=ibm[ibi1].U_lagr_x[l];
                                        ibm[ibi].U_lagr_y[l]=ibm[ibi1].U_lagr_y[l];
                                        ibm[ibi].U_lagr_z[l]=ibm[ibi1].U_lagr_z[l];
                                }
                        }

                        if (jj_periodicWT && j==Ny_WT-1) {
                                ibi=IndWT[k][j][i];
                                int ibi1=IndWT[k][0][i];
                                for (l=0; l<ibm[ibi].n_elmt; l++) {
                                        ibm[ibi].U_lagr_x[l]=ibm[ibi1].U_lagr_x[l];
                                        ibm[ibi].U_lagr_y[l]=ibm[ibi1].U_lagr_y[l];
                                        ibm[ibi].U_lagr_z[l]=ibm[ibi1].U_lagr_z[l];
                                }
                        }

                        if (kk_periodicWT && k==Nz_WT-1) {
                                ibi=IndWT[k][j][i];
                                int ibi1=IndWT[0][j][i];
                                for (l=0; l<ibm[ibi].n_elmt; l++) {
                                        ibm[ibi].U_lagr_x[l]=ibm[ibi1].U_lagr_x[l];
                                        ibm[ibi].U_lagr_y[l]=ibm[ibi1].U_lagr_y[l];
                                        ibm[ibi].U_lagr_z[l]=ibm[ibi1].U_lagr_z[l];
                                }
                        }
                }

//        	PetscPrintf(PETSC_COMM_WORLD, "HERE 4 \n");

	}


	if (MoveFrame) {
		for (ibi=0; ibi<NumberOfObjects; ibi++)
	  	for (l=0; l<ibm[ibi].n_elmt; l++) {
                	ibm[ibi].U_lagr_x[l] += u_frame;
                	ibm[ibi].U_lagr_y[l] += v_frame;
             		ibm[ibi].U_lagr_z[l] += w_frame;
  		}
  	}


  	DMDAVecRestoreArray(fda, Coor, &coor);
  	DMDAVecRestoreArray(fda, user->lUcat, &ucat);
  	DMDAVecRestoreArray(fda, user->lCsi,  &csi);
  	DMDAVecRestoreArray(fda, user->lEta,  &eta);
  	DMDAVecRestoreArray(fda, user->lZet,  &zet);
  	DMDAVecRestoreArray(da,  user->lAj,  &aj);

	return(0);
}


// distribute the force on the turbines to background grid.
//
// called after Calc_F_lagr_ACL or Calc_F_lagr before solving the momentum equatiion
PetscErrorCode Calc_F_eul(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int NumberOfObjects)
{

  	DM              da = user->da, fda = user->fda;
  	DMDALocalInfo     info;
  	PetscInt        xs, xe, ys, ye, zs, ze; // Local grid information
  	PetscInt        mx, my, mz; // Dimensions in three directions
  	PetscInt        i, j, k, l, ibi;
  	PetscInt	lxs, lxe, lys, lye, lzs, lze;

  	Cmpnts		***lf_eul, ***coor, ***csi, ***eta, ***zet;

  	PetscReal 	***aj, ***nvert;

  	Vec		Coor;

  	PetscReal	xc, yc, zc, hx, hy, hz, vol_eul, vol_lagr;
  
  	double		dfunc;

  	double r1, r2, r3;

  	DMDAGetLocalInfo(da, &info);
  	mx = info.mx; my = info.my; mz = info.mz;
  	xs = info.xs; xe = xs + info.xm;
  	ys = info.ys; ye = ys + info.ym;
  	zs = info.zs; ze = zs + info.zm;


	lxs = xs; lxe = xe;
  	lys = ys; lye = ye;
  	lzs = zs; lze = ze;

  	if (xs==0) lxs = xs+1;
  	if (ys==0) lys = ys+1;
  	if (zs==0) lzs = zs+1;

  	if (xe==mx) lxe = xe-1;
  	if (ye==my) lye = ye-1;
  	if (ze==mz) lze = ze-1;


  	DMDAGetGhostedCoordinates(da, &Coor);
  	DMDAVecGetArray(fda, Coor, &coor);
  	DMDAVecGetArray(fda, user->lF_eul, &lf_eul);
  	DMDAVecGetArray(fda, user->lCsi,  &csi);
  	DMDAVecGetArray(fda, user->lEta,  &eta);
  	DMDAVecGetArray(fda, user->lZet,  &zet);
  	DMDAVecGetArray(da,  user->lAj,  &aj);

	DMDAVecGetArray(da, user->lNvert, &nvert);

  	double ni[3], nj[3], nk[3];
  	for (ibi=0; ibi<NumberOfObjects; ibi++) {
  	for (l=0; l<ibm[ibi].n_elmt; l++) {

	    	for (k=ibm[ibi].k_min[l]; k<ibm[ibi].k_max[l]; k++) 
	      	for (j=ibm[ibi].j_min[l]; j<ibm[ibi].j_max[l]; j++) 
	        for (i=ibm[ibi].i_min[l]; i<ibm[ibi].i_max[l]; i++) {

//                xc = (coor[k  ][j  ][i].x + coor[k-1][j  ][i].x + coor[k  ][j-1][i].x + coor[k-1][j-1][i].x) * 0.25;
//                yc = (coor[k  ][j  ][i].y + coor[k-1][j  ][i].y + coor[k  ][j-1][i].y + coor[k-1][j-1][i].y) * 0.25;
//                zc = (coor[k  ][j  ][i].z + coor[k-1][j  ][i].z + coor[k  ][j-1][i].z + coor[k-1][j-1][i].z) * 0.25;

			xc=coor[k][j][i].x;
			yc=coor[k][j][i].y;
			zc=coor[k][j][i].z;

	            	double rx= (xc - ibm[ibi].cent_x[l]), ry = (yc - ibm[ibi].cent_y[l]), rz = (zc - ibm[ibi].cent_z[l]);

			Calculate_normal(csi[k][j][i], eta[k][j][i], zet[k][j][i], ni, nj, nk);

			double dhx_=ibm[ibi].dhx[l];
			double dhy_=ibm[ibi].dhy[l];
			double dhz_=ibm[ibi].dhz[l];

			r1=fabs(rx*ni[0]+ry*ni[1]+rz*ni[2])/dhx_; 
			r2=fabs(rx*nj[0]+ry*nj[1]+rz*nj[2])/dhy_; 
			r3=fabs(rx*nk[0]+ry*nk[1]+rz*nk[2])/dhz_; 

	          	vol_eul = aj[k][j][i];

			/*
			if (deltafunc == 1) dfunc = vol_eul * dfunc_2h(r1) * dfunc_2h(r2) * dfunc_2h(r3);
			if (deltafunc == 2) dfunc = vol_eul * dfunc_4h(r1) * dfunc_4h(r2) * dfunc_4h(r3);
			if (deltafunc == 3) dfunc = vol_eul * dfunc_s3h(r1) * dfunc_s3h(r2) * dfunc_s3h(r3);
			if (deltafunc == 4) dfunc = vol_eul * dfunc_s4h(r1) * dfunc_s4h(r2) * dfunc_s4h(r3);
			*/

			//dfunc = vol_eul * dfunc_nh(r1,3) * dfunc_nh(r2,3) * dfunc_nh(r3,3);
			dfunc = vol_eul * dfunc_s3h(r1) * dfunc_s3h(r2) * dfunc_s3h(r3);
			//dfunc = vol_eul * dfunc_nh(r1, dfunc_wd) * dfunc_nh(r2, dfunc_wd) * dfunc_nh(r3, dfunc_wd);

			//dfunc = vol_eul * dfunc_2h(r1) * dfunc_2h(r2) * dfunc_2h(r3);

	            	lf_eul[k][j][i].x += ibm[ibi].F_lagr_x[l] * dfunc * ibm[ibi].dA[l] * csi[k][j][i].x +
	                                 ibm[ibi].F_lagr_y[l] * dfunc * ibm[ibi].dA[l] * csi[k][j][i].y +
                                 	ibm[ibi].F_lagr_z[l] * dfunc * ibm[ibi].dA[l] * csi[k][j][i].z;

	            	lf_eul[k][j][i].y += ibm[ibi].F_lagr_x[l] * dfunc * ibm[ibi].dA[l] *  eta[k][j][i].x +
        	                         ibm[ibi].F_lagr_y[l] * dfunc * ibm[ibi].dA[l] *  eta[k][j][i].y +
                	                 ibm[ibi].F_lagr_z[l] * dfunc * ibm[ibi].dA[l] *  eta[k][j][i].z;

	              	lf_eul[k][j][i].z += ibm[ibi].F_lagr_x[l] * dfunc * ibm[ibi].dA[l] *  zet[k][j][i].x +
        	                         ibm[ibi].F_lagr_y[l] * dfunc * ibm[ibi].dA[l] *  zet[k][j][i].y +
                	                 ibm[ibi].F_lagr_z[l] * dfunc * ibm[ibi].dA[l] *  zet[k][j][i].z;
       
//			ibm[ibi].F_lagr_x[l]=1.00022323;
//			lf_eul[k][j][i].x+=ibm[ibi].F_lagr_x[l] ; 
//			lf_eul[k][j][i].y+=ibm[ibi].F_lagr_x[l] ; 
//			lf_eul[k][j][i].z+=ibm[ibi].F_lagr_x[l] ; 
	      	}
	}
	}


	for (k=lzs;k<lze;k++) 
	for (j=lys;j<lye;j++) 
	for (i=lxs;i<lxe;i++) {

		if (nvert[k][j][i]>0.1) {
                        lf_eul[k][j][i].x=0.0;
                        lf_eul[k][j][i].y=0.0;
                        lf_eul[k][j][i].z=0.0;
		}


		if (i==1) {
			lf_eul[k][j][i-1].x=0.0;
			lf_eul[k][j][i-1].y=0.0;
			lf_eul[k][j][i-1].z=0.0;
		}

		if (j==1) {
			lf_eul[k][j-1][i].x=0.0;
			lf_eul[k][j-1][i].y=0.0;
			lf_eul[k][j-1][i].z=0.0;
		}

		if (k==1) {
			lf_eul[k-1][j][i].x=0.0;
			lf_eul[k-1][j][i].y=0.0;
			lf_eul[k-1][j][i].z=0.0;
		}

		if (i==mx-2) {
			lf_eul[k][j][i+1].x=0.0;
			lf_eul[k][j][i+1].y=0.0;
			lf_eul[k][j][i+1].z=0.0;
		}

		if (j==my-2) {
			lf_eul[k][j+1][i].x=0.0;
			lf_eul[k][j+1][i].y=0.0;
			lf_eul[k][j+1][i].z=0.0;
		}

		if (k==mz-2) {
			lf_eul[k+1][j][i].x=0.0;
			lf_eul[k+1][j][i].y=0.0;
			lf_eul[k+1][j][i].z=0.0;
		}



	}
  
 	DMDAVecRestoreArray(fda, Coor, &coor);
  	DMDAVecRestoreArray(fda, user->lF_eul, &lf_eul);
  	DMDAVecRestoreArray(fda, user->lCsi,  &csi);
  	DMDAVecRestoreArray(fda, user->lEta,  &eta);
  	DMDAVecRestoreArray(fda, user->lZet,  &zet);
  	DMDAVecRestoreArray(da,  user->lAj,  &aj);

	DMDAVecRestoreArray(da, user->lNvert, &nvert);

  	DMLocalToGlobalBegin(fda, user->lF_eul, INSERT_VALUES, user->F_eul);
  	DMLocalToGlobalEnd(fda, user->lF_eul, INSERT_VALUES, user->F_eul);


  return(0);
}


// exporting the forces on the turbines actuator disk model
PetscErrorCode Calc_forces_rotor(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int bi)
{
  	PetscInt      	l, ibi;
  	double        	pi = 3.141592653589793;
  	PetscReal	A_Sum, F_Sum, P_Sum, U_Sum;


  	for (ibi=0; ibi<NumberOfTurbines; ibi++) {
    		A_Sum = 0.0; P_Sum = 0.0; U_Sum = 0.0, F_Sum=0.0;
		double nx=fsi[ibi].nx_tb, ny=fsi[ibi].ny_tb, nz=fsi[ibi].nz_tb;

    		for (l=0; l<ibm[ibi].n_elmt; l++) {

      			A_Sum += ibm[ibi].dA[l] ;

			double F_axis=ibm[ibi].F_lagr_x[l]*nx+ibm[ibi].F_lagr_y[l]*ny+ibm[ibi].F_lagr_z[l]*nz;
			double U_axis=ibm[ibi].U_lagr_x[l]*nx+ibm[ibi].U_lagr_y[l]*ny+ibm[ibi].U_lagr_z[l]*nz;

      			F_Sum += F_axis*ibm[ibi].dA[l] ;
      			P_Sum += F_axis*ibm[ibi].dA[l]*U_axis;

	      		U_Sum += U_axis*ibm[ibi].dA[l] ;

    		}
   
		U_Sum=U_Sum/A_Sum;

	    	int rank=0;
    		MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	    	if (!rank) {
      			FILE *f;
      			char filen[80];
	      		sprintf(filen, "Turbine_AD%2.2d_%2.2d",rotor_model,ibi);
      			if (ti==1) {
				f = fopen(filen, "w");
      				PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=\"time\", \"a\", \"F\", \"U\", \"P\" \n");
			} else f = fopen(filen, "a");

      			PetscFPrintf(PETSC_COMM_WORLD, f, "%d %le %le %le %le \n",ti, ibm[ibi].indf_axis,-F_Sum,U_Sum,-P_Sum);
	      		fclose(f);
	    	}


        	if ( ti % tiout == 0 /*ti == (ti/tiout) * tiout */) {
		        int rank=0;
        		MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
			int i;

			double dt = user->dt;
		        if (!rank) {
        		        FILE *f;
	                	char filen[80];
        		        sprintf(filen, "%s/ACDsurface%06d_%03d_nf.dat",path,ti,ibi);

		                f = fopen(filen, "w");
	
				int n_v=ibm[ibi].n_v; 	
				int n_elmt=ibm[ibi].n_elmt; 	

		                PetscFPrintf(PETSC_COMM_WORLD, f, "TITLE = \"Actuator disk mesh\" \n ");
			    	PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z,n_x,n_y,n_z\n");
		    		PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-6]=CELLCENTERED)\n", n_v, n_elmt);
                		PetscFPrintf(PETSC_COMM_WORLD, f, "STRANDID=0.1 SOLUTIONTIME=%le \n", ((double)ti)*dt);

		    		for (i=0; i<n_v; i++) {
	      				PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm[ibi].x_bp[i]);
		    		}
    				for (i=0; i<n_v; i++) {
		      			PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm[ibi].y_bp[i]);
		    		}
		    		for (i=0; i<n_v; i++) {	
		      			PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm[ibi].z_bp[i]);
		    		}
		    		for (i=0; i<n_elmt; i++) {
      					PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm[ibi].nf_x[i]);
		    		}
    				for (i=0; i<n_elmt; i++) {
		      			PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm[ibi].nf_y[i]);
    				}
		    		for (i=0; i<n_elmt; i++) {
      					PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm[ibi].nf_z[i]);
		    		}
    				for (i=0; i<n_elmt; i++) {
		      			PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
		    		}
 

                		fclose(f);

		        }
        	}
	}

        return false;
}

// exporting the forces on the turbines, actuator line model
PetscErrorCode Calc_forces_ACL(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int bi)
{
  	PetscInt      l, ibi;
  	double        pi = 3.141592653589793;
	PetscReal A_Sum = 0.0, P_Sum = 0.0, U_Sum = 0.0, F_Sum=0.0, M_Sum=0.0, Fx_Sum=0.0, Fy_Sum=0.0, Fz_Sum=0.0;
  	PetscReal F_z, P, r, rx, ry, rz; 

  	int rank=0;
  	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);


  	for (ibi=0; ibi<NumberOfTurbines; ibi++) {
		A_Sum = 0.0; P_Sum = 0.0; U_Sum = 0.0; F_Sum=0.0; M_Sum=0.0; Fx_Sum=0.0; Fy_Sum=0.0; Fz_Sum=0.0;


		double nx=fsi[ibi].nx_tb, ny=fsi[ibi].ny_tb, nz=fsi[ibi].nz_tb;

    		for (l=0; l<ibm[ibi].n_elmt; l++) {

      			A_Sum += ibm[ibi].dA[l] ;

			double F_axis=ibm[ibi].F_lagr_x[l]*nx+ibm[ibi].F_lagr_y[l]*ny+ibm[ibi].F_lagr_z[l]*nz;
			double U_axis=ibm[ibi].U_lagr_x[l]*nx+ibm[ibi].U_lagr_y[l]*ny+ibm[ibi].U_lagr_z[l]*nz;

      			F_Sum += F_axis*ibm[ibi].dA[l] ;
      			P_Sum += F_axis*ibm[ibi].dA[l]*U_axis;

	      		U_Sum += U_axis*ibm[ibi].dA[l] ;
	
				
      			Fx_Sum += ibm[ibi].F_lagr_x[l]*ibm[ibi].dA[l] ;
      			Fy_Sum += ibm[ibi].F_lagr_y[l]*ibm[ibi].dA[l] ;
      			Fz_Sum += ibm[ibi].F_lagr_z[l]*ibm[ibi].dA[l] ;

//      			r = sqrt( pow((ibm[ibi].cent_x[l]-fsi[ibi].x_c),2) + pow((ibm[ibi].cent_y[l]-fsi[ibi].y_c),2) + pow((ibm[ibi].cent_z[l]-fsi[ibi].z_c),2));	

      			rx = ibm[ibi].cent_x[l]-fsi[ibi].x_c;	
      			ry = ibm[ibi].cent_y[l]-fsi[ibi].y_c;	
      			rz = ibm[ibi].cent_z[l]-fsi[ibi].z_c;	

//			double nx = rx/r;
//			double ny = ry/r;
//			double nz = rz/r;
	
//      			rx = r*nx; ry = r*ny; rz = r*nz;

      			double M_x= ry*ibm[ibi].F_lagr_z[l] * ibm[ibi].dA[l] - rz*ibm[ibi].F_lagr_y[l] * ibm[ibi].dA[l];  
      			double M_y= rz*ibm[ibi].F_lagr_x[l] * ibm[ibi].dA[l] - rx*ibm[ibi].F_lagr_z[l] * ibm[ibi].dA[l];  
      			double M_z= rx*ibm[ibi].F_lagr_y[l] * ibm[ibi].dA[l] - ry*ibm[ibi].F_lagr_x[l] * ibm[ibi].dA[l];  

			M_Sum+=M_x*nx+M_y*ny+M_z*nz;

    		}

		U_Sum=U_Sum/A_Sum;

		//double angvel_axis=fsi[ibi].angvel_x*nx+fsi[ibi].angvel_y*ny+fsi[ibi].angvel_z*nz;

		double P_moment=M_Sum*fsi[ibi].angvel_axis;

	    	if (!rank) {
      			FILE *f;
      			char filen[80];
	      		sprintf(filen, "Turbine_AL%2.2d_%2.2d",rotor_model,ibi);
      			if (ti==1) {
				f = fopen(filen, "w");
      				PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=\"time\", \"angvel_axis\", \"F_axis\", \"Fx\", \"Fy\", \"Fz\", \"U\", \"M\", \"P_force\", \"P_moment\" \n");
			} else f = fopen(filen, "a");

      			PetscFPrintf(PETSC_COMM_WORLD, f, "%d %le %le %le %le %le %le %le %le %le \n",ti, fsi[ibi].angvel_axis,-F_Sum, Fx_Sum, Fy_Sum, Fz_Sum, U_Sum,-M_Sum,-P_Sum, -P_moment);
	      		fclose(f);
	    	}

  	}

	return(0);
}

// calculating the reference velocity for actuator line model
PetscErrorCode Uref_ACL(UserCtx *user, IBMNodes *ibm_ACL, IBMNodes *ibm_ACD, FSInfo *fsi_wt, int NumberOfObjects)
{
  	PetscInt      l, ibi;
  	double        pi = 3.141592653589793;
  	PetscReal     F_Sum, A_Sum, U_Sum, CT1, indf_a;

        int rank=0;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	Calc_U_lagr(user, ibm_ACD, fsi_wt, NumberOfObjects);

  	for (ibi=0; ibi<NumberOfObjects; ibi++) {

		double nx=fsi_wt[ibi].nx_tb, ny=fsi_wt[ibi].ny_tb, nz=fsi_wt[ibi].nz_tb;

		F_Sum=0.0;
    		for (l=0; l<ibm_ACL[ibi].n_elmt; l++) {
			double F_axis=ibm_ACL[ibi].F_lagr_x[l]*nx+ibm_ACL[ibi].F_lagr_y[l]*ny+ibm_ACL[ibi].F_lagr_z[l]*nz;
      			F_Sum += F_axis*ibm_ACL[ibi].dA[l] ;
		}


    		U_Sum = 0.0; A_Sum = 0.0; 
                for (l=0; l<ibm_ACD[ibi].n_elmt; l++) {
			double U_axis=ibm_ACD[ibi].U_lagr_x[l]*nx+ibm_ACD[ibi].U_lagr_y[l]*ny+ibm_ACD[ibi].U_lagr_z[l]*nz;
	      		U_Sum += U_axis*ibm_ACD[ibi].dA[l] ;
                        A_Sum += ibm_ACD[ibi].dA[l] ;
                }

                U_Sum /= A_Sum;
		
		ibm_ACL[ibi].U_ref = U_Sum; // / (1.0 - indf_a);

		fsi_wt[ibi].angvel_axis=ibm_ACL[ibi].Tipspeedratio*U_Sum/(r_rotor/reflength_wt);

		//fsi_wt[ibi].angvel_x = angvel_axis*nx;	
		//fsi_wt[ibi].angvel_y = angvel_axis*ny;	
		//fsi_wt[ibi].angvel_z = angvel_axis*nz;	
		

                double CT = -2.0*F_Sum/(A_Sum*U_Sum*U_Sum+1.e-11);
		if (CT>1) CT=1.;
                indf_ax = 0.5*(1-sqrt(1-CT));


    		if (!rank) {
		      	FILE *f;
      			char filen[80];
   			sprintf(filen, "IndfFromAL=%2.2dibi=%2.2d",rotor_model, ibi);
      			if (ti==1) {
				f = fopen(filen, "w");
      				PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=\"time\", \"a\", \"CT\", \"F\", \"U\", \"angvel_axis\"\n");
			} else f = fopen(filen, "a");

      			PetscFPrintf(PETSC_COMM_WORLD, f, "%d %le %le %le %le %le\n",ti, indf_ax, CT, -F_Sum, U_Sum, fsi_wt[ibi].angvel_axis);
      			fclose(f);
	    	}
  	}

        return false;
}
// Translate the coordinates of the disk 1D upstream. Assume z is the streamwise direction from negtive to positive
PetscErrorCode Trans1DUp_XYZ(IBMNodes *ibm, int NumberOfObjects)
{

	int ibi, i;
	
  	for (ibi=0; ibi<NumberOfObjects; ibi++) {

                for (i=0; i<ibm[ibi].n_v; i++){
                        ibm[ibi].z_bp[i]-=2.0*r_rotor/reflength_wt;
                }

                for (i=0; i<ibm[ibi].n_elmt; i++){
                        ibm[ibi].cent_z[i]-=2.0*r_rotor/reflength_wt;
                }
	}

        return false;
}


// update the coordinates for grid nodes of actuator disks/lines for moving frame
PetscErrorCode UpdateXYZ_MoveFrame(UserCtx *user, IBMNodes *ibm, FSInfo *fsi, int NumberOfObjects)
{

	int ibi, i;

        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	
  	for (ibi=0; ibi<NumberOfObjects; ibi++) {
		fsi[ibi].x_c=fsi[ibi].x_c0+ti*user->dt*u_frame;
		fsi[ibi].y_c=fsi[ibi].y_c0+ti*user->dt*v_frame;
		fsi[ibi].z_c=fsi[ibi].z_c0+ti*user->dt*w_frame;


		if (ii_periodicWT) {
                	double Lp_x=Nx_WT*Sx_WT;
			if ((fsi[ibi].x_c-xmin)<1.e-9) {
                                int np_x = (int)((xmax-fsi[ibi].x_c+1.e-9)/Lp_x);
                                fsi[ibi].x_c+=Lp_x*np_x;
			}
			if ((fsi[ibi].x_c-xmax)>-1.e-9) {
                                int np_x = (int)((fsi[ibi].x_c-xmin+1.e-9)/Lp_x);
				fsi[ibi].x_c-=Lp_x*np_x;
			}
		}

		if (jj_periodicWT) {
                	double Lp_y=Ny_WT*Sy_WT;
                        if ((fsi[ibi].y_c-ymin)<1.e-9) {
				int np_y = (int)((ymax-fsi[ibi].y_c+1.e-9)/Lp_y);
				fsi[ibi].y_c+=Lp_y*np_y;
			}
                        if ((fsi[ibi].y_c-ymax)>-1.e-9) {
                                int np_y = (int)((fsi[ibi].y_c-ymin+1.e-9)/Lp_y);
				fsi[ibi].y_c-=Lp_y*np_y;
			}
		}

		if (kk_periodicWT) {
                	double Lp_z=Nz_WT*Sz_WT;
                        if ((fsi[ibi].z_c-zmin)<1.e-9) {
				int np_z = (int)((zmax-fsi[ibi].z_c+1.e-9)/Lp_z);
				fsi[ibi].z_c+=Lp_z*np_z;
			}
                        if ((fsi[ibi].z_c-zmax)>-1.e-9) {
                                int np_z = (int)((fsi[ibi].z_c-zmin+1.e-9)/Lp_z);
				fsi[ibi].z_c-=Lp_z*np_z;
			}
		}

//			if (!rank) printf("z_c: %d %le %le \n", ibi, fsi[ibi].z_c0, fsi[ibi].z_c);
                for (i=0; i<ibm[ibi].n_v; i++){
//			printf("z: %d %le %le \n", rank, ibm[ibi].z_bp[i], ibm[ibi].z_bp0[i]+fsi[ibi].z_c0);
//			int a;
//			cout << "Hi \n"; 
//			cin >> a;
                        ibm[ibi].x_bp[i]=ibm[ibi].x_bp_i[i]+fsi[ibi].x_c;
                        ibm[ibi].y_bp[i]=ibm[ibi].y_bp_i[i]+fsi[ibi].y_c;
                        ibm[ibi].z_bp[i]=ibm[ibi].z_bp_i[i]+fsi[ibi].z_c;
                }

                for (i=0; i<ibm[ibi].n_elmt; i++){
                        int n1e = ibm[ibi].nv1[i], n2e = ibm[ibi].nv2[i], n3e = ibm[ibi].nv3[i];
                        if (rotor_model == 3) {
                                ibm[ibi].cent_x[i]= (ibm[ibi].x_bp[n1e]+ibm[ibi].x_bp[n2e])/2.;
                                ibm[ibi].cent_y[i]= (ibm[ibi].y_bp[n1e]+ibm[ibi].y_bp[n2e])/2.;
                                ibm[ibi].cent_z[i]= (ibm[ibi].z_bp[n1e]+ibm[ibi].z_bp[n2e])/2.;
                        }
                        if (rotor_model == 1 || rotor_model == 2) {
                                ibm[ibi].cent_x[i]= (ibm[ibi].x_bp[n1e]+ibm[ibi].x_bp[n2e]+ibm[ibi].x_bp[n3e])/3.;
                                ibm[ibi].cent_y[i]= (ibm[ibi].y_bp[n1e]+ibm[ibi].y_bp[n2e]+ibm[ibi].y_bp[n3e])/3.;
                                ibm[ibi].cent_z[i]= (ibm[ibi].z_bp[n1e]+ibm[ibi].z_bp[n2e]+ibm[ibi].z_bp[n3e])/3.;
                        }

                }


	}

	if (!rank) {
		FILE *fd;
              	char str[256];
               	sprintf(str, "./CenterWT.dat");
               	fd = fopen(str, "w");

      		for (ibi=0;ibi<NumberOfObjects;ibi++) {
                       	fprintf(fd, "%le %le %le \n", fsi[ibi].x_c, fsi[ibi].y_c, fsi[ibi].z_c);
               	}
        		fclose(fd);
	}



        return false;
}


/* ==================================================================================             */
PetscErrorCode rotor_Rot(FSInfo *FSinfo, IBMNodes *ibm, PetscReal dt, int ibi)
{

  	int n_v = ibm->n_v, n_elmt = ibm->n_elmt;
  	PetscReal  x_c=FSinfo->x_c, y_c=FSinfo->y_c, z_c=FSinfo->z_c;
  	int i,j,k;
  	int n1e, n2e, n3e;
  	PetscReal dx12, dy12, dz12, dx13, dy13, dz13, dr;
  	PetscReal rx,ry,rz;
  	int n1;
 
	double rot_angle;

	if (!FixTipSpeedRatio) {
	char angvel_file[400];
	sprintf(angvel_file, "%s/Angvel%3.3d", path, ibi);
	FILE *fp=fopen(angvel_file, "r");

	if(fp!=NULL) {
		fscanf(fp, "%le\n", &(FSinfo->angvel_axis));
	} else {
		PetscReal angvel_tb;
		PetscOptionsGetReal(PETSC_NULL, "-angvel_tb", &angvel_tb, PETSC_NULL);
		FSinfo->angvel_axis=angvel_tb;
	}
	}

	PetscPrintf(PETSC_COMM_WORLD, "angvel_tb=%f\n", FSinfo->angvel_axis);

	
	for (i=0; i<n_v; i++) {
                // rotate_xyz (ti+ti_lastsave, dt, angvel1, x_c, y_c, z_c, ibm->x_bp0[i], ibm->y_bp0[i], ibm->z_bp0[i], &ibm->x_bp[i], &ibm->y_bp[i], &ibm->z_bp[i], &rot_angle);
		
		Cmpnts p, r, q; 
		p.x=ibm->x_bp[i]-FSinfo->x_c;
		p.y=ibm->y_bp[i]-FSinfo->y_c;
		p.z=ibm->z_bp[i]-FSinfo->z_c;

		r.x=FSinfo->nx_tb;	
		r.y=FSinfo->ny_tb;	
		r.z=FSinfo->nz_tb;	

		double theta=FSinfo->angvel_axis*dt;

		q=ArbitraryRotate(p,theta,r);

		ibm->u[i].x = (q.x - p.x) / dt;
		ibm->u[i].y = (q.y - p.y) / dt;
		ibm->u[i].z = (q.z - p.z) / dt;

		ibm->x_bp[i]=q.x+FSinfo->x_c;
		ibm->y_bp[i]=q.y+FSinfo->y_c;
		ibm->z_bp[i]=q.z+FSinfo->z_c;
		
		//double x1, y1, z1;
		//double x2, y2, z2;
		//double tmp, eps=1.e-6;
		//rotate_xyz (ti+ti_lastsave-1, dt, angvel1, x_c, y_c, z_c, ibm->x_bp0[i], ibm->y_bp0[i], ibm->z_bp0[i], &x1, &y1, &z1, &tmp);
		//rotate_xyz (ti+ti_lastsave+1, dt, angvel1, x_c, y_c, z_c, ibm->x_bp0[i], ibm->y_bp0[i], ibm->z_bp0[i], &x2, &y2, &z2, &tmp);
		//ibm->u[i].x = (x2 - x1) / dt * 0.5;
		//ibm->u[i].y = (y2 - y1) / dt * 0.5;
		//ibm->u[i].z = (z2 - z1) / dt * 0.5;

//		PetscPrintf(PETSC_COMM_WORLD, "Calc Vel of TUrbine  %d Elmt %d x2 %le x1 %le \n", ibi,i,x2,x1);		
		//PetscPrintf(PETSC_COMM_WORLD, "Speed of Turbine %d Elmt %d u.x %le u.y %le u.z %le\n", ibi,i,ibm->u[i].x,ibm->u[i].y,ibm->u[i].z);		
	}
	
	//if(rotdir==0) FSinfo->S_ang_n[0] = rot_angle;
	//if(rotdir==1) FSinfo->S_ang_n[2] = rot_angle;
	//if(rotdir==2) FSinfo->S_ang_n[4] = rot_angle;
	
	//if(rotdir==0) FSinfo->S_ang_n[1] = angvel1;
	//if(rotdir==1) FSinfo->S_ang_n[3] = angvel1;
	//if(rotdir==2) FSinfo->S_ang_n[5] = angvel1;
	

	if (rotor_model == 2) {
     
  		for (i=0; i<n_elmt; i++) {

    			n1e = ibm->nv1[i]; n2e =ibm->nv2[i]; n3e =ibm->nv3[i];
    			dx12 = ibm->x_bp[n2e] - ibm->x_bp[n1e]; 
    			dy12 = ibm->y_bp[n2e] - ibm->y_bp[n1e]; 
    			dz12 = ibm->z_bp[n2e] - ibm->z_bp[n1e]; 
    
    			dx13 = ibm->x_bp[n3e] - ibm->x_bp[n1e]; 
    			dy13 = ibm->y_bp[n3e] - ibm->y_bp[n1e]; 
    			dz13 = ibm->z_bp[n3e] - ibm->z_bp[n1e]; 
    
    			ibm->nf_x[i] = dy12 * dz13 - dz12 * dy13;
    			ibm->nf_y[i] = -dx12 * dz13 + dz12 * dx13;
    			ibm->nf_z[i] = dx12 * dy13 - dy12 * dx13;

    			dr = sqrt(ibm->nf_x[i]*ibm->nf_x[i] + ibm->nf_y[i]*ibm->nf_y[i] + 
	      		ibm->nf_z[i]*ibm->nf_z[i]);
    
    			ibm->nf_x[i] /=dr; ibm->nf_y[i]/=dr; ibm->nf_z[i]/=dr;
    
    			if ((((1.-ibm->nf_z[i])<=1e-6 )&&((-1.+ibm->nf_z[i])<1e-6))||
			    (((ibm->nf_z[i]+1.)<=1e-6 )&&((-1.-ibm->nf_z[i])<1e-6))) {
      				ibm->ns_x[i] = 1.;
      				ibm->ns_y[i] = 0.;
      				ibm->ns_z[i] = 0 ;

      				// nt = ns x nf
      				ibm->nt_x[i] = 0.;
      				ibm->nt_y[i] = 1.;
      				ibm->nt_z[i] = 0.;
      			} else {
      				ibm->ns_x[i] =  ibm->nf_y[i]/ sqrt(ibm->nf_x[i]*ibm->nf_x[i] + ibm->nf_y[i]*ibm->nf_y[i]);
      				ibm->ns_y[i] = -ibm->nf_x[i]/ sqrt(ibm->nf_x[i]*ibm->nf_x[i] + ibm->nf_y[i]*ibm->nf_y[i]);
      				ibm->ns_z[i] = 0 ;

      				// nt = ns x nf
      				ibm->nt_x[i] = -ibm->nf_x[i]*ibm->nf_z[i]/ sqrt(ibm->nf_x[i]*ibm->nf_x[i] + ibm->nf_y[i]*ibm->nf_y[i]);
      				ibm->nt_y[i] = -ibm->nf_y[i]*ibm->nf_z[i]/ sqrt(ibm->nf_x[i]*ibm->nf_x[i] + ibm->nf_y[i]*ibm->nf_y[i]);
      				ibm->nt_z[i] = sqrt(ibm->nf_x[i]*ibm->nf_x[i] + ibm->nf_y[i]*ibm->nf_y[i]);
      			}

    			ibm->dA[i] = dr/2.;

    			ibm->cent_x[i]= (ibm->x_bp[n1e]+ibm->x_bp[n2e]+ibm->x_bp[n3e])/3.;
    			ibm->cent_y[i]= (ibm->y_bp[n1e]+ibm->y_bp[n2e]+ibm->y_bp[n3e])/3.;
    			ibm->cent_z[i]= (ibm->z_bp[n1e]+ibm->z_bp[n2e]+ibm->z_bp[n3e])/3.;

  		}
  
	}


// 	for ACL, the nf_x, nf_y, nf_z denote the direction of the actuator line. and dA denotes the length of each element.
	int nb;

	int n_elmt_1 = (ibm->n_elmt)/num_blade;

//      PetscPrintf(PETSC_COMM_WORLD, "rotormodel %d \n", rotor_model);
	if (rotor_model == 3) {
      		for (nb=0; nb<num_blade; nb++) {

                for (j=0; j<n_elmt_1; j++) {
                        i = nb * n_elmt_1 + j;
      
      			n1e = ibm->nv1[i]; n2e = ibm->nv2[i]; 
      			dx12 = ibm->x_bp[n2e] - ibm->x_bp[n1e];
      			dy12 = ibm->y_bp[n2e] - ibm->y_bp[n1e];
      			dz12 = ibm->z_bp[n2e] - ibm->z_bp[n1e];
      
      			ibm->nf_x[i] = dx12;
      			ibm->nf_y[i] = dy12;
      			ibm->nf_z[i] = dz12;
      
      			dr = sqrt(ibm->nf_x[i]*ibm->nf_x[i] + ibm->nf_y[i]*ibm->nf_y[i] + ibm->nf_z[i]*ibm->nf_z[i]);
      
      			ibm->nf_x[i] /=dr; ibm->nf_y[i]/=dr; ibm->nf_z[i]/=dr;
      
			ibm->ns_x[i] = 0.;     
			ibm->ns_y[i] = 0.;     
			ibm->ns_z[i] = 0. ;
	
			ibm->nt_x[i] = 0.;
			ibm->nt_y[i] = 0.;
			ibm->nt_z[i] = 0.;
     
			ibm->dA[i] = dr;

      			ibm->cent_x[i]= (ibm->x_bp[n1e]+ibm->x_bp[n2e])/2.;
      			ibm->cent_y[i]= (ibm->y_bp[n1e]+ibm->y_bp[n2e])/2.;
      			ibm->cent_z[i]= (ibm->z_bp[n1e]+ibm->z_bp[n2e])/2.;

	
      		}
      
    		}

	}

	if (rotor_model == 2) {
        	if ( ti % tiout == 0 /*ti == (ti/tiout) * tiout */) {
		        int rank=0;
        		MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
			int i;

		        if (!rank) {
        		        FILE *f;
	                	char filen[80];
        		        sprintf(filen, "%s/Bladesurface%06d_%03d_nf.dat",path,ti,ibi);

		                f = fopen(filen, "w");
	
				int n_v=ibm->n_v; 	
				int n_elmt=ibm->n_elmt; 	

		                PetscFPrintf(PETSC_COMM_WORLD, f, "TITLE = \"Actuator disk mesh\" \n ");
			    	PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z,n_x,n_y,n_z\n");
		    		PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-6]=CELLCENTERED)\n", n_v, n_elmt);
                		PetscFPrintf(PETSC_COMM_WORLD, f, "STRANDID=0.1 SOLUTIONTIME=%le \n", ((double)ti)*dt);

		    		for (i=0; i<n_v; i++) {
	      				PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm->x_bp[i]);
		    		}
    				for (i=0; i<n_v; i++) {
		      			PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm->y_bp[i]);
		    		}
		    		for (i=0; i<n_v; i++) {	
		      			PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm->z_bp[i]);
		    		}
		    		for (i=0; i<n_elmt; i++) {
      					PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm->nf_x[i]);
		    		}
    				for (i=0; i<n_elmt; i++) {
		      			PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm->nf_y[i]);
    				}
		    		for (i=0; i<n_elmt; i++) {
      					PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm->nf_z[i]);
		    		}
    				for (i=0; i<n_elmt; i++) {
		      			PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
		    		}
 

                		fclose(f);

		        }
        	}

	}

	if (rotor_model == 3) {
		if ( ti % tiout == 0 /*ti == (ti/tiout) * tiout */) {
	    	int rank=0;
	    	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

		if (!rank) {
        	        FILE *f;
	    		char filen[80];  
        	        sprintf(filen, "%s/line%06d_%03d_nf.dat",path,ti,ibi);
	                f = fopen(filen, "w");

			PetscFPrintf(PETSC_COMM_WORLD, f, "TITLE = \"Actuator line mesh\" \n ");
			PetscFPrintf(PETSC_COMM_WORLD, f, "VARIABLES = \"X\", \"Y\", \"Z\" \n");
			PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T=\"P_1\", DATAPACKING=POINT, NODES=%d, ELEMENTS=%d, ZONETYPE=FELINESEG \n", ibm->n_v, ibm->n_elmt);
			PetscFPrintf(PETSC_COMM_WORLD, f, "STRANDID=0.1 SOLUTIONTIME=%le \n", ((double)ti)*dt);

	                for (i=0; i<ibm->n_v; i++) {
        	                PetscFPrintf(PETSC_COMM_WORLD, f, "%le %le %le \n", ibm->x_bp[i], ibm->y_bp[i], ibm->z_bp[i]);
	                }

			for (i=0; i<ibm->n_elmt; i++) {
				PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d \n", ibm->nv1[i]+1, ibm->nv2[i]+1);
			}


	                fclose(f);


        	        sprintf(filen, "%s/acldata%03d.grd",path,ibi);
	                f = fopen(filen, "w");
			int ii=0;
			int n_nodes=ibm->n_v/num_blade;
	                for (i=0; i<num_blade; i++) {
				PetscFPrintf(PETSC_COMM_WORLD, f, "%d \n", n_nodes);
		                for (j=0; j<n_nodes; j++) {				
					ii=i*n_nodes+j;
					PetscFPrintf(PETSC_COMM_WORLD, f, "%le %le %le \n", ibm->x_bp[ii]-x_c, ibm->y_bp[ii]-y_c, ibm->z_bp[ii]-z_c);
				}
			}
	                fclose(f);
	  	}
	 	} 
	}

  	return(0);
}


double dfunc_4h(double r)
{
  if (fabs(r) <= 1.0) {
    return (3.0-2.0*fabs(r)+sqrt(1.0+4.0*fabs(r)-4.0*r*r))/8.0;
  } else if (fabs(r) >= 1.0 && fabs(r) <= 2.0) {
    return (5.0-2.0*fabs(r)-sqrt(-7.0+12.0*fabs(r)-4.0*r*r))/8.0;
  } else {
    return 0.0;
  }
}

double dfunc_s3h(double r)
{
  if (fabs(r) <= 1.0) {
    return 17.0/48.0 + sqrt(3.0)*3.14159265/108.0 + fabs(r)/4.0 - r*r/4.0 + (1.0-2.0*fabs(r))*sqrt(-12.0*r*r+12.0*fabs(r)+1.0)/16.0 - sqrt(3.0)*asin(sqrt(3.0)*(2.0*fabs(r)-1.0)/2.0)/12.0;
  } else if (fabs(r) >= 1.0 && fabs(r) <= 2.0) {
    return 55.0/48.0 - sqrt(3.0)*3.14159265/108.0 - 13.0*fabs(r)/12.0 + r*r/4.0 + (2.0*fabs(r)-3.0)*sqrt(-12.0*r*r+36.0*fabs(r)-23.0)/48.0 + sqrt(3.0)*asin(sqrt(3.0)*(2.0*fabs(r)-3.0)/2.0)/36.0;
  } else { 
    return 0.0;
  }

}


double dfunc_2h(double r)
{
  if (fabs(r) < 1.0) {
    return 1.0-fabs(r);
  } else {
    return 0.0;
  }
}


double dfunc_s4h(double r)
{

  if (fabs(r) <= 0.5) {
    return 3.0/8.0+3.14159265/32.0-pow(r,2)/4.0;
  } else if (fabs(r) >= 0.5 && fabs(r) <= 1.5) {
    return 1.0/4.0+(1.0-fabs(r))*sqrt(-2.0+8.0*fabs(r)-4.0*pow(r,2))/8.0-asin(sqrt(2.0)*(fabs(r)-1.0))/8.0;
  } else if (fabs(r) >= 1.5 && fabs(r) <= 2.5) {
    return
17.0/16.0-3.14159265/64.0-3.0*fabs(r)/4.0+pow(r,2)/8.0+(fabs(r)-2.0)*sqrt(-14.0+16.0*fabs(r)-4.0*pow(r,2))/16.0+asin(sqrt(2.0)*(fabs(r)-2.0))/16.0;
  } else {
    return 0.0;
  }

}


Cmpnts ArbitraryRotate(Cmpnts p,double theta,Cmpnts r) 
{
   	Cmpnts q = {0.0,0.0,0.0};
   	double costheta,sintheta;
	
	double rr=sqrt(r.x*r.x+r.y*r.y+r.z*r.z)+1.0e-11;
	r.x=r.x/rr; r.y=r.y/rr; r.z=r.z/rr;

   	costheta = cos(theta);
   	sintheta = sin(theta);

   	q.x += (costheta + (1 - costheta) * r.x * r.x) * p.x;
   	q.x += ((1 - costheta) * r.x * r.y - r.z * sintheta) * p.y;
   	q.x += ((1 - costheta) * r.x * r.z + r.y * sintheta) * p.z;

   	q.y += ((1 - costheta) * r.x * r.y + r.z * sintheta) * p.x;
   	q.y += (costheta + (1 - costheta) * r.y * r.y) * p.y;
   	q.y += ((1 - costheta) * r.y * r.z - r.x * sintheta) * p.z;

   	q.z += ((1 - costheta) * r.x * r.z - r.y * sintheta) * p.x;
   	q.z += ((1 - costheta) * r.y * r.z + r.x * sintheta) * p.y;
   	q.z += (costheta + (1 - costheta) * r.z * r.z) * p.z;

   	return(q);
}



void rotate_ArbitraryAxis (double dt, double angvel, double x_c, double y_c, double z_c, double x_bp0, double y_bp0, double z_bp0, double *x_bp, double *y_bp, double *z_bp, double *rot_angle)
{
	int i;
	if(rotdir==0) { // rotate around x-axis
		double rot_x = angvel * dt * ti;
		
		*x_bp = x_bp0;
		*y_bp = y_c + (y_bp0-y_c)*cos(rot_x) - (z_bp0-z_c)*sin(rot_x);
		*z_bp = z_c + (y_bp0-y_c)*sin(rot_x) + (z_bp0-z_c)*cos(rot_x);
		
		*rot_angle = rot_x;
	}
	else if(rotdir==1) { // rotate around y-axis
		double rot_y = angvel * dt * ti;

		*y_bp = y_bp0;
		*z_bp = z_c + (z_bp0-z_c)*cos(rot_y) - (x_bp0-x_c)*sin(rot_y);
		*x_bp = x_c + (z_bp0-z_c)*sin(rot_y) + (x_bp0-x_c)*cos(rot_y);
		
		*rot_angle = rot_y;
	}
	else { // rotate around z-axis
		double rot_z = angvel * dt * ti;
		
		*z_bp = z_bp0;
		*x_bp = x_c + (x_bp0-x_c)*cos(rot_z) - (y_bp0-y_c)*sin(rot_z);
		*y_bp = y_c + (x_bp0-x_c)*sin(rot_z) + (y_bp0-y_c)*cos(rot_z);
		
		
		*rot_angle = rot_z;
	}
}


double dfunc_nh(double r, double n)
{

  if (fabs(r) < n) {
    return (1.0/n-fabs(r)/pow(n,2));
  } else {
    return 0.0;
  }
}


