#include "variables.h"
extern PetscReal CMx_c,CMy_c,CMz_c, L_dim;
//extern int  cop, wing;
void str_to_buffer(char *str, std::vector<char> &large_buffer) //seokkoo
{
	char buffer[256];
	sprintf(buffer, str);
	int len = strlen(buffer), old_size = large_buffer.size();
	large_buffer.resize( old_size + len );
	for(int i=0; i<len; i++) large_buffer[old_size+i] = buffer[i];
}

PetscErrorCode ibm_surface_out(IBMNodes *ibm, int ti,
			       int ibi)
{
  int rank,i;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (!rank) {
    if (ti == ti) {
      FILE *f;
      char filen[80];
      sprintf(filen, "surface%06d_%2.2d.dat",ti,ibi);
      f = fopen(filen, "w");
	    
		int N_block=100*1024*1024; // 100Mb
		char str[256];
		char carriage_return = '\n';
		std::vector<char> large_buffer;
	    
      PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z,u_x,u_y,u_z,n_x,n_y,n_z,nt_x,nt_y,nt_z\n");
      PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-6]=NODAL,[7-12]=CELLCENTERED)\n", ibm->n_v, ibm->n_elmt);

      for (i=0; i<ibm->n_v; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->x_bp[i]);
      }
      for (i=0; i<ibm->n_v; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->y_bp[i]);
      }
      for (i=0; i<ibm->n_v; i++) {	
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->z_bp[i]);
      }
      for (i=0; i<ibm->n_v; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->u[i].x);
      }
      for (i=0; i<ibm->n_v; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->u[i].y);
      }
      for (i=0; i<ibm->n_v; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->u[i].z);
      }
      for (i=0; i<ibm->n_elmt; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_x[i]);
      }
      for (i=0; i<ibm->n_elmt; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_y[i]);
      }
      for (i=0; i<ibm->n_elmt; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_z[i]);
      }
      for (i=0; i<ibm->n_elmt; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_x[i]);
      }
      for (i=0; i<ibm->n_elmt; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_y[i]);
      }
      for (i=0; i<ibm->n_elmt; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_z[i]);
      }
      for (i=0; i<ibm->n_elmt; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
      }
      fclose(f);
    } 
  }
  return(0);
}

PetscErrorCode ibm_read_ucd(IBMNodes *ibm, int ibi)
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

  //PetscReal cl = 1.;
  //cl=1./L_dim;
  //PetscOptionsGetReal(PETSC_NULL, "-char_length_ibm", &cl, PETSC_NULL);     

		PetscReal cl = 1.;

		PetscOptionsGetReal(PETSC_NULL, "-chact_leng", &cl, PETSC_NULL);


  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if(!rank) { // root processor read in the data
    FILE *fd;
    PetscPrintf(PETSC_COMM_SELF, "READ ibmdata\n");
    char filen[80];  
    sprintf(filen,"%s/ibmdata%2.2d" , path, ibi);
 
    fd = fopen(filen, "r"); 
    if (!fd) printf("Cannot open %s !!\n", filen),exit(1);
    else printf("Opened %s !\n", filen);

    n_v =0;

    if (fd) {
      strcpy(string, "#");
      while (string[0] == '#')
        fgets(string, 128, fd);
      
      sscanf(string, "%i %i %i %i %i",&n_v,&n_elmt,&ii,&ii,&ii);
      PetscPrintf(PETSC_COMM_SELF, "number of nodes & elements %d %d\n",n_v, n_elmt);
      
      ibm->n_v = n_v;
      ibm->n_elmt = n_elmt;      
      
      MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
      //    PetscPrintf(PETSC_COMM_WORLD, "nv, %d %e \n", n_v, xt);
      PetscMalloc(n_v*sizeof(PetscReal), &x_bp);
      PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
      PetscMalloc(n_v*sizeof(PetscReal), &z_bp);
      
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
      
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
	
	x_bp[i] = x_bp[i]/cl + CMx_c;//0.25 ;// 24.;	
	y_bp[i] = y_bp[i]/cl + CMy_c ;//+ ibi*2.;//5.;//2.;//8.;//6.;//2.   ;// 24.;
	z_bp[i] = z_bp[i]/cl + CMz_c ;//+ ibi*2.;//2.;//8.;//15.;//2.   ;// 24.;
	
	ibm->x_bp[i] = x_bp[i];
	ibm->y_bp[i] = y_bp[i];
	ibm->z_bp[i] = z_bp[i];

	ibm->x_bp0[i] = x_bp[i];
	ibm->y_bp0[i] = y_bp[i];
	ibm->z_bp0[i] = z_bp[i];

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

/*       ibm->x_bp0 = x_bp; ibm->y_bp0 = y_bp; ibm->z_bp0 = z_bp; */
/*       ibm->x_bp_o = x_bp; ibm->y_bp_o = y_bp; ibm->z_bp_o = z_bp; */

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
      
       /** seokkoo **/
      PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->shear);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->mean_shear);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress1);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress2);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress3);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->pressure);
      PetscMalloc(n_elmt*sizeof(Cmpnts), &ibm->rel_velocity);

      for (i=0; i<n_elmt; i++) {

	fscanf(fd, "%i %i %s %i %i %i\n", &ii,&ii, ss, nv1+i, nv2+i, nv3+i);
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
/*       if (fabs(nf_x[i])<.5) */
/* 	nf_x[i]=0.; */

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

 /*    PetscFree(dA); */
/*     PetscFree(nf_x);PetscFree(nf_y);PetscFree(nf_z); */
/*     PetscFree(nt_x);PetscFree(nt_y);PetscFree(nt_z); */
/*     PetscFree(ns_x);PetscFree(ns_y);PetscFree(ns_z); */
/*     PetscFree(nv1);PetscFree(nv2);PetscFree(nv3); */
/*     PetscFree(x_bp);PetscFree(y_bp);PetscFree(z_bp); */

    int ti=0;
    FILE *f;
    //char filen[80];
    sprintf(filen, "surface_nf%3.3d_%2.2d.dat",ti,ibi);
    f = fopen(filen, "w");
    
		int N_block=100*1024*1024; // 100Mb
		char str[256];
		char carriage_return = '\n';
		std::vector<char> large_buffer;
		
    //PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z,n_x,n_y,n_z,nt_x,nt_y,nt_z,ns_x,ns_y,ns_z\n");
    //PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-12]=CELLCENTERED)\n", n_v, n_elmt);
    
		sprintf(str, "Variables=x,y,z,n_x,n_y,n_z,nt_x,nt_y,nt_z,ns_x,ns_y,ns_z");
		str_to_buffer(str, large_buffer);
		large_buffer.push_back(carriage_return);
		
		sprintf(str, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-12]=CELLCENTERED)", n_v, n_elmt);
		str_to_buffer(str, large_buffer);
		large_buffer.push_back(carriage_return);
		
		for (i=0; i<n_v; i++) {
			//PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->x_bp[i]);
			sprintf(str, "%e ", ibm->x_bp[i]);
			str_to_buffer(str, large_buffer);
			if( (i+1)%10==0 || i==n_v-1) large_buffer.push_back(carriage_return);
			
			if(large_buffer.size()>N_block) {
				fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
				large_buffer.resize(0);
			}
		}
		for (i=0; i<n_v; i++) {
			//PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->y_bp[i]);
			sprintf(str, "%e ", ibm->y_bp[i]);
			str_to_buffer(str, large_buffer);
			if( (i+1)%10==0 || i==n_v-1) large_buffer.push_back(carriage_return);
			
			if(large_buffer.size()>N_block) {
				fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
				large_buffer.resize(0);
			}
		}
		for (i=0; i<n_v; i++) {	
			//PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->z_bp[i]);
			sprintf(str, "%e ", ibm->z_bp[i]);
			str_to_buffer(str, large_buffer);
			if( (i+1)%10==0 || i==n_v-1) large_buffer.push_back(carriage_return);
			
			if(large_buffer.size()>N_block) {
				fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
				large_buffer.resize(0);
			}
		}
		for (i=0; i<n_elmt; i++) {
			//PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_x[i]);
			sprintf(str, "%e ", ibm->nf_x[i]);
			str_to_buffer(str, large_buffer);
			if( (i+1)%10==0 || i==n_elmt-1) large_buffer.push_back(carriage_return);
			
			if(large_buffer.size()>N_block) {
				fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
				large_buffer.resize(0);
			}
		}
		for (i=0; i<n_elmt; i++) {
			//PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_y[i]);
			sprintf(str, "%e ", ibm->nf_y[i]);
			str_to_buffer(str, large_buffer);
			if( (i+1)%10==0 || i==n_elmt-1) large_buffer.push_back(carriage_return);
			
			if(large_buffer.size()>N_block) {
				fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
				large_buffer.resize(0);
			}
		}
		for (i=0; i<n_elmt; i++) {
			//PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nf_z[i]);
			sprintf(str, "%e ", ibm->nf_z[i]);
			str_to_buffer(str, large_buffer);
			if( (i+1)%10==0 || i==n_elmt-1) large_buffer.push_back(carriage_return);
			
			if(large_buffer.size()>N_block) {
				fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
				large_buffer.resize(0);
			}
		}
		for (i=0; i<n_elmt; i++) {
			//PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_x[i]);
			sprintf(str, "%e ", ibm->nt_x[i]);
			str_to_buffer(str, large_buffer);
			if( (i+1)%10==0 || i==n_elmt-1) large_buffer.push_back(carriage_return);
			
			if(large_buffer.size()>N_block) {
				fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
				large_buffer.resize(0);
			}
		}
		for (i=0; i<n_elmt; i++) {
			//PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_y[i]);
			sprintf(str, "%e ", ibm->nt_y[i]);
			str_to_buffer(str, large_buffer);
			if( (i+1)%10==0 || i==n_elmt-1) large_buffer.push_back(carriage_return);
			
			if(large_buffer.size()>N_block) {
				fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
				large_buffer.resize(0);
			}
		}
		for (i=0; i<n_elmt; i++) {
			//PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->nt_z[i]);
			sprintf(str, "%e ", ibm->nt_z[i]);
			str_to_buffer(str, large_buffer);
			if( (i+1)%10==0 || i==n_elmt-1) large_buffer.push_back(carriage_return);
			
			if(large_buffer.size()>N_block) {
				fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
				large_buffer.resize(0);
			}
		}
		for (i=0; i<n_elmt; i++) {
			//PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_x[i]);
			sprintf(str, "%e ", ibm->ns_x[i]);
			str_to_buffer(str, large_buffer);
			if( (i+1)%10==0 || i==n_elmt-1) large_buffer.push_back(carriage_return);
			
			if(large_buffer.size()>N_block) {
				fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
				large_buffer.resize(0);
			}
		}
		for (i=0; i<n_elmt; i++) {
			//PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_y[i]);
			sprintf(str, "%e ", ibm->ns_y[i]);
			str_to_buffer(str, large_buffer);
			if( (i+1)%10==0 || i==n_elmt-1) large_buffer.push_back(carriage_return);
			
			if(large_buffer.size()>N_block) {
				fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
				large_buffer.resize(0);
			}
		}
		for (i=0; i<n_elmt; i++) {
			//PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->ns_z[i]);
			sprintf(str, "%e ", ibm->ns_z[i]);
			str_to_buffer(str, large_buffer);
			if( (i+1)%10==0 || i==n_elmt-1) large_buffer.push_back(carriage_return);
			
			if(large_buffer.size()>N_block) {
				fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
				large_buffer.resize(0);
			}
		}
		for (i=0; i<n_elmt; i++) {
			//PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
			sprintf(str, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
			str_to_buffer(str, large_buffer);
			large_buffer.push_back(carriage_return);
			
			if(large_buffer.size()>N_block) {
				fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
				large_buffer.resize(0);
			}
		}
		
		if(large_buffer.size()) fwrite(&large_buffer[0], sizeof(char), large_buffer.size(), f);
    
    fclose(f);

  }
  else if (rank) {
    MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    ibm->n_v = n_v;
    
    PetscMalloc(n_v*sizeof(PetscReal), &x_bp);
    PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
    PetscMalloc(n_v*sizeof(PetscReal), &z_bp);
    
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
    
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
    
     /** seokkoo **/
      PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->shear);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->mean_shear);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress1);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress2);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress3);
      PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->pressure);
      PetscMalloc(n_elmt*sizeof(Cmpnts), &ibm->rel_velocity);

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

  return(0);
}

PetscErrorCode calc_ibm_normal(IBMNodes *ibm)
{
  int   n1e, n2e, n3e, i;
  PetscReal  dx12, dy12, dz12, dx13, dy13, dz13, dr;
  
  for (i=0; i<ibm->n_elmt; i++) {
    //PetscPrintf(PETSC_COMM_WORLD, "cop nf %d !\n",i);       
    n1e = ibm->nv1[i]; n2e =ibm->nv2[i]; n3e = ibm->nv3[i];
    dx12 = ibm->x_bp[n2e] - ibm->x_bp[n1e];
    dy12 = ibm->y_bp[n2e] - ibm->y_bp[n1e];
    dz12 = ibm->z_bp[n2e] - ibm->z_bp[n1e];
    
    dx13 = ibm->x_bp[n3e] - ibm->x_bp[n1e];
    dy13 = ibm->y_bp[n3e] - ibm->y_bp[n1e];
    dz13 = ibm->z_bp[n3e] - ibm->z_bp[n1e];
    
    ibm->nf_x[i] = dy12 * dz13 - dz12 * dy13;
    ibm->nf_y[i] = -dx12 * dz13 + dz12 * dx13;
    ibm->nf_z[i] = dx12 * dy13 - dy12 * dx13;
    
    dr = sqrt(ibm->nf_x[i]*ibm->nf_x[i] + 
	      ibm->nf_y[i]*ibm->nf_y[i] + 
	      ibm->nf_z[i]*ibm->nf_z[i]);
    
    ibm->nf_x[i] /=dr; ibm->nf_y[i]/=dr; ibm->nf_z[i]/=dr;

    if ((((1.-ibm->nf_z[i])<=1e-6 )&((-1.+ibm->nf_z[i])<1e-6))|
	(((ibm->nf_z[i]+1.)<=1e-6 )&((-1.-ibm->nf_z[i])<1e-6))) {
      ibm->ns_x[i] = 1.;     
      ibm->ns_y[i] = 0.;     
      ibm->ns_z[i] = 0. ;
      
      ibm->nt_x[i] = 0.;
      ibm->nt_y[i] = 1.;
      ibm->nt_z[i] = 0.;
    } else {
      ibm->ns_x[i] =  ibm->nf_y[i]/ sqrt(ibm->nf_x[i]*ibm->nf_x[i] + 
					 ibm->nf_y[i]*ibm->nf_y[i]);      
      ibm->ns_y[i] = -ibm->nf_x[i]/ sqrt(ibm->nf_x[i]*ibm->nf_x[i] + 
					 ibm->nf_y[i]*ibm->nf_y[i]);     
      ibm->ns_z[i] = 0. ;
      
      ibm->nt_x[i] = -ibm->nf_x[i]*ibm->nf_z[i]/ sqrt(ibm->nf_x[i]*ibm->nf_x[i] + 
						      ibm->nf_y[i]*ibm->nf_y[i]);
      ibm->nt_y[i] = -ibm->nf_y[i]*ibm->nf_z[i]/ sqrt(ibm->nf_x[i]*ibm->nf_x[i] + 
						      ibm->nf_y[i]*ibm->nf_y[i]);
      ibm->nt_z[i] = sqrt(ibm->nf_x[i]*ibm->nf_x[i] + ibm->nf_y[i]*ibm->nf_y[i]);
    }
    
    //Added 4/1/06 iman
    ibm->dA[i] = dr/2.; 
    
    // Added 6/4/06 iman
    // Calc the center of the element
    ibm->cent_x[i]= (ibm->x_bp[n1e]+ibm->x_bp[n2e]+ibm->x_bp[n3e])/3.;
    ibm->cent_y[i]= (ibm->y_bp[n1e]+ibm->y_bp[n2e]+ibm->y_bp[n3e])/3.;
    ibm->cent_z[i]= (ibm->z_bp[n1e]+ibm->z_bp[n2e]+ibm->z_bp[n3e])/3.;
  }
  return(0);
}

PetscErrorCode calc_ibm_velocity(IBMNodes *ibm, PetscReal delti)
{
  PetscReal  v_max=0.;
  int   i, i_vmax=0;
  for (i=0; i<ibm->n_v; i++) {
    ibm->u[i].x = (ibm->x_bp[i] - ibm->x_bp_o[i]) / delti;
    ibm->u[i].y = (ibm->y_bp[i] - ibm->y_bp_o[i]) / delti;
    ibm->u[i].z = (ibm->z_bp[i] - ibm->z_bp_o[i]) / delti;
    
    if (v_max<fabs(ibm->u[i].z)) {
      i_vmax=i;
      v_max= fabs(ibm->u[i].z);
    }
    if (v_max<fabs(ibm->u[i].y)) {
      i_vmax=i;
      v_max= fabs(ibm->u[i].y);
    }
    if (v_max<fabs(ibm->u[i].x)) {
      i_vmax=i;
      v_max= fabs(ibm->u[i].x);
    }
  }
  PetscPrintf(PETSC_COMM_WORLD, "MAX fish Velocity:%le %le %le %le\n", v_max, ibm->x_bp[i_vmax],ibm->y_bp[i_vmax],ibm->z_bp[i_vmax]);
  
  return(0);
}


#define CROSS(dest, v1, v2) \
	dest[0] = v1[1] * v2[2] - v1[2] * v2[1]; \
	dest[1] = v1[2] * v2[0] - v1[0] * v2[2]; \
	dest[2] = v1[0] * v2[1] - v1[1] * v2[0];

#define DOT(v1, v2) (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2])

#define SUB(dest, v1, v2) \
	dest[0] = v1[0] - v2[0]; \
	dest[1] = v1[1] - v2[1]; \
	dest[2] = v1[2] - v2[2];

PetscErrorCode calc_ibm_volumeFlux(IBMNodes *ibm, PetscReal delti, PetscReal *VolumeFlux)
{
  int   i,ii, n_elmt=ibm->n_elmt;
  int n1e, n2e, n3e;
  PetscReal  Vol=0.,p1[3],p2[3],p3[3],p4[3],p5[3],p6[3],sign,nf[3];
  PetscReal  edge1[3],edge2[3],edge3[3],edge2c3[3],volTH,cent[3],cent_o[3],dir[3];

  for (i=0; i<n_elmt; i++) {
    n1e = ibm->nv1[i]; n2e =ibm->nv2[i]; n3e =ibm->nv3[i];

    nf[0]=ibm->nf_x[i];nf[1]=ibm->nf_y[i];nf[2]=ibm->nf_z[i];

    p1[0]=ibm->x_bp[n1e];p1[1]=ibm->y_bp[n1e];p1[2]=ibm->z_bp[n1e];
    p2[0]=ibm->x_bp[n2e];p2[1]=ibm->y_bp[n2e];p2[2]=ibm->z_bp[n2e];
    p3[0]=ibm->x_bp[n3e];p3[1]=ibm->y_bp[n3e];p3[2]=ibm->z_bp[n3e];

    p4[0]=ibm->x_bp_o[n1e];p4[1]=ibm->y_bp_o[n1e];p4[2]=ibm->z_bp_o[n1e];
    p5[0]=ibm->x_bp_o[n2e];p5[1]=ibm->y_bp_o[n2e];p5[2]=ibm->z_bp_o[n2e];
    p6[0]=ibm->x_bp_o[n3e];p6[1]=ibm->y_bp_o[n3e];p6[2]=ibm->z_bp_o[n3e];

    for (ii=0; ii<3; ii++) {
      cent[ii]  =(p1[ii]+p2[ii]+p3[ii])/3.;
      cent_o[ii]=(p4[ii]+p5[ii]+p6[ii])/3.;
    }
    
    // calculate volume flux
    SUB(dir,cent,cent_o);
    sign=DOT(dir,nf);
    if (fabs(sign)>1e-15) 
      sign /=fabs(sign);
    else
      sign =0.;

    SUB(edge1,p4,p1);
    SUB(edge2,p4,p2);
    SUB(edge3,p4,p3);
    CROSS(edge2c3,edge2,edge3);
    volTH=DOT(edge1,edge2c3);
    
    Vol +=sign*fabs(volTH/6.)/delti;

    SUB(edge1,p5,p4);
    SUB(edge2,p5,p2);
    SUB(edge3,p5,p3);
    CROSS(edge2c3,edge2,edge3);
    volTH=DOT(edge1,edge2c3);
    
    Vol +=sign*fabs(volTH/6.)/delti;

    SUB(edge1,p6,p5);
    SUB(edge2,p6,p4);
    SUB(edge3,p6,p3);
    CROSS(edge2c3,edge2,edge3);
    volTH=DOT(edge1,edge2c3);
    
    Vol +=sign*fabs(volTH/6.)/delti;
  }
  *VolumeFlux = Vol;
  PetscPrintf(PETSC_COMM_WORLD, "Volume Flux %e\n", Vol);

  return(0);
}



PetscErrorCode read_bed_cell_depth(IBMNodes *ibm, PetscInt tistart)
{
  int	rank;
  PetscReal	*CellDepth;
  int	i,ii;
  int      n_elmt=ibm->n_elmt;

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if(!rank) { // root processor read in the data
    FILE *fd;
    PetscPrintf(PETSC_COMM_SELF, "READ ib_cell_depth\n");
    char filen[80];  
    sprintf(filen,"ib_cell_depth.dat");
 
    fd = fopen(filen, "r"); if (!fd) SETERRQ(PETSC_COMM_SELF, 1, "Cannot open cell_depth file\n");
    
    if(fd) {
       for (i=0; i<n_elmt; i++) {
//	   fscanf(fd, "%le\n", &CellDepth[i]);
//           ibm->elmt_depth[i] = CellDepth[i];
	   fscanf(fd, "%le %d\n", &(ibm->elmt_depth[i]), &(ibm->Rigidity[i]));
          
	   }
      fclose(fd);
    }
             
    MPI_Bcast(ibm->elmt_depth, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->Rigidity, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
    
    FILE *f;
    
    sprintf(filen, "surface_elmt_depth_%5.5d.dat",tistart);
    f = fopen(filen, "w");
    PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z,elmt_depth,rigidity\n");
    PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEBLOCK, ET=TRIANGLE, VARLOCATION=([1-3]=NODAL,[4-5]=CELLCENTERED)\n",ibm->n_v, ibm->n_elmt);
    for (i=0; i<ibm->n_v; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->x_bp[i]);
    }
    for (i=0; i<ibm->n_v; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->y_bp[i]);
    }
    for (i=0; i<ibm->n_v; i++) {	
      PetscFPrintf(PETSC_COMM_WORLD, f, "%e\n", ibm->z_bp[i]);
    }
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%le\n", ibm->elmt_depth[i]);
    }
    for (i=0; i<ibm->n_elmt; i++) {
      PetscFPrintf(PETSC_COMM_WORLD, f, "%d\n", ibm->Rigidity[i]);
    }

      for (i=0; i<ibm->n_elmt; i++) {
	PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1);
      }

    fclose(f);

  }
  else if (rank) {
    MPI_Bcast(ibm->elmt_depth, n_elmt, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(ibm->Rigidity, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
                 }

  return(0);
}



PetscErrorCode ibm_read_ucd_sedi(IBMNodes *ibm, PetscInt ibi)
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
    sprintf(filen,"%s/ibmdata%2.2d" , path, ibi);
 
    fd = fopen(filen, "r"); if (!fd) SETERRQ(PETSC_COMM_SELF, 1, "Cannot open IBM node file\n");
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
	    
	    /*
      PetscMalloc(n_v*sizeof(PetscReal), &x_bp);	// removed by seokkoo 03.04.2009
      PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
      PetscMalloc(n_v*sizeof(PetscReal), &z_bp);
      */
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));	// added by seokkoo 03.04.2009
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
      
      x_bp = ibm->x_bp;	// seokkoo
      y_bp = ibm->y_bp;
      z_bp = ibm->z_bp;
      
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
	
	x_bp[i] = x_bp[i]*L_dim + CMx_c;//0.25 ;// 24.;	
	y_bp[i] = y_bp[i]*L_dim + CMy_c ;//+ ibi*2.;//5.;//2.;//8.;//6.;//2.   ;// 24.;
	z_bp[i] = z_bp[i]*L_dim + CMz_c ;//+ ibi*1.5;//2.;//8.;//15.;//2.   ;// 24.;
	
	ibm->x_bp[i] = x_bp[i];
	ibm->y_bp[i] = y_bp[i];
	ibm->z_bp[i] = z_bp[i];

	ibm->x_bp0[i] = x_bp[i];
	ibm->y_bp0[i] = y_bp[i];
	ibm->z_bp0[i] = z_bp[i];

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

/*       ibm->x_bp0 = x_bp; ibm->y_bp0 = y_bp; ibm->z_bp0 = z_bp; */
/*       ibm->x_bp_o = x_bp; ibm->y_bp_o = y_bp; ibm->z_bp_o = z_bp; */

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
      
      // Added 1/11/2010 ali
      PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->Bvel));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->u_grad_x));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->u_grad_y));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->v_grad_x));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->v_grad_y));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->c_grad_x));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->c_grad_y));
      //PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->z_grad_x));
      //PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->z_grad_y));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->sbb));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->scc));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->u_max));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->v_max));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->c_max));
      //PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->z_max));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Bvel_u));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Bvel_v));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Bvel_w));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->vx));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->vy));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->vz));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Shvel));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->BShS));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->BCShS));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_flux_12));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_flux_13));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_flux_23));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->SCont));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Delz));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dVol));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dzbp));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->elmt_depth));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z_old));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_zl));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z_AVE));
     // PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->atke));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->atke_old));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->netflux_old));
      //PetscMalloc(100000*sizeof(PetscReal), &(ibm->dtime));
      //PetscMalloc(100000*sizeof(PetscReal), &(ibm->time_bedchange));
      PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_l));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->deltz_p_ds));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->deltz_p_us));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->A_us));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->A_ds));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->max_bed_angle));
      PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->C));
      PetscMalloc(n_elmt*sizeof(int), &(ibm->Rigidity));
      PetscMalloc(n_elmt*sizeof(int), &(ibm->Mobility));
      if(periodic_morpho) PetscMalloc(Nseg*sizeof(PetscReal), &(ibm->SedFlux));
     // end added 
      
      

      
	//seokkoo begin
	/*
	{	//only for rank 0
		PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->_nv1);
		PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->_nv2);
		PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->_nv3);
		
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->_x_bp);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->_y_bp);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->_z_bp);
	}
	
	PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->count);
	PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->local2global_elmt);
	*/

	PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->shear);
	PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->mean_shear);
	PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress1);
	PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress2);
	PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress3);
	PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->pressure);
      	PetscMalloc(n_elmt*sizeof(Cmpnts), &ibm->rel_velocity);
	//seokkoo end
	

	// add xiaolei  SEDI

	PetscMalloc(n_v*sizeof(Node2Cell), &ibm->n2c);
	PetscMalloc(n_elmt*sizeof(Cell2Cell), &ibm->c2c);
	PetscMalloc(n_elmt*sizeof(Cell2AllCell), &ibm->c2ac);

        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->count));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->ib_elmt));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->jb_elmt));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->kb_elmt));

        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->xb_elmt));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->yb_elmt));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->zb_elmt));



	// end add xiaolei


      for (i=0; i<n_elmt; i++) {

	fscanf(fd, "%i %i %s %i %i %i\n", &ii,&ii, &ss, nv1+i, nv2+i, nv3+i);
	nv1[i] = nv1[i] - 1; nv2[i] = nv2[i]-1; nv3[i] = nv3[i] - 1;

		/*	      
		// seokkoo
	      ibm->_nv1[i] = nv1[i];
	      ibm->_nv2[i] = nv2[i];
	      ibm->_nv3[i] = nv3[i];
	      // seokkoo
	      ibm->_x_bp[i] = ibm->x_bp[i];
	      ibm->_y_bp[i] = ibm->y_bp[i];
	      ibm->_z_bp[i] = ibm->z_bp[i];
		*/
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
/*       if (fabs(nf_x[i])<.5) */
/* 	nf_x[i]=0.; */

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

 /*    PetscFree(dA); */
/*     PetscFree(nf_x);PetscFree(nf_y);PetscFree(nf_z); */
/*     PetscFree(nt_x);PetscFree(nt_y);PetscFree(nt_z); */
/*     PetscFree(ns_x);PetscFree(ns_y);PetscFree(ns_z); */
/*     PetscFree(nv1);PetscFree(nv2);PetscFree(nv3); */
/*     PetscFree(x_bp);PetscFree(y_bp);PetscFree(z_bp); */
    PetscInt ti=0;
    FILE *f;
    //char filen[80];
    sprintf(filen, "%s/surface%3.3d_%2.2d_nf.dat",path,ti,ibi);
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
    
/*     PetscFPrintf(PETSC_COMM_WORLD, f, "Variables=x,y,z\n"); */
/*     PetscFPrintf(PETSC_COMM_WORLD, f, "ZONE T='TRIANGLES', N=%d, E=%d, F=FEPOINT, ET=TRIANGLE\n", n_v, n_elmt); */
/*     for (i=0; i<n_v; i++) { */
      
/*       PetscFPrintf(PETSC_COMM_WORLD, f, "%e %e %e\n", ibm->x_bp[i], ibm->y_bp[i], ibm->z_bp[i]); */
/*     } */
/*     for (i=0; i<n_elmt; i++) { */
/*       PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %d\n", ibm->nv1[i]+1, ibm->nv2[i]+1, ibm->nv3[i]+1); */
/*     } */
    fclose(f);

  }
  else if (rank) {
    MPI_Bcast(&(n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
    ibm->n_v = n_v;
    /*
    PetscMalloc(n_v*sizeof(PetscReal), &x_bp);	// removed by seokkoo 03.04.2009
    PetscMalloc(n_v*sizeof(PetscReal), &y_bp);
    PetscMalloc(n_v*sizeof(PetscReal), &z_bp);
    */
	
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
	  
	x_bp = ibm->x_bp;	// added by seokkoo 03.04.2009
	y_bp = ibm->y_bp;
	z_bp = ibm->z_bp;
	  
    
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

    // Added 1/11/2010 ali
    PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->Bvel));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->u_grad_x));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->u_grad_y));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->v_grad_x));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->v_grad_y));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->c_grad_x));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->c_grad_y));
    //PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->z_grad_x));
    //PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->z_grad_y));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->sbb));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->scc));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->u_max));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->v_max));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->c_max));
    //PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->z_max));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Bvel_u));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Bvel_v));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Bvel_w));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->vx));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->vy));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->vz));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Shvel));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->BShS));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->BCShS));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_flux_12));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_flux_13));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_flux_23));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->SCont));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Delz));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dVol));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dzbp));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->elmt_depth));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z_old));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z_AVE));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_zl));
    // PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->atke));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->atke_old));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->netflux_old));
    //PetscMalloc(100000*sizeof(PetscReal), &(ibm->dtime));
    //PetscMalloc(100000*sizeof(PetscReal), &(ibm->time_bedchange));
    PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_l));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->deltz_p_ds));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->deltz_p_us));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->A_us));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->A_ds));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->max_bed_angle));
    PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->C));
    PetscMalloc(n_elmt*sizeof(int), &(ibm->Rigidity));
    PetscMalloc(n_elmt*sizeof(int), &(ibm->Mobility));
    if(periodic_morpho) PetscMalloc(Nseg*sizeof(PetscReal), &(ibm->SedFlux));
    // end added


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
   
		/*	
		//seokkoo
		PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->count);
		PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->local2global_elmt);
		*/
		//seokkoo
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->shear);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->mean_shear);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress1);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress2);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->reynolds_stress3);
		PetscMalloc(n_elmt*sizeof(PetscReal), &ibm->pressure);
      		PetscMalloc(n_elmt*sizeof(Cmpnts), &ibm->rel_velocity);
		

	// add xiaolei  SEDI

	PetscMalloc(n_v*sizeof(Node2Cell), &ibm->n2c);
	PetscMalloc(n_elmt*sizeof(Cell2Cell), &ibm->c2c);
	PetscMalloc(n_elmt*sizeof(Cell2AllCell), &ibm->c2ac);

        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->count));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->ib_elmt));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->jb_elmt));
        PetscMalloc(n_elmt*sizeof(PetscInt), &(ibm->kb_elmt));

        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->xb_elmt));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->yb_elmt));
        PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->zb_elmt));



	// end add xiaolei







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
	PetscPrintf(PETSC_COMM_WORLD, "Read ucd file !\n");
  return(0);
}


// xiaolei add read immersed grid data in xpatch format 
PetscErrorCode ibm_read_xpatch(IBMNodes *ibm, PetscInt ibi)
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
    		sprintf(filen,"%s/ibmdata_xpatch%2.2d" , path, ibi);
 
    		fd = fopen(filen, "r"); if (!fd) SETERRQ(PETSC_COMM_SELF, 1, "Cannot open IBM node file\n");
    		n_v =0;

    		if (fd) {
      			fgets(string, 128, fd);
      			fgets(string, 128, fd);
      			fgets(string, 128, fd);
      			fgets(string, 128, fd);
      
      			fscanf(fd, "%i",&n_v);
      			PetscPrintf(PETSC_COMM_SELF, "number of nodes & elements %d\n",n_v);
      
      			ibm->n_v = n_v;
      
      			MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
	    
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));	// added by seokkoo 03.04.2009
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
      
      			x_bp = ibm->x_bp;	// seokkoo
      			y_bp = ibm->y_bp;
      			z_bp = ibm->z_bp;
      
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
				fscanf(fd, "%le %le %le", &x_bp[i], &y_bp[i], &z_bp[i]);
	
				x_bp[i] = x_bp[i]*L_dim + CMx_c;//0.25 ;// 24.;	
				y_bp[i] = y_bp[i]*L_dim + CMy_c ;//+ ibi*2.;//5.;//2.;//8.;//6.;//2.   ;// 24.;
				z_bp[i] = z_bp[i]*L_dim + CMz_c ;//+ ibi*1.5;//2.;//8.;//15.;//2.   ;// 24.;
	
				ibm->x_bp[i] = x_bp[i];
				ibm->y_bp[i] = y_bp[i];
				ibm->z_bp[i] = z_bp[i];

				ibm->x_bp0[i] = x_bp[i];
				ibm->y_bp0[i] = y_bp[i];
				ibm->z_bp0[i] = z_bp[i];

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


      			MPI_Bcast(ibm->x_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp0, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

      			MPI_Bcast(ibm->x_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      
      			MPI_Bcast(ibm->x_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->y_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);
      			MPI_Bcast(ibm->z_bp_o, n_v, MPIU_REAL, 0, PETSC_COMM_WORLD);

			fscanf(fd, "%d", &ii);
      			fgets(string, 128, fd);
			fscanf(fd, "%d %d", &n_elmt, &ii);
			ibm->n_elmt=n_elmt;
      			MPI_Bcast(&(ibm->n_elmt), 1, MPI_INT, 0, PETSC_COMM_WORLD);

      			PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
      			PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
      			PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);
      
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
      
     			// Added 1/11/2010 ali
      			PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->Bvel));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->u_grad_x));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->u_grad_y));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->v_grad_x));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->v_grad_y));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->c_grad_x));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->c_grad_y));
      			//PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->z_grad_x));
      			//PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->z_grad_y));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->sbb));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->scc));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->u_max));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->v_max));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->c_max));
      			//PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->z_max));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Bvel_u));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Bvel_v));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Bvel_w));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->vx));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->vy));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->vz));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Shvel));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->BShS));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->BCShS));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_flux_12));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_flux_13));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_flux_23));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->SCont));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Delz));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dVol));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dzbp));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->elmt_depth));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z_old));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_zl));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z_AVE));
     			// PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->atke));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->atke_old));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->netflux_old));
      			//PetscMalloc(100000*sizeof(PetscReal), &(ibm->dtime));
      			//PetscMalloc(100000*sizeof(PetscReal), &(ibm->time_bedchange));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_l));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->deltz_p_ds));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->deltz_p_us));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->A_us));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->A_ds));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->max_bed_angle));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->C));
      			PetscMalloc(n_elmt*sizeof(int), &(ibm->Rigidity));
      			PetscMalloc(n_elmt*sizeof(int), &(ibm->Mobility));
      			if(periodic_morpho) PetscMalloc(Nseg*sizeof(PetscReal), &(ibm->SedFlux));
     			// end added 
      

			// add xiaolei  SEDI

			PetscMalloc(n_v*sizeof(Node2Cell), &ibm->n2c);
			PetscMalloc(n_elmt*sizeof(Cell2Cell), &ibm->c2c);
			PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->color);

			// end add xiaolei

      			for (i=0; i<n_elmt; i++) {

				fscanf(fd, "%i %i %i %i %i %i\n", nv1+i, nv2+i, nv3+i, &ii, &(ibm->color[i]), &ii);
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

		// xiaolei add 
    		MPI_Bcast(ibm->color, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);

    		PetscInt ti=0;
    		FILE *f;
    		//char filen[80];
    		sprintf(filen, "%s/surface%3.3d_%2.2d_nf.dat",path,ti,ibi);
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

    		{
      
      			MPI_Bcast(&(ibm->n_v), 1, MPI_INT, 0, PETSC_COMM_WORLD);
	    
      			n_v = ibm->n_v;

      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp));	// added by seokkoo 03.04.2009
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp));
      
      			x_bp = ibm->x_bp;	// seokkoo
      			y_bp = ibm->y_bp;
      			z_bp = ibm->z_bp;
      
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp_o));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp_o));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_o));

      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->x_bp0));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->y_bp0));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp0));
      
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->u));
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->uold));
      			PetscMalloc(n_v*sizeof(Cmpnts), &(ibm->urm1));

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

			n_elmt = ibm->n_elmt=n_elmt;

      			PetscMalloc(n_elmt*sizeof(PetscInt), &nv1);
      			PetscMalloc(n_elmt*sizeof(PetscInt), &nv2);
      			PetscMalloc(n_elmt*sizeof(PetscInt), &nv3);
      
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
      
     			// Added 1/11/2010 ali
      			PetscMalloc(n_elmt*sizeof(Cmpnts), &(ibm->Bvel));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->u_grad_x));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->u_grad_y));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->v_grad_x));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->v_grad_y));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->c_grad_x));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->c_grad_y));
      			//PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->z_grad_x));
      			//PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->z_grad_y));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->sbb));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->scc));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->u_max));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->v_max));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->c_max));
      			//PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->z_max));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Bvel_u));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Bvel_v));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Bvel_w));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->vx));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->vy));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->vz));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Shvel));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->BShS));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->BCShS));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_flux_12));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_flux_13));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->F_flux_23));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->SCont));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->Delz));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dVol));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->dzbp));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->elmt_depth));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z_old));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_zl));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->cent_z_AVE));
     			// PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->atke));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->atke_old));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->netflux_old));
      			//PetscMalloc(100000*sizeof(PetscReal), &(ibm->dtime));
      			//PetscMalloc(100000*sizeof(PetscReal), &(ibm->time_bedchange));
      			PetscMalloc(n_v*sizeof(PetscReal), &(ibm->z_bp_l));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->deltz_p_ds));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->deltz_p_us));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->A_us));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->A_ds));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->max_bed_angle));
      			PetscMalloc(n_elmt*sizeof(PetscReal), &(ibm->C));
      			PetscMalloc(n_elmt*sizeof(int), &(ibm->Rigidity));
      			PetscMalloc(n_elmt*sizeof(int), &(ibm->Mobility));
      			if(periodic_morpho) PetscMalloc(Nseg*sizeof(PetscReal), &(ibm->SedFlux));
     			// end added 
      

			// add xiaolei  SEDI

			PetscMalloc(n_v*sizeof(Node2Cell), &ibm->n2c);
			PetscMalloc(n_elmt*sizeof(Cell2Cell), &ibm->c2c);
			PetscMalloc(n_elmt*sizeof(PetscInt), &ibm->color);

			// end add xiaolei

    		}
      
    
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

		// xiaolei add 
    		MPI_Bcast(ibm->color, n_elmt, MPI_INT, 0, PETSC_COMM_WORLD);
	}
	PetscPrintf(PETSC_COMM_WORLD, "Read xpatch file !\n");

  	return(0);
}







