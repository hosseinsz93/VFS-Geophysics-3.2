#include "variables.h"
#include "petscksp.h"
#include "petscpc.h"
#include "petscpcmg.h"
#include <stdlib.h>

/*
extern int pseudo_periodic, rans, implicit;
extern int block_number, freesurface, immersed, ti, tistart, les, tiout;
extern char path[256];
extern int movefsi, rotatefsi;
extern int tistart;

*/

PetscReal poisson_threshold=0.1;

#define lidx2(i,j,k,user)	(PetscInt)(gid[k][j][i])

//PetscErrorCode VolumeFlux(UserCtx *user, Vec lUcor, PetscReal *ibm_Flux, PetscReal *ibm_Area);

double time_coeff()
{
  if (/*levelset ||*/ rans)
  {
    if (ti == tistart)
      return 1.;
    else
      return 1.5;
  }
  else return 1.;
	
};
		
PetscErrorCode MyKSPMonitor(KSP ksp, int n, PetscReal rnom, void *dummy)
{
  UserCtx *user = (UserCtx*)dummy;
  Vec x;
  PetscInt tttt, mi, j, k;
  PetscReal norm;


  DMDALocalInfo	info = user->info;
  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;

  VecDuplicate(user->P, &x);
  KSPBuildResidual(ksp, PETSC_NULL, PETSC_NULL, &x);
  VecMax(x, &tttt, &norm);
  PetscPrintf(PETSC_COMM_WORLD, "KSP Max %d %le\n", tttt, norm);
  for (k=zs; k<ze; k++) {
    for (j=ys; j<ye; j++) {
      for (mi=xs; mi<xe; mi++) {
	if (lidx(mi,j,k,user) ==tttt) {
	  PetscPrintf(PETSC_COMM_SELF, "KspMMax %d %d %d %d %le\n", lidx(mi,j,k,user),mi,j, k, norm);
	}
      }
    }
  }

  VecMin(x, &tttt, &norm);
  PetscPrintf(PETSC_COMM_WORLD, "KSP Max %d %le\n", tttt, norm);

  for (k=zs; k<ze; k++) {
    for (j=ys; j<ye; j++) {
      for (mi=xs; mi<xe; mi++) {
	if (lidx(mi,j,k,user) ==tttt) {
	  PetscPrintf(PETSC_COMM_SELF, "KspMMin %d %d %d %d %le\n", lidx(mi,j,k,user),mi,j, k, norm);
	}
      }
    }
  }

  VecDestroy(&x);

  return 0;
}

int lidx(int i, int j, int k, UserCtx *user)
{
	DMDALocalInfo	info = user->info;
	int	gxs, gxe, gys, gye, gzs, gze;

	gxs = info.gxs; gxe = gxs + info.gxm;
	gys = info.gys; gye = gys + info.gym;
	gzs = info.gzs; gze = gzs + info.gzm;


	if (!(user->aotopetsc)) {
		user->aotopetsc = PETSC_TRUE;
		DMDAGetGlobalIndices(user->da, PETSC_NULL, &user->idx_from);
	}
	
	return (user->idx_from[(k-gzs) * (info.gxm*info.gym) + (j-gys)*(info.gxm) + (i-gxs)]);
  
  //  return (i + j * mx + k * mx * my);
}

void Convert_Phi2_Phi(UserCtx *user)
{
	DMDALocalInfo	info = user->info;
	DM		da = user->da;
	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
	int i, j, k;
	PetscReal ***nvert, ***phi, ***lphi, *phi2;
	
	int lxs = xs, lxe = xe;
	int lys = ys, lye = ye;
	int lzs = zs, lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	VecSet(user->Phi,0);
	DMDAVecGetArray(da, user->Phi, &phi);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	VecGetArray(user->Phi2, &phi2);
	
	int pos=0;
	for(k=lzs; k<lze; k++)
	for(j=lys; j<lye; j++)
	for(i=lxs; i<lxe; i++) {
		/*if(i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1) {
			phi[k][j][i]=0;
		}
		else */
		if( (int) nvert[k][j][i]>poisson_threshold ) {
			/*if(movefsi || rotatefsi) {
				phi[k][j][i] = phi2[pos++];
			}
			else*/ phi[k][j][i]=0;
		}
		else {
			phi[k][j][i] = phi2[pos++];
		}
	}
	
	DMDAVecRestoreArray(da, user->Phi, &phi);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	VecRestoreArray(user->Phi2, &phi2);
	
	DMGlobalToLocalBegin(da, user->Phi, INSERT_VALUES, user->lPhi);
	DMGlobalToLocalEnd(da, user->Phi, INSERT_VALUES, user->lPhi);
	
	DMDAVecGetArray(da, user->lPhi, &lphi);
	DMDAVecGetArray(da, user->Phi, &phi);
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		int flag=0, a=i, b=j, c=k;
		
		if(i_periodic && i==0) a=mx-2, flag=1;
		else if(i_periodic && i==mx-1) a=1, flag=1;
		
		if(j_periodic && j==0) b=my-2, flag=1;
		else if(j_periodic && j==my-1) b=1, flag=1;
		
		if(k_periodic && k==0) c=mz-2, flag=1;
		else if(k_periodic && k==mz-1) c=1, flag=1;
		
		if(ii_periodic && i==0) a=-2, flag=1;
		else if(ii_periodic && i==mx-1) a=mx+1, flag=1;
		
		if(jj_periodic && j==0) b=-2, flag=1;
		else if(jj_periodic && j==my-1) b=my+1, flag=1;
		
		if(kk_periodic && k==0) c=-2, flag=1;
		else if(kk_periodic && k==mz-1) c=mz+1, flag=1;
		
		if(flag) {
			phi[k][j][i] = lphi[c][b][a];
		}
		
		//if(k==1 && i!=0)	printf("%d,%d,%d, %f %f %f\n", i,j,k, lphi[-2][j][i], lphi[k][j][i], lphi[k+1][j][i]);
	}
	DMDAVecRestoreArray(da, user->lPhi, &lphi);
	DMDAVecRestoreArray(da, user->Phi, &phi);
	//exit(0);
}

int setup_lidx2(UserCtx *user)
{
	DMDALocalInfo	info = user->info;
	DM		da = user->da;
	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
	int i, j, k;
	PetscReal ***nvert, ***gid, ***lid;

	Vec	Lid;
	
	int lxs = xs, lxe = xe;
	int lys = ys, lye = ye;
	int lzs = zs, lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;

	if(ti==tistart) {
		VecDuplicate(user->lNvert, &user->Gid);
	}
	VecDuplicate(user->lNvert, &Lid);
	
	VecSet(user->Gid, -1);
	VecSet(Lid, -1);
	
	DMDAVecGetArray(da, user->Gid, &gid);
	DMDAVecGetArray(da, Lid, &lid);
	DMDAVecGetArray(da, user->lNvert, &nvert);

	int r, myrank, size;
	MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
	MPI_Comm_size(PETSC_COMM_WORLD,&size);
		
	std::vector<int> ndof_node(size), ndof_node_tmp(size);	// # of pressure dof for processors
	
	int ndof_node_accu;
	
	for(r=0; r<size; r++) {
		ndof_node_tmp[r] = 0;
	}
	
	for(k=zs; k<ze; k++)
	for(j=ys; j<ye; j++)
	for(i=xs; i<xe; i++) {
		
		if(poisson==-1) {
			lid[k][j][i] = (PetscReal)ndof_node_tmp[myrank]++;
		}
		else {
			if(i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1) { }
			else if( (int)nvert[k][j][i]>poisson_threshold ) {
				/*if(movefsi || rotatefsi) {
					lid[k][j][i] = (PetscReal)ndof_node_tmp[myrank]++;
				}
				else {}*/
			}
			else {
				lid[k][j][i] = (PetscReal)ndof_node_tmp[myrank]++;
			}
		}
	}
	
	MPI_Allreduce( &ndof_node_tmp[0], &ndof_node[0], size, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);
		
	ndof_node_accu = 0;
	for(r=0; r<myrank; r++) ndof_node_accu += ndof_node[r];
		
	PetscInt n;
	user->p_global_begin = ndof_node_accu;
		
	VecGetSize(user->Phi,&n);
	if(myrank==size-1) {
		printf("\n\n********* %d %d ***********\n\n", ndof_node_accu + ndof_node[myrank], (int)n);
		user->reduced_p_size = ndof_node_accu + ndof_node[myrank];
	}
		
	MPI_Bcast(&user->reduced_p_size, 1, MPI_INT, size-1, PETSC_COMM_WORLD);
		
	PetscBarrier(PETSC_NULL);
		
	for(k=zs; k<ze; k++)
	for(j=ys; j<ye; j++)
	for(i=xs; i<xe; i++) {
		if((int)(lid[k][j][i])>=0) {
			gid[k][j][i] = lid[k][j][i] + ndof_node_accu;	// gid is double, be careful
		}
	}
		
	user->local_Phi2_size = ndof_node[myrank];
	
	if(ti!=tistart) VecDestroy(&user->Phi2);
	
	VecCreateMPI(PETSC_COMM_WORLD, ndof_node[myrank], PETSC_DETERMINE, &user->Phi2);
		
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(da, user->Gid, &gid);
	DMDAVecRestoreArray(da, Lid, &lid);
	
	VecDestroy(&Lid);

        DMDALocalToLocalBegin(da, user->Gid, INSERT_VALUES, user->Gid);
        DMDALocalToLocalEnd(da, user->Gid, INSERT_VALUES, user->Gid);
	
	if(periodic) {
		DMDAVecGetArray(da, user->Gid, &gid);
		for(k=zs; k<ze; k++)
		for(j=ys; j<ye; j++)
		for(i=xs; i<xe; i++) {
			int flag=0, a=i, b=j, c=k;
			
			if(i_periodic && i==0) a=mx-2, flag=1;
			else if(i_periodic && i==mx-1) a=1, flag=1;
			
			if(j_periodic && j==0) b=my-2, flag=1;
			else if(j_periodic && j==my-1) b=1, flag=1;
			
			if(k_periodic && k==0) c=mz-2, flag=1;
			else if(k_periodic && k==mz-1) c=1, flag=1;
			
			if(ii_periodic && i==0) a=-2, flag=1;
			else if(ii_periodic && i==mx-1) a=mx+1, flag=1;
			
			if(jj_periodic && j==0) b=-2, flag=1;
			else if(jj_periodic && j==my-1) b=my+1, flag=1;
			
			if(kk_periodic && k==0) c=-2, flag=1;
			else if(kk_periodic && k==mz-1) c=mz+1, flag=1;
			
			if(flag) gid[k][j][i] = gid[c][b][a];
			
			//if(i==1 && k==mz-2) printf("%d, %d, %d, %d <= %d %d %d \n", (int)gid[k][j][i], k,j,i, c,b,a);
		}
		DMDAVecRestoreArray(da, user->Gid, &gid);
		
		DMDALocalToLocalBegin(da, user->Gid, INSERT_VALUES, user->Gid);
		DMDALocalToLocalEnd(da, user->Gid, INSERT_VALUES, user->Gid);
	}	
	
	return 0;
}

PetscErrorCode GridRestriction(int i, int j, int k,
			       int *ih, int *jh, int *kh,
			       UserCtx *user)
{
  if (*(user->isc)) {
    *ih = i;
  }
  else {
    *ih = 2 * i;
  }

  if (*(user->jsc)) {
    *jh = j;
  }
  else {
    *jh = 2 * j;
  }

  if (*(user->ksc)) {
    *kh = k;
  }
  else {
    *kh = 2 * k;
  }

  return 0;
}

PetscErrorCode MyRestriction(Mat A, Vec X, Vec F)
{
  UserCtx *user;

  MatShellGetContext(A, (void**)&user);


  
  DM da = user->da;

  DM da_f = *user->da_f;

  DMDALocalInfo	info;
  DMDAGetLocalInfo(da, &info);
  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;
  
  PetscReal ***f, ***x, ***nvert;
  int i, j, k, ih, jh, kh, ia, ja, ka;

  DMDAVecGetArray(da,   F, &f);

  Vec lX;
  //  DMGetLocalVector(da_f, &lX);
  DMCreateLocalVector(da_f, &lX);
  DMGlobalToLocalBegin(da_f, X, INSERT_VALUES, lX);
  DMGlobalToLocalEnd(da_f, X, INSERT_VALUES, lX);  
  DMDAVecGetArray(da_f, lX, &x);

  DMDAVecGetArray(da, user->lNvert, &nvert);

  PetscReal ***nvert_f;
  DMDAVecGetArray(da_f, user->user_f->lNvert, &nvert_f);

  if (*(user->isc)) ia = 0;
  else ia = 1;

  if (*(user->jsc)) ja = 0;
  else ja = 1;

  if (*(user->ksc)) ka = 0;
  else ka = 1;

  for (k=zs; k<ze; k++) {
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
	if (k==0) {
	  f[k][j][i] = 0.;
	}
	else if (k==mz-1) {
	  f[k][j][i] = 0.;
	}
	else if (j==0) {
	  f[k][j][i] = 0.;
	}
	else if (j==my-1) {
	  f[k][j][i] = 0.;
	}
	else if (i==0) {
	  f[k][j][i] = 0.;
	}
	else if (i==mx-1) {
	  f[k][j][i] = 0.;
	}
	else {
	  GridRestriction(i, j, k, &ih, &jh, &kh, user);
	  f[k][j][i] = 0.125 *
	    (x[kh   ][jh   ][ih   ] * PetscMax(0., 1 - nvert_f[kh   ][jh   ][ih   ]) +
	     x[kh   ][jh   ][ih-ia] * PetscMax(0., 1 - nvert_f[kh   ][jh   ][ih-ia]) +
	     x[kh   ][jh-ja][ih   ] * PetscMax(0., 1 - nvert_f[kh   ][jh-ja][ih   ]) +
	     x[kh-ka][jh   ][ih   ] * PetscMax(0., 1 - nvert_f[kh-ka][jh   ][ih   ]) +
	     x[kh   ][jh-ja][ih-ia] * PetscMax(0., 1 - nvert_f[kh   ][jh-ja][ih-ia]) +
	     x[kh-ka][jh-ja][ih   ] * PetscMax(0., 1 - nvert_f[kh-ka][jh-ja][ih   ]) +
	     x[kh-ka][jh   ][ih-ia] * PetscMax(0., 1 - nvert_f[kh-ka][jh   ][ih-ia]) +
	     x[kh-ka][jh-ja][ih-ia] * PetscMax(0., 1 - nvert_f[kh-ka][jh-ja][ih-ia]));

/* 	  PetscReal scale; */
/* 	  scale = (PetscMax(0., 1 - nvert_f[kh   ][jh   ][ih   ]) + */
/* 		   PetscMax(0., 1 - nvert_f[kh   ][jh   ][ih-ia]) + */
/* 		   PetscMax(0., 1 - nvert_f[kh   ][jh-ja][ih   ]) + */
/* 		   PetscMax(0., 1 - nvert_f[kh-ka][jh   ][ih   ]) + */
/* 		   PetscMax(0., 1 - nvert_f[kh   ][jh-ja][ih-ia]) + */
/* 		   PetscMax(0., 1 - nvert_f[kh-ka][jh-ja][ih   ]) + */
/* 		   PetscMax(0., 1 - nvert_f[kh-ka][jh   ][ih-ia]) + */
/* 		   PetscMax(0., 1 - nvert_f[kh-ka][jh-ja][ih-ia])); */
/* 	  if (scale > 0) f[k][j][i] *= (8./scale); */

	  if (nvert[k][j][i] > 0.1) f[k][j][i] = 0.;
	}
      }
    }
  }


  DMDAVecRestoreArray(da_f, user->user_f->lNvert, &nvert_f);

  DMDAVecRestoreArray(da_f, lX, &x);
  VecDestroy(&lX);
  //  DMRestoreLocalVector(da_f, &lX);
  DMDAVecRestoreArray(da,   F,  &f);
  DMDAVecRestoreArray(da, user->lNvert, &nvert);
/*   PetscReal norm; */
/*   VecNorm(F, NORM_2, &norm); */
/*   PetscPrintf(PETSC_COMM_WORLD, "Restriction Norm %le\n", norm); */
  return 0;
}

#define GridInterpolation(i, j, k, ic, jc, kc, ia, ja, ka, user) \
  if (*(user->isc)) { \
    ic = i; \
    ia = 0; \
  } \
  else { \
    ic = (i+1) / 2; \
    ia = (i - 2 * (ic)) == 0 ? 1 : -1; \
    if (i==1 || i==mx-2) ia = 0; \
  }\
  if (*(user->jsc)) { \
    jc = j; \
    ja = 0; \
  } \
  else { \
    jc = (j+1) / 2; \
    ja = (j - 2 * (jc)) == 0 ? 1 : -1; \
    if (j==1 || j==my-2) ja = 0; \
  } \
  if (*(user->ksc)) { \
    kc = k; \
    ka = 0; \
  } \
  else { \
    kc = (k+1) / 2; \
    ka = (k - 2 * (kc)) == 0 ? 1 : -1; \
    if (k==1 || k==mz-2) ka = 0; \
  } \
  if (ka==-1 && nvert_c[kc-1][jc][ic] > 0.1) ka = 0; \
  else if (ka==1 && nvert_c[kc+1][jc][ic] > 0.1) ka = 0; \
  if (ja==-1 && nvert_c[kc][jc-1][ic] > 0.1) ja = 0; \
  else if (ja==1 && nvert_c[kc][jc+1][ic] > 0.1) ja = 0; \
  if (ia==-1 && nvert_c[kc][jc][ic-1] > 0.1) ia = 0; \
  else if (ia==1 && nvert_c[kc][jc][ic+1] > 0.1) ia = 0;
 

/* PetscErrorCode MyCoarseBC(Mat A, Vec X) */
/* { */
/*   UserCtx *user; */

/*   MatShellGetContext(A, (void*)&user); */

/*   UserCtx *user_c; */
/*   user_c = user->user_c; */

/*   DMDALocalInfo	info = user_c->info; */
/*   int	xs = info.xs, xe = info.xs + info.xm; */
/*   int  	ys = info.ys, ye = info.ys + info.ym; */
/*   int	zs = info.zs, ze = info.zs + info.zm; */
/*   int	mx = info.mx, my = info.my, mz = info.mz; */
/*   int	lxs, lxe, lys, lye, lzs, lze; */

/* } */
PetscErrorCode MyInterpolation(Mat A, Vec X, Vec F)
{
  UserCtx *user;

  MatShellGetContext(A, (void**)&user);

  
  
  DM da = user->da;

  DM da_c = *user->da_c;

  DMDALocalInfo	info = user->info;
  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;
  int	lxs, lxe, lys, lye, lzs, lze;
  
  PetscReal ***f, ***x, ***nvert, ***nvert_c;
  int i, j, k, ic, jc, kc, ia, ja, ka;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;


  DMDAVecGetArray(da,   F, &f);


  Vec lX;
  DMCreateLocalVector(da_c, &lX);
  //  DMGetLocalVector(da_c, &lX);
  DMGlobalToLocalBegin(da_c, X, INSERT_VALUES, lX);
  DMGlobalToLocalEnd(da_c, X, INSERT_VALUES, lX);  
  DMDAVecGetArray(da_c, lX, &x);

  DMDAVecGetArray(da, user->lNvert, &nvert);
  DMDAVecGetArray(da_c, *(user->lNvert_c), &nvert_c);
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {

	GridInterpolation(i, j, k, ic, jc, kc, ia, ja, ka, user);
/* 	  kc = (k + 1) / 2; */
/* 	  jc = (j + 1) / 2; */
/* 	  ic = (i + 1) / 2; */
/* 	  ka = (k - 2 * kc)==0 ? 1 : -1; */
/* 	  ja = (j - 2 * jc)==0 ? 1 : -1; */
/* 	  ia = (i - 2 * ic)==0 ? 1 : -1; */
/* 	  if (ka==-1 &&(k==1 || nvert_c[kc-1][jc][ic]>0.1)) ka = 0; */
/* 	  else if (ka==1 && (k==mz-2 || nvert_c[kc+1][jc][ic]>0.1)) ka=0; */

/* 	  if (ja==-1 &&(j==1 || nvert_c[kc][jc-1][ic]>0.1)) ja = 0; */
/* 	  else if (ja==1 &&(j==my-2 || nvert_c[kc][jc+1][ic]>0.1)) ja=0; */

/* 	  if (ia==-1 &&(i==1 || nvert_c[kc][jc][ic-1]>0.1)) ia = 0; */
/* 	  else if (ia==1 && (i==mx-2 || nvert_c[kc][jc][ic+1]>0.1)) ia=0; */

/* 	f[k][j][i] = x[kc][jc][ic]; */
	  f[k][j][i] = (x[kc   ][jc   ][ic   ] * 9 +
			x[kc   ][jc+ja][ic   ] * 3 +
			x[kc   ][jc   ][ic+ia] * 3 +
			x[kc   ][jc+ja][ic+ia]) * 3./64. +
	    (x[kc+ka][jc   ][ic   ] * 9 +
	     x[kc+ka][jc+ja][ic   ] * 3 +
	     x[kc+ka][jc   ][ic+ia] * 3 +
	     x[kc+ka][jc+ja][ic+ia]) /64.;
      }
    }
  }

  for (k=zs; k<ze; k++) {
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {

	if (i==0) {
	  f[k][j][i] = 0.;//-f[k][j][i+1];
	}
	else if (i==mx-1) {
	  f[k][j][i] = 0.;//-f[k][j][i-1];
	}
	else if (j==0) {
	  f[k][j][i] = 0.;//-f[k][j+1][i];
	}
	else if (j==my-1) {
	  f[k][j][i] = 0.;//-f[k][j-1][i];
	}
	else if (k==0) {
	  f[k][j][i] = 0.;//-f[k+1][j][i];
	}
	else if (k==mz-1) {
	  f[k][j][i] = 0.;//-f[k-1][j][i];
	}
	if (nvert[k][j][i] > 0.1) f[k][j][i] = 0.;
/* 	  f[k][j][i] = 0.125 * */
/* 	    (x[kc  ][jh  ][ih  ] + */
/* 	     x[kc  ][jh  ][ih-1] + */
/* 	     x[kh  ][jh-1][ih  ] + */
/* 	     x[kh-1][jh  ][ih  ] + */
/* 	     x[kh  ][jh-1][ih-1] + */
/* 	     x[kh-1][jh-1][ih  ] + */
/* 	     x[kh-1][jh  ][ih-1] + */
/* 	     x[kh-1][jh-1][ih-1]); */
/* 	} */
      }
    }
  }

  DMDAVecRestoreArray(da, user->lNvert, &nvert);
  DMDAVecRestoreArray(da_c, *(user->lNvert_c), &nvert_c);

  DMDAVecRestoreArray(da_c, lX, &x);
  //  DMRestoreLocalVector(da_c, &lX);
  VecDestroy(&lX);
  DMDAVecRestoreArray(da,   F,  &f);

/*   PetscReal sum; */
/*   int N; */
/*   VecSum(F, &sum); */
/*   VecGetSize(F, &N); */
/*   sum = sum / (-1.*N); */
/*   VecShift(F, sum); */

/*   PetscReal max, min; */
/*   int  maxi, mini; */
/*   VecMax(X, &maxi, &max); */
/*   VecMin(X, &mini, &min); */
/*   PetscPrintf(PETSC_COMM_WORLD, "MM %d %le %d %le\n", maxi, max, mini, min); */
/*   PetscReal norm; */
/*   VecNorm(F, NORM_2, &norm); */
/*   PetscPrintf(PETSC_COMM_WORLD, "Interpolation Norm %le\n", norm); */

  return 0;
  //  MatNullSpaceRemove(da.nullsp, F, PETSC_NULL);
}

PetscErrorCode PoissonNullSpaceFunction(Vec X, void *ctx)
{
  UserCtx *user = (UserCtx*)ctx;

  DM da = user->da;

  DMDALocalInfo	info = user->info;
  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;
  int	lxs, lxe, lys, lye, lzs, lze;

  PetscReal	/* ***x,*/ ***nvert;
  int	i, j, k;
	
	PetscReal	***aj, ***gid;
	DMDAVecGetArray(user->da, user->lAj, &aj);
	DMDAVecGetArray(user->da, user->Gid, &gid);
	

/*   /\* First remove a constant from the Vec field X *\/ */
/*   MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, PETSC_NULL, &nullsp); */
/*   MatNullSpaceRemove(nullsp, X, PETSC_NULL); */
/*   MatNullSpaceDestroy(&nullsp); */

  /* Then apply boundary conditions */
  
  DMDAVecGetArray(da, user->lNvert, &nvert);

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  PetscReal lsum, sum;
  PetscReal  lnum, num;

  if (user->multinullspace) PetscPrintf(PETSC_COMM_WORLD, "MultiNullSpace!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  if (!user->multinullspace) {
    lsum = 0;
    lnum = 0;
    for (k=lzs; k<lze; k++) {
      for (j=lys; j<lye; j++) {
	for (i=lxs; i<lxe; i++) {
	  if (nvert[k][j][i] < 0.1) {
	    //lsum += x[k][j][i];
	    lnum ++;
	  }
	}
      }
    }
    //GlobalSum_All(&lsum, &sum, PETSC_COMM_WORLD);
    VecSum(X, &sum);
    GlobalSum_All(&lnum, &num, PETSC_COMM_WORLD);
    sum = sum / (-1.0 * num);
/*     PetscPrintf(PETSC_COMM_WORLD, "NullSpace %e\n", sum); */
    for (k=zs; k<ze; k++) {
      for (j=ys; j<ye; j++) {
	for (i=xs; i<xe; i++) {
	  if((int)nvert[k][j][i]==0 && i!=0 && i!=mx-1 && j!=0 && j!=my-1 && k!=0 && k!=mz-1) {
		//x[k][j][i] +=sum;
		double val=sum;
		VecSetValue(X, (int)gid[k][j][i], val, ADD_VALUES);
	  }
	}
      }
    }
  }
  VecAssemblyBegin(X);
  VecAssemblyEnd(X);
  
  DMDAVecRestoreArray(da, user->lNvert, &nvert);
  DMDAVecRestoreArray(user->da, user->lAj, &aj);
  DMDAVecRestoreArray(user->da, user->Gid, &gid);
  return 0;
}

PetscErrorCode PoissonNullSpaceFunction_original(MatNullSpace nsp, Vec X, void *ctx)
{
  UserCtx *user = (UserCtx*)ctx;

  DM da = user->da;

  DMDALocalInfo	info = user->info;
  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;
  int	lxs, lxe, lys, lye, lzs, lze;

  PetscReal	***x, ***nvert;
  int	i, j, k;

/*   /\* First remove a constant from the Vec field X *\/ */
/*   MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, PETSC_NULL, &nullsp); */
/*   MatNullSpaceRemove(nullsp, X, PETSC_NULL); */
/*   MatNullSpaceDestroy(&nullsp); */

  /* Then apply boundary conditions */
  DMDAVecGetArray(da, X, &x);
  DMDAVecGetArray(da, user->lNvert, &nvert);

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  PetscReal lsum, sum;
  PetscReal  lnum, num;

  if (user->multinullspace) PetscPrintf(PETSC_COMM_WORLD, "MultiNullSpace!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  if (!user->multinullspace) {
    lsum = 0;
    lnum = 0;
    for (k=lzs; k<lze; k++) {
      for (j=lys; j<lye; j++) {
	for (i=lxs; i<lxe; i++) {
	  if (nvert[k][j][i] < 0.1) {
	    lsum += x[k][j][i];
	    lnum ++;
	  }
	}
      }
    }
    GlobalSum_All(&lsum, &sum, PETSC_COMM_WORLD);
    GlobalSum_All(&lnum, &num, PETSC_COMM_WORLD);
    sum = sum / (-1.0 * num);
/*     PetscPrintf(PETSC_COMM_WORLD, "NullSpace %e\n", sum); */
    for (k=lzs; k<lze; k++) {
      for (j=lys; j<lye; j++) {
	for (i=lxs; i<lxe; i++) {
	  if (nvert[k][j][i] < 0.1) {
	    x[k][j][i] +=sum;
	  }
	}
      }
    }
  }
  else {
    lsum = 0;
    lnum = 0;
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	for (k=lzs; k<lze; k++) {
	  if (k<user->KSKE[2*(j*mx+i)] && nvert[k][j][i]<0.1) {
	    lsum += x[k][j][i];
	    lnum ++;
	  }
	}
      }
    }
    GlobalSum_All(&lsum, &sum, PETSC_COMM_WORLD);
    GlobalSum_All(&lnum, &num, PETSC_COMM_WORLD);
    sum /= -num;
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	for (k=lzs; k<lze; k++) {
	  if (k<user->KSKE[2*(j*mx+i)] && nvert[k][j][i]<0.1) {
	    x[k][j][i] += sum;
	  }
	}
      }
    }

    lsum = 0;
    lnum = 0;
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	for (k=lzs; k<lze; k++) {
	  if (k>=user->KSKE[2*(j*mx+i)] && nvert[k][j][i]<0.1) {
	    lsum += x[k][j][i];
	    lnum ++;
	  }
	}
      }
    }
    GlobalSum_All(&lsum, &sum, PETSC_COMM_WORLD);
    GlobalSum_All(&lnum, &num, PETSC_COMM_WORLD);
    sum /= -num;
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	for (k=lzs; k<lze; k++) {
	  if (k>=user->KSKE[2*(j*mx+i)] && nvert[k][j][i]<0.1) {
	    x[k][j][i] += sum;
	  }
	}
      }
    }
			   
  }
  if (zs == 0) {
    k = 0;
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
	x[k][j][i] = 0.;
      }
    }
  }

  if (ze == mz) {
    k = mz-1;
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
	x[k][j][i] = 0.;
      }
    }
  }

  if (ys == 0) {
    j = 0;
    for (k=zs; k<ze; k++) {
      for (i=xs; i<xe; i++) {
	x[k][j][i] = 0.;
      }
    }
  }

  if (ye == my) {
    j = my-1;
    for (k=zs; k<ze; k++) {
      for (i=xs; i<xe; i++) {
	x[k][j][i] = 0.;
      }
    }
  }

  if (xs == 0) {
    i = 0;
    for (k=zs; k<ze; k++) {
      for (j=ys; j<ye; j++) {
	x[k][j][i] = 0.;
      }
    }
  }

  if (xe == mx) {
    i = mx-1;
    for (k=zs; k<ze; k++) {
      for (j=ys; j<ye; j++) {
	x[k][j][i] = 0.;
      }
    }
  }

  for (k=zs; k<ze; k++) {
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
	if (nvert[k][j][i] > 0.1)
	  x[k][j][i] = 0.;
      }
    }
  }
  DMDAVecRestoreArray(da, X, &x);
  DMDAVecRestoreArray(da, user->lNvert, &nvert);

  return 0;
}

PetscErrorCode mymatmultadd(Mat mat, Vec v1, Vec v2, Vec v3)
{

  Vec vt;
  VecDuplicate(v3, &vt);
  MatMult(mat, v1, vt);
  VecWAXPY(v3, 1., v2, vt);
  VecDestroy(&vt);
  return(0);
}


#define CP 0
#define EP 1
#define WP 2
#define NP 3
#define SP 4
#define TP 5
#define BP 6
#define NE 7
#define SE 8
#define NW 9
#define SW 10
#define TN 11
#define BN 12
#define TS 13
#define BS 14
#define TE 15
#define BE 16
#define TW 17
#define BW 18

PetscErrorCode PoissonLHSNew(UserCtx *user, HYPRE_IJMatrix &Ap)
{
	DM		da = user->da, fda = user->fda;
	DMDALocalInfo	info = user->info;

	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;

	Cmpnts	***csi, ***eta, ***zet;
	Cmpnts	***icsi, ***ieta, ***izet;
	Cmpnts	***jcsi, ***jeta, ***jzet;
	Cmpnts	***kcsi, ***keta, ***kzet;

	PetscReal	***aj, ***iaj, ***jaj, ***kaj;

	int lxs, lxe, lys, lye, lzs, lze;
	int gxs, gxe, gys, gye, gzs, gze;

	Vec		G11, G12, G13, G21, G22, G23, G31, G32, G33;
	PetscReal	***g11, ***g12, ***g13, ***g21, ***g22, ***g23;
	PetscReal	***g31, ***g32, ***g33;
	PetscReal	***rho;

	PetscReal	***nvert, ***nvert_o, ***gid, ***level;
	PetscScalar	vol[27];
	//PetscInt idx[27], row;
	int idx[27], row;

	int	i, j, k, N;
	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;

	gxs = info.gxs; gxe = gxs + info.gxm;
	gys = info.gys; gye = gys + info.gym;
	gzs = info.gzs; gze = gzs + info.gzm;
	
	if (!user->assignedA) {
	  /*
		user->assignedA = PETSC_TRUE;
		
		setup_lidx2(user);
		N = mx * my * mz;
		int M;
	  */
		/*
		MatCreate(PETSC_COMM_WORLD, &(user->A));
		VecGetLocalSize(user->Phi2, &M);
		MatSetSizes(user->A,M,M,PETSC_DETERMINE,PETSC_DETERMINE);
		MatSetType(user->A,MATMPIAIJ);
		MatMPIAIJSetPreallocation(user->A, 19, PETSC_NULL, 19, PETSC_NULL);
		MatSetFromOptions(user->A);
		*/
	}

	//MatZeroEntries(user->A);

	if (levelset) {
		DMDAVecGetArray(da, user->lDensity, &rho);
		DMDAVecGetArray(da, user->lLevelset, &level);
	}
	
	DMDAVecGetArray(fda, user->lCsi, &csi);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lZet, &zet);

	DMDAVecGetArray(fda, user->lICsi, &icsi);
	DMDAVecGetArray(fda, user->lIEta, &ieta);
	DMDAVecGetArray(fda, user->lIZet, &izet);

	DMDAVecGetArray(fda, user->lJCsi, &jcsi);
	DMDAVecGetArray(fda, user->lJEta, &jeta);
	DMDAVecGetArray(fda, user->lJZet, &jzet);

	DMDAVecGetArray(fda, user->lKCsi, &kcsi);
	DMDAVecGetArray(fda, user->lKEta, &keta);
	DMDAVecGetArray(fda, user->lKZet, &kzet);

	DMDAVecGetArray(da, user->lAj, &aj);
	DMDAVecGetArray(da, user->lIAj, &iaj);
	DMDAVecGetArray(da, user->lJAj, &jaj);
	DMDAVecGetArray(da, user->lKAj, &kaj);

	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(da, user->lNvert_o, &nvert_o);

	VecDuplicate(user->lAj, &G11);
	VecDuplicate(user->lAj, &G12);
	VecDuplicate(user->lAj, &G13);
	VecDuplicate(user->lAj, &G21);
	VecDuplicate(user->lAj, &G22);
	VecDuplicate(user->lAj, &G23);
	VecDuplicate(user->lAj, &G31);
	VecDuplicate(user->lAj, &G32);
	VecDuplicate(user->lAj, &G33);
/*	
	VecSet(G11,1.e10);
	VecSet(G12,1.e10);
	VecSet(G13,1.e10);
	VecSet(G21,1.e10);
	VecSet(G22,1.e10);
	VecSet(G23,1.e10);
	VecSet(G31,1.e10);
	VecSet(G32,1.e10);
	VecSet(G33,1.e10);
*/
	DMDAVecGetArray(da, G11, &g11);
	DMDAVecGetArray(da, G12, &g12);
	DMDAVecGetArray(da, G13, &g13);
	DMDAVecGetArray(da, G21, &g21);
	DMDAVecGetArray(da, G22, &g22);
	DMDAVecGetArray(da, G23, &g23);
	DMDAVecGetArray(da, G31, &g31);
	DMDAVecGetArray(da, G32, &g32);
	DMDAVecGetArray(da, G33, &g33);
	
	/*for (k=gzs; k<gze; k++)
	for (j=gys; j<gye; j++)
	for (i=gxs; i<gxe; i++) */
	for (k=lzs-1; k<lze+1; k++)
	for (j=lys-1; j<lye+1; j++)
	for (i=lxs-1; i<lxe+1; i++) 
	{
		int a=i, b=j, c=k;
		
		
			int i_flag=0, j_flag=0, k_flag=0;
			
			if(i_periodic && i==0) a=mx-2, i_flag=1;
			else if(i_periodic && i==mx-1) a=1, i_flag=1;
			
			if(j_periodic && j==0) b=my-2, j_flag=1;
			else if(j_periodic && j==my-1) b=1, j_flag=1;
			
			if(k_periodic && k==0) c=mz-2, k_flag=1;
			else if(k_periodic && k==mz-1) c=1, k_flag=1;
			
			if(ii_periodic && i==0) a=-2, i_flag=1;
			else if(ii_periodic && i==mx-1) a=mx+1, i_flag=1;
			
			if(jj_periodic && j==0) b=-2, j_flag=1;
			else if(jj_periodic && j==my-1) b=my+1, j_flag=1;
			
			if(kk_periodic && k==0) c=-2, k_flag=1;
			else if(kk_periodic && k==mz-1) c=mz+1, k_flag=1;
			
		g11[k][j][i] = (icsi[c][b][a].x * icsi[c][b][a].x + icsi[c][b][a].y * icsi[c][b][a].y + icsi[c][b][a].z * icsi[c][b][a].z) * iaj[c][b][a];
		g12[k][j][i] = (ieta[c][b][a].x * icsi[c][b][a].x + ieta[c][b][a].y * icsi[c][b][a].y + ieta[c][b][a].z * icsi[c][b][a].z) * iaj[c][b][a];
		g13[k][j][i] = (izet[c][b][a].x * icsi[c][b][a].x + izet[c][b][a].y * icsi[c][b][a].y + izet[c][b][a].z * icsi[c][b][a].z) * iaj[c][b][a];
		g21[k][j][i] = (jcsi[c][b][a].x * jeta[c][b][a].x + jcsi[c][b][a].y * jeta[c][b][a].y + jcsi[c][b][a].z * jeta[c][b][a].z) * jaj[c][b][a];
		g22[k][j][i] = (jeta[c][b][a].x * jeta[c][b][a].x + jeta[c][b][a].y * jeta[c][b][a].y + jeta[c][b][a].z * jeta[c][b][a].z) * jaj[c][b][a];
		g23[k][j][i] = (jzet[c][b][a].x * jeta[c][b][a].x + jzet[c][b][a].y * jeta[c][b][a].y + jzet[c][b][a].z * jeta[c][b][a].z) * jaj[c][b][a];
		g31[k][j][i] = (kcsi[c][b][a].x * kzet[c][b][a].x + kcsi[c][b][a].y * kzet[c][b][a].y + kcsi[c][b][a].z * kzet[c][b][a].z) * kaj[c][b][a];
		g32[k][j][i] = (keta[c][b][a].x * kzet[c][b][a].x + keta[c][b][a].y * kzet[c][b][a].y + keta[c][b][a].z * kzet[c][b][a].z) * kaj[c][b][a];
		g33[k][j][i] = (kzet[c][b][a].x * kzet[c][b][a].x + kzet[c][b][a].y * kzet[c][b][a].y + kzet[c][b][a].z * kzet[c][b][a].z) * kaj[c][b][a];	
	}
	
	int m;
	DMDAVecGetArray(da, user->Gid, &gid);	// for macro gid2()

	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		double one=1.0;
		/*
		if(poisson==-1) row = lidx(i, j, k, user);
		else*/ 
		row = lidx2(i, j, k, user);

		if ( i == 0 || i == mx-1 || j==0 || j==my-1 || k==0 || k==mz-1 ) {
			if(poisson==-1) {
				if(periodic) row = lidx(i, j, k, user);	// tricky!!!!!!!!!!!!!!!!!!!!!!!! seokkoo
				//MatSetValues(user->A, 1, &row, 1, &row, &one, INSERT_VALUES);
			}
		}
		else if ( nvert[k][j][i]>0.1 && row>=0 ) {	// for fsi
			//PetscInt _row=lidx2(i, j, k, user);
			//MatSetValues(user->A, 1, &_row, 1, &_row, &one, INSERT_VALUES); 
		}
		else {
			if (nvert[k][j][i] > poisson_threshold) { // i, j, k is not a fluid point
				continue;
			}
			else { // i, j, k is a fluid point
				for (m=0; m<19; m++) vol[m] = 0.;
			    
				/* Contribution from i+1 - i */
				if (nvert[k][j][i+1] < poisson_threshold && (i != mx-2 || i_periodic || ii_periodic) ) { // i+1, j, k is a fluid point
					double r = 1.0;
					if (levelset) {
						r = mean ( rho[k][j][i+1], rho[k][j][i] );
					}
					
					/* dpdc{i} = (p_{i+1} - p_{i}) * g11_{i} */
					vol[CP] -= g11[k][j][i] / r; //i, j, k
					vol[EP] += g11[k][j][i] / r; // i+1, j, k
					
					/* dpde{i} = ({p_{i+1,j+1} + p{i, j+1} - p{i+1, j-1} - p{i, j-1}) * 0.25 * g12[k][j][i] */
					if ( (j == my-2 && !j_periodic && !jj_periodic) || nvert[k][j+1][i] + nvert[k][j+1][i+1] > poisson_threshold) {
						if (nvert[k][j-1][i] + nvert[k][j-1][i+1] < poisson_threshold && j!=1 ) {
							vol[CP] += g12[k][j][i] * 0.5 / r; //i, j, k
							vol[EP] += g12[k][j][i] * 0.5 / r; // i+1, j, k
							vol[SP] -= g12[k][j][i] * 0.5 / r; //i, j-1, k
							vol[SE] -= g12[k][j][i] * 0.5 / r; //i+1, j-1, k
						}
					}
					else if ( (j == 1 && !j_periodic && !jj_periodic) || nvert[k][j-1][i] + nvert[k][j-1][i+1] > poisson_threshold) {
						if (nvert[k][j+1][i] + nvert[k][j+1][i+1] < poisson_threshold) {
							vol[NP] += g12[k][j][i] * 0.5 / r;  //i, j+1, k
							vol[NE] += g12[k][j][i] * 0.5 / r; //i+1, j+1, k
							vol[CP] -= g12[k][j][i] * 0.5 / r; //i, j, k
							vol[EP] -= g12[k][j][i] * 0.5 / r; //i+1, j, k
						}
					}
					else {
						vol[NP] += g12[k][j][i] * 0.25 / r; // i, j+1, k
						vol[NE] += g12[k][j][i] * 0.25 / r; // i+1, j+1, k
						vol[SP] -= g12[k][j][i] * 0.25 / r; // i, j-1, k
						vol[SE] -= g12[k][j][i] * 0.25 / r; // i+1, j-1, k
						
					}
					/* dpdz{i}=(p_{i,k+1} + p{i+1,k+1} - p{i, k-1} - p{i+1, k-1}) * 0.25 / r * g13[k][j][i] */
					if ( (k == mz-2 && !k_periodic  && !kk_periodic) || nvert[k+1][j][i] + nvert[k+1][j][i+1] > poisson_threshold) {
						if (nvert[k-1][j][i] + nvert[k-1][j][i+1] < poisson_threshold && k!=1) {
							vol[CP] += g13[k][j][i] * 0.5 / r; // i, j, k
							vol[EP] += g13[k][j][i] * 0.5 / r; // i+1, j, k
							vol[BP] -= g13[k][j][i] * 0.5 / r; // i, j, k-1
							vol[BE] -= g13[k][j][i] * 0.5 / r; // i+1, j, k-1
						
						}
					}
					else if ( (k == 1 && !k_periodic && !kk_periodic) || nvert[k-1][j][i] + nvert[k-1][j][i+1] > poisson_threshold) {
						if (nvert[k+1][j][i] + nvert[k+1][j][i+1] < poisson_threshold) {
							vol[TP] += g13[k][j][i] * 0.5 / r;  // i, j, k+1
							vol[TE] += g13[k][j][i] * 0.5 / r; // i+1, j, k+1
							vol[CP] -= g13[k][j][i] * 0.5 / r;  // i, j, k
							vol[EP] -= g13[k][j][i] * 0.5 / r;  // i+1, j, k
						}
					}
					else {
						vol[TP] += g13[k][j][i] * 0.25 / r; //i, j, k+1
						vol[TE] += g13[k][j][i] * 0.25 / r; //i+1, j, k+1
						vol[BP] -= g13[k][j][i] * 0.25 / r; //i, j, k-1
						vol[BE] -= g13[k][j][i] * 0.25 / r; //i+1, j, k-1
						
					}
				}  // end of i+1 - i

				/* Contribution from i - i-1 */
				if (nvert[k][j][i-1] < poisson_threshold && (i != 1 || i_periodic || ii_periodic) ) { // i-1, j, k is a fluid point
					double r=1.0;
					if (levelset) {
						r = mean ( rho[k][j][i-1], rho[k][j][i] );
					}
					
					/* -dpdc{i-1} = -(p_{i} - p_{i-1}) * g11_{i} */
					vol[CP] -= g11[k][j][i-1] / r;  //i, j, k
					vol[WP] += g11[k][j][i-1] / r;  //i-1, j, k
					
					/* -dpde{i-1} = -({p_{i,j+1}+p{i-1, j+1} - p{i, j-1}-p{i-1, j-1}) * 0.25 / r * g12[k][j][i-1] */
					if ( (j == my-2 && !j_periodic && !jj_periodic) || nvert[k][j+1][i] + nvert[k][j+1][i-1] > poisson_threshold) {
						if (nvert[k][j-1][i] + nvert[k][j-1][i-1] < poisson_threshold && j!=1 ) {
							vol[CP] -= g12[k][j][i-1] * 0.5 / r; // i, j, k
							vol[WP] -= g12[k][j][i-1] * 0.5 / r; // i-1, j, k
							vol[SP] += g12[k][j][i-1] * 0.5 / r; //i, j-1, k
							vol[SW] += g12[k][j][i-1] * 0.5 / r; // i-1, j-1, k
						}
					}
					else if ( (j == 1 && !j_periodic && !jj_periodic) || nvert[k][j-1][i] + nvert[k][j-1][i-1] > poisson_threshold) {
						if (nvert[k][j+1][i] + nvert[k][j+1][i-1] < poisson_threshold) {
							vol[NP] -= g12[k][j][i-1] * 0.5 / r; // i, j+1, k
							vol[NW] -= g12[k][j][i-1] * 0.5 / r; // i-1, j+1, k
							vol[CP] += g12[k][j][i-1] * 0.5 / r; // i, j, k
							vol[WP] += g12[k][j][i-1] * 0.5 / r; // i-1, j, k
						}
					}
					else {
						vol[NP] -= g12[k][j][i-1] * 0.25 / r; // i, j+1, k
						vol[NW] -= g12[k][j][i-1] * 0.25 / r; //i-1, j+1, k
						vol[SP] += g12[k][j][i-1] * 0.25 / r; // i, j-1, k
						vol[SW] += g12[k][j][i-1] * 0.25 / r; // i-1, j-1, k
					}

					/* -dpdz{i-1}=-(p_{i,k+1}+p{i-1,k+1} - p{i, k-1}-p{i-1, k-1}) * 0.25 / r * g13[k][j][i] */
					if ( (k == mz-2 && !k_periodic && !kk_periodic) || nvert[k+1][j][i] + nvert[k+1][j][i-1] > poisson_threshold) {
						if (nvert[k-1][j][i] + nvert[k-1][j][i-1] < poisson_threshold && k!=1 ) {
							vol[CP] -= g13[k][j][i-1] * 0.5 / r; // i, j, k
							vol[WP] -= g13[k][j][i-1] * 0.5 / r; // i-1, j, k
							vol[BP] += g13[k][j][i-1] * 0.5 / r; // i, j, k-1
							vol[BW] += g13[k][j][i-1] * 0.5 / r; // i-1, j, k-1
						}
					}
					else if ( (k == 1 && !k_periodic && !kk_periodic) || nvert[k-1][j][i] + nvert[k-1][j][i-1] > poisson_threshold) {
						if (nvert[k+1][j][i] + nvert[k+1][j][i-1] < poisson_threshold) {
							vol[TP] -= g13[k][j][i-1] * 0.5 / r; // i, j, k+1
							vol[TW] -= g13[k][j][i-1] * 0.5 / r; //i-1, j, k+1
							vol[CP] += g13[k][j][i-1] * 0.5 / r; // i, j, k
							vol[WP] += g13[k][j][i-1] * 0.5 / r; //i-1, j, k
						}
					}
					else {
						vol[TP] -= g13[k][j][i-1] * 0.25 / r;  // i, j, k+1
						vol[TW] -= g13[k][j][i-1] * 0.25 / r; // i-1, j, k+1
						vol[BP] += g13[k][j][i-1] * 0.25 / r;  // i, j, k-1
						vol[BW] += g13[k][j][i-1] * 0.25 / r; // i-1, j, k-1
					}
				} // end of i - i-1

				/* Contribution from j+1 - j */
				if (nvert[k][j+1][i] < poisson_threshold && (j != my-2 || j_periodic || jj_periodic || (levelset && user->bctype[3]==-10 && j==my-2) ) ) {
					double r=1.0;
					if (levelset) {
						r = mean ( rho[k][j+1][i], rho[k][j][i] );
					}
										
					/* dpdc{j} = (p_{i+1,j}+p{i+1,j+1} - p{i-1,j}-p{i-1,j+1}) * 0.25 / r */
					if ( (i == mx-2 && !i_periodic && !ii_periodic) || nvert[k][j][i+1] + nvert[k][j+1][i+1] > poisson_threshold) {
						if (nvert[k][j][i-1] + nvert[k][j+1][i-1] < poisson_threshold && i!=1 ) {
							vol[CP] += g21[k][j][i] * 0.5 / r; // i, j, k
							vol[NP] += g21[k][j][i] * 0.5 / r; // i, j+1, k
							vol[WP] -= g21[k][j][i] * 0.5 / r; // i-1, j, k
							vol[NW] -= g21[k][j][i] * 0.5 / r; // i-1, j+1, k
						}
					}
					else if ( (i == 1 && !i_periodic && !ii_periodic) || nvert[k][j][i-1] + nvert[k][j+1][i-1] > poisson_threshold) {
						if (nvert[k][j][i+1] + nvert[k][j+1][i+1] < poisson_threshold) {
							vol[EP] += g21[k][j][i] * 0.5 / r; // i+1, j, k
							vol[NE] += g21[k][j][i] * 0.5 / r; // i+1, j+1, k
							vol[CP] -= g21[k][j][i] * 0.5 / r; // i, j, k
							vol[NP] -= g21[k][j][i] * 0.5 / r; // i, j+1, k
						}
					}
					else {
						vol[EP] += g21[k][j][i] * 0.25 / r; //i+1, j, k
						vol[NE] += g21[k][j][i] * 0.25 / r; //i+1, j+1, k
						vol[WP] -= g21[k][j][i] * 0.25 / r; //i-1, j, k
						vol[NW] -= g21[k][j][i] * 0.25 / r; //i-1, j+1, k
					}
					
					if( levelset && user->bctype[3]==-10 && j==my-2 ) {
						vol[CP] -= g22[k][j][i] / 0.5 / r;
					}
					else {
						/* dpde{j} = (p{j+1} - p{j}) * g22[k][j][i] */
						vol[CP] -= g22[k][j][i] / r;
						vol[NP] += g22[k][j][i] / r;
					}
					
					/* dpdz{j} = (p{j, k+1}+p{j+1,k+1} - p{j,k-1}-p{j+1,k-1}) *0.25 / r*/
					if ( (k == mz-2 && !k_periodic && !kk_periodic) || nvert[k+1][j][i] + nvert[k+1][j+1][i] > poisson_threshold) {
						if (nvert[k-1][j][i] + nvert[k-1][j+1][i] < poisson_threshold && k!=1 ) {
							vol[CP] += g23[k][j][i] * 0.5 / r; //i,j,k
							vol[NP] += g23[k][j][i] * 0.5 / r; //i, j+1, k
							vol[BP] -= g23[k][j][i] * 0.5 / r;//i, j, k-1
							vol[BN] -= g23[k][j][i] * 0.5 / r;//i, j+1, k-1
						}
					}
					else if ( (k == 1 && !k_periodic && !kk_periodic) || nvert[k-1][j][i] + nvert[k-1][j+1][i] > poisson_threshold) {
						if (nvert[k+1][j][i] + nvert[k+1][j+1][i] < poisson_threshold) {
							vol[TP] += g23[k][j][i] * 0.5 / r; //i, j, k+1
							vol[TN] += g23[k][j][i] * 0.5 / r;//i, j+1, k+1
							vol[CP] -= g23[k][j][i] * 0.5 / r;//i, j, k
							vol[NP] -= g23[k][j][i] * 0.5 / r;//i, j+1, k
						}
					}
					else {
						vol[TP] += g23[k][j][i] * 0.25 / r; // i, j, k+1
						vol[TN] += g23[k][j][i] * 0.25 / r; // i, j+1, k+1
						vol[BP] -= g23[k][j][i] * 0.25 / r; // i, j, k-1
						vol[BN] -= g23[k][j][i] * 0.25 / r; // i, j+1, k-1
					}
				} // End of j+1 - j

				/* Contribution j - j-1 */
				if (nvert[k][j-1][i] < poisson_threshold && (j!=1 || j_periodic || jj_periodic) ) {
					double r=1.0;
					if (levelset) {
						r = mean ( rho[k][j-1][i], rho[k][j][i] );
					}
					
					/* -dpdc{j-1} = -(p_{i+1,j}+p{i+1,j-1} - p{i-1,j}-p{i-1,j-1}) * 0.25 / r * g21[k][j-1][i] */
					if ( (i == mx-2 && !i_periodic && !ii_periodic) || nvert[k][j][i+1] + nvert[k][j-1][i+1] > poisson_threshold) {
						if (nvert[k][j][i-1] + nvert[k][j-1][i-1] < poisson_threshold && i!=1 ) {
							vol[CP] -= g21[k][j-1][i] * 0.5 / r;// i, j, k
							vol[SP] -= g21[k][j-1][i] * 0.5 / r;// i, j-1, k
							vol[WP] += g21[k][j-1][i] * 0.5 / r;// i-1, j, k
							vol[SW] += g21[k][j-1][i] * 0.5 / r;// i-1, j-1, k
						}
					}
					else if ( (i == 1 && !i_periodic && !ii_periodic) || nvert[k][j][i-1] + nvert[k][j-1][i-1] > poisson_threshold) {
						if (nvert[k][j][i+1] + nvert[k][j-1][i+1] < poisson_threshold) {
							vol[EP] -= g21[k][j-1][i] * 0.5 / r;//i+1, j, k
							vol[SE] -= g21[k][j-1][i] * 0.5 / r;//i+1, j-1, k
							vol[CP] += g21[k][j-1][i] * 0.5 / r;//i, j, k
							vol[SP] += g21[k][j-1][i] * 0.5 / r;//i, j-1, k
						}
					}
					else {
						vol[EP] -= g21[k][j-1][i] * 0.25 / r;// i+1, j, k
						vol[SE] -= g21[k][j-1][i] * 0.25 / r;// i+1, j-1, k
						vol[WP] += g21[k][j-1][i] * 0.25 / r;// i-1, j, k
						vol[SW] += g21[k][j-1][i] * 0.25 / r;// i-1, j-1, k
					}
			      
					/* -dpde{j-1} = -(p{j} - p{j-1}) * g22[k][j-1][i] */
					vol[CP] -= g22[k][j-1][i] / r;
					vol[SP] += g22[k][j-1][i] / r;

					/* -dpdz{j-1} = -(p{j,k+1}+p{j-1,k+1} - p{j,k-1}-p{j-1,k-1}) * 0.25 / r * g23[k][j-1][i] */
					if ( (k == mz-2 && !k_periodic && !kk_periodic) || nvert[k+1][j][i] + nvert[k+1][j-1][i] > poisson_threshold) {
						if (nvert[k-1][j][i] + nvert[k-1][j-1][i] < poisson_threshold && k!=1 ) {
							vol[CP] -= g23[k][j-1][i] * 0.5 / r;//i, j, k
							vol[SP] -= g23[k][j-1][i] * 0.5 / r;//i, j-1, k
							vol[BP] += g23[k][j-1][i] * 0.5 / r;//i, j, k-1
							vol[BS] += g23[k][j-1][i] * 0.5 / r;//i, j-1, k-1
						}
					}
					else if ( (k == 1 && !k_periodic && !kk_periodic) || nvert[k-1][j][i] + nvert[k-1][j-1][i] > poisson_threshold) {
						if (nvert[k+1][j][i] + nvert[k+1][j-1][i] < poisson_threshold) {
							vol[TP] -= g23[k][j-1][i] * 0.5 / r;//i, j, k+1
							vol[TS] -= g23[k][j-1][i] * 0.5 / r;//i, j-1, k+1
							vol[CP] += g23[k][j-1][i] * 0.5 / r;//i, j, k
							vol[SP] += g23[k][j-1][i] * 0.5 / r;//i, j-1, k
						}
					}
					else {
						vol[TP] -= g23[k][j-1][i] * 0.25 / r;//i, j, k+1
						vol[TS] -= g23[k][j-1][i] * 0.25 / r;//i, j-1, k+1
						vol[BP] += g23[k][j-1][i] * 0.25 / r;//i, j, k-1
						vol[BS] += g23[k][j-1][i] * 0.25 / r;//i, j-1, k-1
					}
				} // End of j - j-1

				/* contribution from k+1 - k */
				// xiaolei deactivate levelset option
				if (nvert[k+1][j][i] < poisson_threshold && (k != mz-2 || k_periodic || kk_periodic /*|| (levelset && user->bctype[5]==4 && k==mz-2)*/ ) ) {
					double r=1.0;
					if (levelset) {
						r = mean ( rho[k+1][j][i], rho[k][j][i] );
					}
					
					/* dpdc{k} = (p{i+1,k}+p{i+1,k+1} - p{i-1,k}-p{i-1,k+1}) * 0.25 / r * g31[k][j][i] */
					if ( (i == mx-2 && !i_periodic && !ii_periodic) || nvert[k][j][i+1] + nvert[k+1][j][i+1] > poisson_threshold) {
						if (nvert[k][j][i-1] + nvert[k+1][j][i-1] < poisson_threshold && i!=1 ) {
							vol[CP] += g31[k][j][i] * 0.5 / r;//i, j, k
							vol[TP] += g31[k][j][i] * 0.5 / r;//i, j, k+1
							vol[WP] -= g31[k][j][i] * 0.5 / r;//i-1, j, k
							vol[TW] -= g31[k][j][i] * 0.5 / r;//i-1, j, k+1
						}
					}
					else if ( (i == 1 && !i_periodic && !ii_periodic) || nvert[k][j][i-1] + nvert[k+1][j][i-1] > poisson_threshold) {
							if (nvert[k][j][i+1] + nvert[k+1][j][i+1] < poisson_threshold) {
								vol[EP] += g31[k][j][i] * 0.5 / r;//i+1, j, k
								vol[TE] += g31[k][j][i] * 0.5 / r;//i+1, j, k+1
								vol[CP] -= g31[k][j][i] * 0.5 / r;//i, j, k
								vol[TP] -= g31[k][j][i] * 0.5 / r;//i, j, k+1
							}
					}
					else {
						vol[EP] += g31[k][j][i] * 0.25 / r;//i+1, j, k
						vol[TE] += g31[k][j][i] * 0.25 / r;//i+1, j, k+1
						vol[WP] -= g31[k][j][i] * 0.25 / r;//i-1, j, k
						vol[TW] -= g31[k][j][i] * 0.25 / r;//i-1, j, k+1
					}

					/* dpde{k} = (p{j+1, k}+p{j+1,k+1} - p{j-1, k}-p{j-1,k+1}) * 0.25 / r * g32[k][j][i] */
					if ( (j == my-2 && !j_periodic && !jj_periodic) || nvert[k][j+1][i] + nvert[k+1][j+1][i] > poisson_threshold) {
						if (nvert[k][j-1][i] + nvert[k+1][j-1][i] < poisson_threshold && j!=1 ) {
							vol[CP] += g32[k][j][i] * 0.5 / r;//i, j,k
							vol[TP] += g32[k][j][i] * 0.5 / r;//i, j, k+1
							vol[SP] -= g32[k][j][i] * 0.5 / r;//i, j-1, k
							vol[TS] -= g32[k][j][i] * 0.5 / r;//i, j-1, k+1
						}
					}
					else if ( (j == 1 && !j_periodic && !jj_periodic) || nvert[k][j-1][i] + nvert[k+1][j-1][i] > poisson_threshold) {
						if (nvert[k][j+1][i] + nvert[k+1][j+1][i] < poisson_threshold) {
							vol[NP] += g32[k][j][i] * 0.5 / r;//i, j+1, k
							vol[TN] += g32[k][j][i] * 0.5 / r;//i, j+1, k+1
							vol[CP] -= g32[k][j][i] * 0.5 / r;//i, j, k
							vol[TP] -= g32[k][j][i] * 0.5 / r;//i, j, k+1
						}
					}
					else {
						vol[NP] += g32[k][j][i] * 0.25 / r;//i, j+1, k
						vol[TN] += g32[k][j][i] * 0.25 / r;//i, j+1, k+1
						vol[SP] -= g32[k][j][i] * 0.25 / r;//i, j-1, k
						vol[TS] -= g32[k][j][i] * 0.25 / r;//i, j-1, k+1
					}
				
					// xiaolei deactivate	
					// zero Dirichlet pressure 
					/*
					if( levelset && user->bctype[5]==4 && k==mz-2) {
					 	vol[CP] -= g33[k][j][i] / 0.5 / r;
					}
					else
					*/
					{
						/* dpdz{k} = p{k+1} - p{k} */
						vol[CP] -= g33[k][j][i] / r; //i, j, k
						vol[TP] += g33[k][j][i] / r; //i, j, k+1
					}
				} // End of k+1 - k

				/* Contribution from k - k-1 */
				if (nvert[k-1][j][i] < poisson_threshold && (k != 1 || k_periodic || kk_periodic) ) {
					double r=1.0;
					if (levelset) {
						if(k==1) r = rho[k][j][i];
						else r = mean ( rho[k-1][j][i], rho[k][j][i] );
					}
					
					/* -dpdc{k-1} = -(p{i+1,k}+p{i+1,k-1} - p{i-1,k}-p{i-1,k-1}) * 0.25 / r * g31[k-1][j][i] */
					if ( (i == mx-2 && !i_periodic && !ii_periodic) || nvert[k][j][i+1] + nvert[k-1][j][i+1] > poisson_threshold) {
						if (nvert[k][j][i-1] + nvert[k-1][j][i-1] < poisson_threshold && i!=1 ) {
							vol[CP] -= g31[k-1][j][i] * 0.5 / r;//i, j, k
							vol[BP] -= g31[k-1][j][i] * 0.5 / r;//i, j, k-1
							vol[WP] += g31[k-1][j][i] * 0.5 / r;//i-1, j, k
							vol[BW] += g31[k-1][j][i] * 0.5 / r;//i-1, j, k-1
						}
					}
					else if ( (i == 1 && !i_periodic && !ii_periodic) || nvert[k][j][i-1] + nvert[k-1][j][i-1] > poisson_threshold) {
						if (nvert[k][j][i+1] + nvert[k-1][j][i+1] < poisson_threshold) {
							vol[EP] -= g31[k-1][j][i] * 0.5 / r;//i+1, j, k
							vol[BE] -= g31[k-1][j][i] * 0.5 / r;//i+1, j, k-1
							vol[CP] += g31[k-1][j][i] * 0.5 / r;//i, j, k
							vol[BP] += g31[k-1][j][i] * 0.5 / r;//i, j, k-1
						}
					}
					else {
						vol[EP] -= g31[k-1][j][i] * 0.25 / r;//i+1, j, k
						vol[BE] -= g31[k-1][j][i] * 0.25 / r;//i+1, j, k-1
						vol[WP] += g31[k-1][j][i] * 0.25 / r;//i-1, j, k
						vol[BW] += g31[k-1][j][i] * 0.25 / r;//i-1, j, k-1
					}
			      
					/* -dpde{k-1} = -(p{j+1, k}+p{j+1,k-1} - p{j-1, k}-p{j-1,k-1}) *  0.25 / r * g32[k-1][j][i] */
					// ( p{i, j+1,k-1/2} - p{i, j-1,k-1/2} ) / (2eta)
					if ( (j == my-2 && !j_periodic && !jj_periodic) || nvert[k][j+1][i] + nvert[k-1][j+1][i] > poisson_threshold) {
						if (nvert[k][j-1][i] + nvert[k-1][j-1][i] < poisson_threshold && j!=1 ) {
							vol[CP] -= g32[k-1][j][i] * 0.5 / r;//i, j,k
							vol[BP] -= g32[k-1][j][i] * 0.5 / r;//i, j, k-1
							vol[SP] += g32[k-1][j][i] * 0.5 / r;//i, j-1, k 
							vol[BS] += g32[k-1][j][i] * 0.5 / r;//i, j-1, k-1
						}
					}
					else if ( (j == 1 && !j_periodic && !jj_periodic) || nvert[k][j-1][i] + nvert[k-1][j-1][i] > poisson_threshold) {
						if (nvert[k][j+1][i] + nvert[k-1][j+1][i] < poisson_threshold) {
							vol[NP] -= g32[k-1][j][i] * 0.5 / r;//i, j+1, k
							vol[BN] -= g32[k-1][j][i] * 0.5 / r;//i, j+1, k-1
							vol[CP] += g32[k-1][j][i] * 0.5 / r;//i, j, k
							vol[BP] += g32[k-1][j][i] * 0.5 / r;//i, j, k-1
						}
					}
					else {
						vol[NP] -= g32[k-1][j][i] * 0.25 / r;//i, j+1, k
						vol[BN] -= g32[k-1][j][i] * 0.25 / r;//i, j+1, k-1
						vol[SP] += g32[k-1][j][i] * 0.25 / r;//i, j-1, k
						vol[BS] += g32[k-1][j][i] * 0.25 / r;//i, j-1, k-1
					}

					// xiaolei deactivate
					// zero Dirichlet pressure 
                                        //if(inflow_levelset && levelset && user->bctype[4]==5 && k==1) {
					//	vol[CP] -= g33[k-1][j][i] / 0.5 / r;  
                                        //}
                                        //else
					{
					  /* -dpdz{k-1} = -(p{k} - p{k-1}) * g33[k-1][j][i] */
					  vol[CP] -= g33[k-1][j][i] / r; // i, j, k
					  vol[BP] += g33[k-1][j][i] / r; //i, j, k-1
					}
				} // End of k - k-1
				
				if(poisson==-1) for (m=0; m<19; m++)  vol[m] *= aj[k][j][i];
				/*
				if(poisson==-1) {
					idx[CP] = lidx(i  , j  , k  , user);
					idx[EP] = lidx(i+1, j  , k  , user);
					idx[WP] = lidx(i-1, j  , k  , user);
					idx[NP] = lidx(i  , j+1, k  , user);
					idx[SP] = lidx(i  , j-1, k  , user);
					idx[TP] = lidx(i  , j  , k+1, user);
					idx[BP] = lidx(i  , j  , k-1, user);
					idx[NE] = lidx(i+1, j+1, k  , user);
					idx[SE] = lidx(i+1, j-1, k  , user);
					idx[NW] = lidx(i-1, j+1, k  , user);
					idx[SW] = lidx(i-1, j-1, k  , user);
					idx[TN] = lidx(i  , j+1, k+1, user);
					idx[BN] = lidx(i  , j+1, k-1, user);
					idx[TS] = lidx(i  , j-1, k+1, user);
					idx[BS] = lidx(i  , j-1, k-1, user);
					idx[TE] = lidx(i+1, j  , k+1, user);
					idx[BE] = lidx(i+1, j  , k-1, user);
					idx[TW] = lidx(i-1, j  , k+1, user);
					idx[BW] = lidx(i-1, j  , k-1, user);
				}
				else*/ {
					idx[CP] = lidx2(i  , j  , k  , user);
					idx[EP] = lidx2(i+1, j  , k  , user);
					idx[WP] = lidx2(i-1, j  , k  , user);
					idx[NP] = lidx2(i  , j+1, k  , user);
					idx[SP] = lidx2(i  , j-1, k  , user);
					idx[TP] = lidx2(i  , j  , k+1, user);
					idx[BP] = lidx2(i  , j  , k-1, user);
					idx[NE] = lidx2(i+1, j+1, k  , user);
					idx[SE] = lidx2(i+1, j-1, k  , user);
					idx[NW] = lidx2(i-1, j+1, k  , user);
					idx[SW] = lidx2(i-1, j-1, k  , user);
					idx[TN] = lidx2(i  , j+1, k+1, user);
					idx[BN] = lidx2(i  , j+1, k-1, user);
					idx[TS] = lidx2(i  , j-1, k+1, user);
					idx[BS] = lidx2(i  , j-1, k-1, user);
					idx[TE] = lidx2(i+1, j  , k+1, user);
					idx[BE] = lidx2(i+1, j  , k-1, user);
					idx[TW] = lidx2(i-1, j  , k+1, user);
					idx[BW] = lidx2(i-1, j  , k-1, user);
				}
				
				//MatSetValues(user->A, 1, &row, 19, idx, vol, INSERT_VALUES);
				//for(m=0; m<19; m++) if( fabs(vol[m])>1.e-70 || m==CP ) MatSetValues(user->A, 1, &row, 1, &idx[m], &vol[m], INSERT_VALUES);
				for(m=0; m<19; m++) {
					if( (fabs(vol[m])>1.e-10 && idx[m]>=0) || m==CP ) {
						//MatSetValues(user->A, 1, &row, 1, &idx[m], &vol[m], INSERT_VALUES);
						//PetscInt nrows=1, ncols=1;
						int nrows=1, ncols=1;
						//PetscInt col=idx[m];
						int col=idx[m];
						HYPRE_IJMatrixSetValues(Ap, nrows, &ncols, &row, &col, &vol[m]); 
					}
				}

				/*
				if( (i==1 && j==2 && k==1) || (i==1 && j==2 && k==2)  || (i==1 && j==2 && k==3)  || (i==1 && j==2 && k==59) || (i==1 && j==2 && k==60) ) {
					printf("%d %d %d, my_idx = %d\n", i, j, k, idx[0]);
					for(m=0; m<19; m++) printf("%d\t", m);
					printf("\n");
					for(m=0; m<19; m++) printf("%d\t", idx[m]);
					printf("\n");
					for(m=0; m<19; m++) printf("%.2f\t", vol[m]);
					printf("\n\n");
				}*/
			} // End of fluid point
		} // End of interial points
	}
    //exit(0);
	DMDAVecRestoreArray(da, user->Gid, &gid);

	//kangsk  
	PetscPrintf(PETSC_COMM_WORLD, "MatAssemblyBegin...\n");
	//MatAssemblyBegin(user->A, MAT_FINAL_ASSEMBLY);
	//MatAssemblyEnd(user->A, MAT_FINAL_ASSEMBLY);
	HYPRE_IJMatrixAssemble(Ap);
	PetscPrintf(PETSC_COMM_WORLD, "MatAssemblyEnd...\n");

	DMDAVecRestoreArray(da, G11, &g11);
	DMDAVecRestoreArray(da, G12, &g12);
	DMDAVecRestoreArray(da, G13, &g13);
	DMDAVecRestoreArray(da, G21, &g21);
	DMDAVecRestoreArray(da, G22, &g22);
	DMDAVecRestoreArray(da, G23, &g23);
	DMDAVecRestoreArray(da, G31, &g31);
	DMDAVecRestoreArray(da, G32, &g32);
	DMDAVecRestoreArray(da, G33, &g33);
  
	VecDestroy(&G11);
	VecDestroy(&G12);
	VecDestroy(&G13);
	VecDestroy(&G21);
	VecDestroy(&G22);
	VecDestroy(&G23);
	VecDestroy(&G31);
	VecDestroy(&G32);
	VecDestroy(&G33);

	if (levelset) {
		DMDAVecRestoreArray(da, user->lDensity, &rho);
		DMDAVecRestoreArray(da, user->lLevelset, &level);
	}
	
	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lZet, &zet);

	DMDAVecRestoreArray(fda, user->lICsi, &icsi);
	DMDAVecRestoreArray(fda, user->lIEta, &ieta);
	DMDAVecRestoreArray(fda, user->lIZet, &izet);

	DMDAVecRestoreArray(fda, user->lJCsi, &jcsi);
	DMDAVecRestoreArray(fda, user->lJEta, &jeta);
	DMDAVecRestoreArray(fda, user->lJZet, &jzet);

	DMDAVecRestoreArray(fda, user->lKCsi, &kcsi);
	DMDAVecRestoreArray(fda, user->lKEta, &keta);
	DMDAVecRestoreArray(fda, user->lKZet, &kzet);

	DMDAVecRestoreArray(da, user->lAj, &aj);
	DMDAVecRestoreArray(da, user->lIAj, &iaj);
	DMDAVecRestoreArray(da, user->lJAj, &jaj);
	DMDAVecRestoreArray(da, user->lKAj, &kaj);

	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(da, user->lNvert_o, &nvert_o);
	
	return 0;
}


/*
PetscErrorCode PoissonLHSNew(UserCtx *user, IBMNodes *ibm, IBMInfo *ibminfo)
{
}
*/

PetscErrorCode PoissonRHS(UserCtx *user, Vec B)
{
  DMDALocalInfo info = user->info;
  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;

  int i, j, k;
  PetscReal	***nvert, ***aj, ***rb, ***gid, dt = user->dt;
  Cmpnts	***ucont;

 // DMDAVecGetArray(user->da, B, &rb);
  DMDAVecGetArray(user->fda, user->lUcont, &ucont);
  DMDAVecGetArray(user->da, user->lNvert, &nvert);
  DMDAVecGetArray(user->da, user->lAj, &aj);
  
	DMDAVecGetArray(user->da, user->Gid, &gid);
	

  for (k=zs; k<ze; k++) {
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
	      double val;
	if (i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1) {
		val = 0.;
	}
	else if (nvert[k][j][i] >= poisson_threshold) {
		val = 0;
	}
	else {
		double coeff=time_coeff();
		#ifdef DIRICHLET
		//if(freesurface && j==user->free_surface_j[i][k]) coeff *= user->vof[i][k];
		#endif
		val =- (ucont[k][j][i].x - ucont[k][j][i-1].x + ucont[k][j][i].y - ucont[k][j-1][i].y + ucont[k][j][i].z - ucont[k-1][j][i].z ) / dt * user->st * coeff;// * aj[k][j][i]; 
	}
	VecSetValue(B, lidx(i,j,k,user), val, INSERT_VALUES);
      }
    }
  }
  
  VecAssemblyBegin(B);
  VecAssemblyEnd(B);

  DMDAVecRestoreArray(user->da, user->Gid, &gid);
  
 // DMDAVecRestoreArray(user->da, B, &rb);

#ifndef DIRICHLET  
	PetscInt N;
	double sum0;
	VecGetSize(B,&N);
	VecSum(B,&sum0);
	sum0  = sum0/(-1.0*N);
	VecShift(B,sum0);
	user->multinullspace = PETSC_FALSE;
	PoissonNullSpaceFunction(B, user);
#endif 
  
  DMDAVecGetArray(user->da, B, &rb);
  
   PetscReal lsum=0, sum=0; 
   for (k=zs; k<ze; k++) { 
     for (j=ys; j<ye; j++) { 
       for (i=xs; i<xe; i++) { 
 	if (i==0 || i==mx-1 || j==0 || j==my-1 || 
 	    k==0 || k==mz-1) { 
 	  lsum += 0.; 
 	} 
 	else { 
 	  lsum += rb[k][j][i];// / aj[k][j][i]; 
 	} 
       } 
     } 
   }

   GlobalSum_All(&lsum, &sum, PETSC_COMM_WORLD); 
   PetscPrintf(PETSC_COMM_WORLD, "Summation RHS %e\n", sum); 
	
  DMDAVecRestoreArray(user->fda, user->lUcont, &ucont);
  DMDAVecRestoreArray(user->da, user->lNvert, &nvert);
  DMDAVecRestoreArray(user->da, user->lAj, &aj);
  DMDAVecRestoreArray(user->da, B, &rb);

  return 0;
}


PetscErrorCode PoissonRHS2(UserCtx *user, Vec B)
{
	DMDALocalInfo info = user->info;
	int	xs = info.xs, xe = info.xs + info.xm;
	int ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;

	int i, j, k;
	PetscReal	***nvert, ***aj, ***gid, dt = user->dt;
	Cmpnts	***ucont, ***lucont;
	
	int lxs = xs, lxe = xe;
	int lys = ys, lye = ye;
	int lzs = zs, lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
  	
	DMDAVecGetArray(user->fda, user->lUcont, &ucont);
	DMDAVecGetArray(user->da, user->lNvert, &nvert);
	DMDAVecGetArray(user->da, user->lAj, &aj);
  	DMDAVecGetArray(user->da, user->Gid, &gid);
	
	int lcount=0;
		
	VecSet(B, 0);
		
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		double val;
		if (nvert[k][j][i] >= poisson_threshold) {
			if( (int) (gid[k][j][i]) >=0 ) val = 0;	// for fsi
			else continue;
		}
		else {
			double coeff=time_coeff();
			#ifdef DIRICHLET
			//if(freesurface && j==user->free_surface_j[i][k]) coeff *= user->vof[i][k];
			#endif

			val=0;
			
			val -= ucont[k][j][i].x;

			if(i==1 && i_periodic) val += ucont[k][j][mx-2].x;
			else if(i==1 && ii_periodic)  val += ucont[k][j][-2].x;
			else val += ucont[k][j][i-1].x;

			val -= ucont[k][j][i].y;

			if(j==1 && j_periodic) val += ucont[k][my-2][i].y;
			else if(j==1 && jj_periodic) val += ucont[k][-2][i].y;
			else val += ucont[k][j-1][i].y;

			val -= ucont[k][j][i].z;

			if(k==1 && k_periodic) val += ucont[mz-2][j][i].z;
			else if(k==1 && kk_periodic) val += ucont[-2][j][i].z;
			else val += ucont[k-1][j][i].z;
			
			
		/*
			if(  (i==1 && j==2) ) {
				printf("%d %d %d, my_idx = %d, val=%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", i, j, k, 
					(int)gid[k][j][i], ucont[k][j][i-1].x, ucont[k][j][i].x, ucont[k][j-1][i].y, ucont[k][j][i].y, ucont[k-1][j][i].z, ucont[k][j][i].z);
				printf("\n");
			}*/
		
			val *=  -1.0 / dt * user->st * coeff;
			if(poisson==-1) val *= aj[k][j][i];
	
			lcount++;
		}
		VecSetValue(B, (int)gid[k][j][i], val, INSERT_VALUES);
		
		
	}
  //exit(0);
	VecAssemblyBegin(B);
	VecAssemblyEnd(B);

	
  
#ifndef DIRICHLET  
	
	
	double sum, sum1;
	VecSum(B, &sum);
	/*
	int N;
	VecGetSize(B, &N);
	sum  = sum/(-1.0*N);
	VecShift(B,sum);
	user->multinullspace = PETSC_FALSE;
	*/
	MPI_Allreduce( &lcount, &user->rhs_count, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
	
	double val = -sum/(double) (user->rhs_count);	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		if(nvert[k][j][i]<0.1) VecSetValue(B, (int)gid[k][j][i], val, ADD_VALUES);
	}
	
	VecAssemblyBegin(B);
	VecAssemblyEnd(B);
	VecSum(B, &sum1);
#endif 
  
	PetscPrintf(PETSC_COMM_WORLD, "Summation RHS %e %e\n", sum, sum1); 
	
	DMDAVecRestoreArray(user->da, user->Gid, &gid);
	DMDAVecRestoreArray(user->fda, user->lUcont, &ucont);
	DMDAVecRestoreArray(user->da, user->lNvert, &nvert);
	DMDAVecRestoreArray(user->da, user->lAj, &aj);
	
  

  return 0;
}


PetscErrorCode FullyBlocked(UserCtx *user)
{
  DM da = user->da;
  Vec nNvert;
  DMDALocalInfo info = user->info;
/*   int	xs = info.xs, xe = info.xs + info.xm; */
/*   int  	ys = info.ys, ye = info.ys + info.ym; */
/*   int	zs = info.zs, ze = info.zs + info.zm; */
  int	mx = info.mx, my = info.my, mz = info.mz;

  int i, j, k;

  int *KSKE = user->KSKE;
  PetscReal ***nvert;
  PetscBool *Blocked;

  DMDACreateNaturalVector(da, &nNvert);
  DMDAGlobalToNaturalBegin(da, user->Nvert, INSERT_VALUES, nNvert);
  DMDAGlobalToNaturalEnd(da, user->Nvert, INSERT_VALUES, nNvert);

  VecScatter ctx;
  Vec Zvert;
  VecScatterCreateToZero(nNvert, &ctx, &Zvert);

  VecScatterBegin(ctx, nNvert, Zvert, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(ctx, nNvert, Zvert, INSERT_VALUES, SCATTER_FORWARD);

  VecScatterDestroy(&ctx);
  VecDestroy(&nNvert);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  if (!rank) {

    VecGetArray3d(Zvert, mz, my, mx, 0, 0, 0, &nvert);
    PetscMalloc(mx*my*sizeof(PetscBool), &Blocked);
    for (j=1; j<my-1; j++) {
      for (i=1; i<mx-1; i++) {
	Blocked[j*mx+i] = PETSC_FALSE;
	for (k=0; k<mz; k++) {
	  if (nvert[k][j][i] > 0.1) {
	    if (!Blocked[j*mx+i]) {
	      KSKE[2*(j*mx+i)] = k;
	      Blocked[j*mx+i] = PETSC_TRUE;
	    }
	    else {
	      KSKE[2*(j*mx+i)] = PetscMin(KSKE[2*(j*mx+i)], k);
	    }
	  }
	}
      }
    }


    user->multinullspace = PETSC_TRUE;
    for (j=1; j<my-1; j++) {
      for (i=1; i<mx-1; i++) {
	if (!Blocked[j*mx+i]) {
	  user->multinullspace = PETSC_FALSE;
	  break;
	}
      }
    }
    PetscFree(Blocked);
    VecRestoreArray3d(Zvert, mz, my, mx, 0, 0, 0, &nvert);
    MPI_Bcast(&user->multinullspace, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    if (user->multinullspace) {
      MPI_Bcast(user->KSKE, 2*mx*my, MPI_INT, 0, PETSC_COMM_WORLD);

    }
  }
  else {
    MPI_Bcast(&user->multinullspace, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    if (user->multinullspace) {
      MPI_Bcast(user->KSKE, 2*mx*my, MPI_INT, 0, PETSC_COMM_WORLD);
    }
  }

/*   DMDACreateNaturalVector(da, &nNvert); */
/*   VecDestroy(&nNvert); */

  VecDestroy(&Zvert);
  return 0;
}

PetscErrorCode MyNvertRestriction(UserCtx *user_h, UserCtx *user_c)
{
  DMDALocalInfo	info = user_c->info;
  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;

  int i,j,k;
  int ih, jh, kh, ia, ja, ka;
  int	lxs, lxe, lys, lye, lzs, lze;

  PetscReal ***nvert, ***nvert_h;

  DMDAVecGetArray(user_h->da, user_h->lNvert, &nvert_h);
  DMDAVecGetArray(user_c->da, user_c->Nvert, &nvert);

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  if (*(user_c->isc)) ia = 0;
  else ia = 1;

  if (*(user_c->jsc)) ja = 0;
  else ja = 1;

  if (*(user_c->ksc)) ka = 0;
  else ka = 1;

  VecSet(user_c->Nvert, 0.);
  if (user_c->thislevel > 0) {
    for (k=lzs; k<lze; k++) {
      for (j=lys; j<lye; j++) {
	for (i=lxs; i<lxe; i++) {
	  GridRestriction(i, j, k, &ih, &jh, &kh, user_c);
	  if (nvert_h[kh   ][jh   ][ih   ] *
	      nvert_h[kh   ][jh   ][ih-ia] *
	      nvert_h[kh   ][jh-ja][ih   ] *
	      nvert_h[kh-ka][jh   ][ih   ] *
	      nvert_h[kh   ][jh-ja][ih-ia] *
	      nvert_h[kh-ka][jh   ][ih-ia] *
	      nvert_h[kh-ka][jh-ja][ih   ] *
	      nvert_h[kh-ka][jh-ja][ih-ia] > 0.1) {
	    nvert[k][j][i] = PetscMax(1., nvert[k][j][i]);
	  }
	}
      }
    }
  }
  else {
    for (k=lzs; k<lze; k++) {
      for (j=lys; j<lye; j++) {
	for (i=lxs; i<lxe; i++) {
	  GridRestriction(i, j, k, &ih, &jh, &kh, user_c);
	  if (nvert_h[kh   ][jh   ][ih   ] *
	      nvert_h[kh   ][jh   ][ih-ia] *
	      nvert_h[kh   ][jh-ja][ih   ] *
	      nvert_h[kh-ka][jh   ][ih   ] *
	      nvert_h[kh   ][jh-ja][ih-ia] *
	      nvert_h[kh-ka][jh   ][ih-ia] *
	      nvert_h[kh-ka][jh-ja][ih   ] *
	      nvert_h[kh-ka][jh-ja][ih-ia] > 0.1) {
	    nvert[k][j][i] = PetscMax(1., nvert[k][j][i]);
	  }
	}
      }
    }
  }
  DMDAVecRestoreArray(user_h->da, user_h->lNvert, &nvert_h);
  DMDAVecRestoreArray(user_c->da, user_c->Nvert, &nvert);

  DMGlobalToLocalBegin(user_c->da, user_c->Nvert, INSERT_VALUES, user_c->lNvert);
  DMGlobalToLocalEnd(user_c->da, user_c->Nvert, INSERT_VALUES, user_c->lNvert);
  DMDAVecGetArray(user_c->da, user_c->lNvert, &nvert);

  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	if (nvert[k][j][i] < 0.1) {
	  if (nvert[k][j][i+1] + nvert[k][j][i-1] > 1.1 &&
	      nvert[k][j+1][i] + nvert[k][j-1][i] > 1.1 &&
	      nvert[k+1][j][i] + nvert[k-1][j][i] > 1.1) {
	    nvert[k][j][i] = 1.;
	  }
	}
      }
    }
  }

  DMDAVecRestoreArray(user_c->da, user_c->lNvert, &nvert);
  DMLocalToGlobalBegin(user_c->da, user_c->lNvert, INSERT_VALUES, user_c->Nvert);
  DMLocalToGlobalEnd(user_c->da, user_c->lNvert, INSERT_VALUES, user_c->Nvert);
  return 0;
}


PetscErrorCode MyKSPMonitor1(KSP ksp,PetscInt n,PetscReal rnorm,void *dummy)
{
  //PetscPrintf(PETSC_COMM_WORLD,"     (%D) KSP Residual norm %14.12e \n",n,rnorm);
	KSPMonitorTrueResidualNorm(ksp, n, rnorm, dummy);
	return 0;
}

// Seokkoo Kang, August 2008
// Poisson solver based on algebraic multigrid
KSP ksp_amg[100];
int was_ksp_set=0;
int was_lhs_set=0;

PetscErrorCode Projection(UserCtx *user)
{
  DM		da = user->da, fda = user->fda;
  DMDALocalInfo	info = user->info;

  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;

  int	lxs, lys, lzs, lxe, lye, lze;
  int	i, j, k;

  Cmpnts	***csi, ***eta, ***zet;
  Cmpnts	***icsi, ***ieta, ***izet;
  Cmpnts	***jcsi, ***jeta, ***jzet;
  Cmpnts	***kcsi, ***keta, ***kzet;

  PetscReal	***aj, ***iaj, ***jaj, ***kaj;
  PetscReal	***phi, ***rho, ***p;

  Cmpnts	***ucont, ***lucont;
  PetscReal	***nvert, ***level;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;






  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;



	if (levelset) {
		DMDAVecGetArray(da, user->lDensity, &rho);
		DMDAVecGetArray(da, user->lLevelset, &level);
	}
	
	DMDAVecGetArray(fda, user->lCsi, &csi);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lZet, &zet);

	DMDAVecGetArray(fda, user->lICsi, &icsi);
	DMDAVecGetArray(fda, user->lIEta, &ieta);
	DMDAVecGetArray(fda, user->lIZet, &izet);

	DMDAVecGetArray(fda, user->lJCsi, &jcsi);
	DMDAVecGetArray(fda, user->lJEta, &jeta);
	DMDAVecGetArray(fda, user->lJZet, &jzet);

	DMDAVecGetArray(fda, user->lKCsi, &kcsi);
	DMDAVecGetArray(fda, user->lKEta, &keta);
	DMDAVecGetArray(fda, user->lKZet, &kzet);

	DMDAVecGetArray(da, user->lAj, &aj);
	DMDAVecGetArray(da, user->lIAj, &iaj);
	DMDAVecGetArray(da, user->lJAj, &jaj);
	DMDAVecGetArray(da, user->lKAj, &kaj);

	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(da, user->lPhi, &phi);
	DMDAVecGetArray(da, user->lP, &p);
  
	DMDAVecGetArray(fda, user->Ucont, &ucont);

	PetscReal dpdc, dpde, dpdz;
  
	Cmpnts ***cent;
	DMDAVecGetArray(fda, user->lCent, &cent);

	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
	      
		double coeff = time_coeff();
		double r1=1.0, r2=1.0, r3=1.0;
		
		if(i>0 && i<mx-1 && j>0 && j<my-1 && k>0 && k<mz-1)
		if(levelset) {
			if(i==mx-2 && !i_periodic && !ii_periodic) r1=rho[k][j][i];
			else r1 = mean ( rho[k][j][i], rho[k][j][i+1] );
			
			if(j==my-2 && !j_periodic && !jj_periodic) r2=rho[k][j][i];
			else r2 = mean ( rho[k][j][i], rho[k][j+1][i] );
			
			if(k==mz-2 && !k_periodic && !kk_periodic) r3=rho[k][j][i];
			else r3 = mean ( rho[k][j][i], rho[k+1][j][i] );
		}
		
		if(j>0 && j<my-1 && k>0 && k<mz-1)
		if ( (i>0 && i<mx-2) || ( (i_periodic || ii_periodic) && i==mx-2) ) {
			dpdc = phi[k][j][i+1] - phi[k][j][i];
			if( i==mx-2 && i_periodic) dpdc = phi[k][j][1] - phi[k][j][i];
			else if( i==mx-2 && ii_periodic) dpdc = phi[k][j][mx+1] - phi[k][j][i];

			dpde = 0.;
			dpdz = 0.;
			 
			// seokkoo - take into account Dirichlet pressure here !!
			if ( (j==my-2 && !j_periodic && !jj_periodic) || nvert[k][j+1][i]+nvert[k][j+1][i+1] > poisson_threshold) {
				if (nvert[k][j-1][i] + nvert[k][j-1][i+1] < poisson_threshold && (j!=1) ) {
					dpde = (phi[k][j  ][i] + phi[k][j  ][i+1] - phi[k][j-1][i] - phi[k][j-1][i+1]) * 0.5;
				}
			}
			else if ( (j == 1 && !j_periodic && !jj_periodic) || nvert[k][j-1][i] + nvert[k][j-1][i+1] > poisson_threshold) {
				if (nvert[k][j+1][i] + nvert[k][j+1][i+1] <poisson_threshold) {
					dpde = (phi[k][j+1][i] + phi[k][j+1][i+1] - phi[k][j  ][i] - phi[k][j  ][i+1]) * 0.5;
				}
			}
			else {
				dpde = (phi[k][j+1][i] + phi[k][j+1][i+1] - phi[k][j-1][i] - phi[k][j-1][i+1]) * 0.25;
			}

			if ( (k == mz-2 && !k_periodic && !kk_periodic) || nvert[k+1][j][i] + nvert[k+1][j][i+1] >poisson_threshold) {
				if (nvert[k-1][j][i] + nvert[k-1][j][i+1] < poisson_threshold && (k!=1) ) {
					dpdz = (phi[k  ][j][i] + phi[k  ][j][i+1] - phi[k-1][j][i] - phi[k-1][j][i+1]) * 0.5;
				}
			}
			else if ( (k == 1 && !k_periodic && !kk_periodic) || nvert[k-1][j][i] + nvert[k-1][j][i+1] >poisson_threshold) {
				if (nvert[k+1][j][i] + nvert[k+1][j][i+1] < poisson_threshold) {
					dpdz = (phi[k+1][j][i] + phi[k+1][j][i+1] - phi[k  ][j][i] - phi[k  ][j][i+1]) * 0.5;
				}
			}
			else {
				dpdz = (phi[k+1][j][i] + phi[k+1][j][i+1] - phi[k-1][j][i] - phi[k-1][j][i+1]) * 0.25;
			}
			
			if ((nvert[k][j][i] + nvert[k][j][i+1])<poisson_threshold) {	// how to take account ib interface velocity?
				ucont[k][j][i].x -= 
					(dpdc * (icsi[k][j][i].x * icsi[k][j][i].x + icsi[k][j][i].y * icsi[k][j][i].y + icsi[k][j][i].z * icsi[k][j][i].z) * iaj[k][j][i] +
					dpde * (ieta[k][j][i].x * icsi[k][j][i].x + ieta[k][j][i].y * icsi[k][j][i].y + ieta[k][j][i].z * icsi[k][j][i].z) * iaj[k][j][i] + 
					dpdz * (izet[k][j][i].x * icsi[k][j][i].x + izet[k][j][i].y * icsi[k][j][i].y + izet[k][j][i].z * icsi[k][j][i].z) * iaj[k][j][i]) * user->dt * user->st / coeff / r1;
			}
		}
		
		if(i>0 && i<mx-1 && k>0 && k<mz-1)
		if ( (j>0 && j<my-2) || ( (j_periodic || jj_periodic) && j==my-2) 
			|| (levelset && user->bctype[3]==-10 && j==my-2)
		) {
			dpdc = 0.;
			dpdz = 0.;
			if ( (i == mx-2 && !i_periodic && !ii_periodic) || nvert[k][j][i+1] + nvert[k][j+1][i+1] >poisson_threshold) {	
				if (nvert[k][j][i-1] + nvert[k][j+1][i-1] < poisson_threshold && (i!=1) ) {
					dpdc = (phi[k][j][i  ] + phi[k][j+1][i  ] - phi[k][j][i-1] - phi[k][j+1][i-1]) * 0.5;
				}
			}
			else if ( (i == 1 && !i_periodic && !ii_periodic) || nvert[k][j][i-1] + nvert[k][j+1][i-1] > poisson_threshold) {
				if (nvert[k][j][i+1] + nvert[k][j+1][i+1] < poisson_threshold) {
					dpdc = (phi[k][j][i+1] + phi[k][j+1][i+1] - phi[k][j][i  ] - phi[k][j+1][i  ]) * 0.5;
				}
			}
			else {
				dpdc = (phi[k][j][i+1] + phi[k][j+1][i+1] - phi[k][j][i-1] - phi[k][j+1][i-1]) * 0.25;
			}

			if( j==my-2 && j_periodic) dpde = phi[k][1][i] - phi[k][j][i];
			else if( j==my-2 && jj_periodic) dpde = phi[k][my+1][i] - phi[k][j][i];
			else if( j==my-2 && levelset && user->bctype[3]==-10 ) dpde = phi[k][j][i] - phi[k][j-1][i];
			else {
				dpde = phi[k][j+1][i] - phi[k][j][i];
			}

			if ( (k == mz-2 && !k_periodic && !kk_periodic) || nvert[k+1][j][i] + nvert[k+1][j+1][i] > poisson_threshold) {
				if (nvert[k-1][j][i] + nvert[k-1][j+1][i] < poisson_threshold && (k!=1) ) {
					dpdz = (phi[k  ][j][i] + phi[k  ][j+1][i] - phi[k-1][j][i] - phi[k-1][j+1][i]) * 0.5;
				}
			}
			else if ( (k == 1 && !k_periodic && !kk_periodic) || nvert[k-1][j][i] + nvert[k-1][j+1][i] > poisson_threshold) {
				if (nvert[k+1][j][i] + nvert[k+1][j+1][i] < poisson_threshold) {
					dpdz = (phi[k+1][j][i] + phi[k+1][j+1][i] - phi[k  ][j][i] - phi[k  ][j+1][i]) * 0.5;
				}
			}
			else {
				dpdz = (phi[k+1][j][i] + phi[k+1][j+1][i] - phi[k-1][j][i] - phi[k-1][j+1][i]) * 0.25;
			}
			
			if ((nvert[k][j][i] + nvert[k][j+1][i])<poisson_threshold) {
				ucont[k][j][i].y -=
					(dpdc * (jcsi[k][j][i].x * jeta[k][j][i].x + jcsi[k][j][i].y * jeta[k][j][i].y + jcsi[k][j][i].z * jeta[k][j][i].z) * jaj[k][j][i] +
					dpde * (jeta[k][j][i].x * jeta[k][j][i].x + jeta[k][j][i].y * jeta[k][j][i].y + jeta[k][j][i].z * jeta[k][j][i].z) * jaj[k][j][i] +
					dpdz * (jzet[k][j][i].x * jeta[k][j][i].x + jzet[k][j][i].y * jeta[k][j][i].y + jzet[k][j][i].z * jeta[k][j][i].z) * jaj[k][j][i]) * user->dt * user->st / coeff / r2;
			}
		}
		
		if (i>0 && i<mx-1 && j>0 && j<my-1)
		if ( (k>0 && k<mz-2) || 
			( (kk_periodic || k_periodic) && k==mz-2 ) /*||
			(levelset && user->bctype[5]==4 && k==mz-2)*/   // xiaolei deactivate  
		) {
			dpdc = 0.;
			dpde = 0.;
			if ( (i == mx-2 && !i_periodic && !ii_periodic) || nvert[k][j][i+1] + nvert[k+1][j][i+1] > poisson_threshold) {
				if (nvert[k][j][i-1] + nvert[k+1][j][i-1] < poisson_threshold && (i!=1) ) {
					dpdc = (phi[k][j][i  ] + phi[k+1][j][i  ] - phi[k][j][i-1] - phi[k+1][j][i-1]) * 0.5;
				}
			}
			else if ( (i == 1 && !i_periodic && !ii_periodic) || nvert[k][j][i-1] + nvert[k+1][j][i-1] > poisson_threshold) {
				if (nvert[k][j][i+1] + nvert[k+1][j][i+1] < poisson_threshold) {
					dpdc = (phi[k][j][i+1] + phi[k+1][j][i+1] - phi[k][j][i  ] - phi[k+1][j][i  ]) * 0.5;
				}
			}
			else {
				dpdc = (phi[k][j][i+1] + phi[k+1][j][i+1] - phi[k][j][i-1] - phi[k+1][j][i-1]) * 0.25;
			}
		  
			if ( (j == my-2 && !j_periodic && !jj_periodic) || nvert[k][j+1][i] + nvert[k+1][j+1][i] > poisson_threshold) {
				if (nvert[k][j-1][i] + nvert[k+1][j-1][i] < poisson_threshold && (j!=1) ) {
					dpde = (phi[k][j  ][i] + phi[k+1][j  ][i] - phi[k][j-1][i] - phi[k+1][j-1][i]) * 0.5;
				}
			}
			else if ( (j == 1 && !j_periodic && !jj_periodic) || nvert[k][j-1][i] + nvert[k+1][j-1][i] > poisson_threshold) {
				if (nvert[k][j+1][i] + nvert[k+1][j+1][i] <poisson_threshold) {
					dpde = (phi[k][j+1][i] + phi[k+1][j+1][i] - phi[k][j  ][i] - phi[k+1][j  ][i]) * 0.5;
				}
			}
			else {
				dpde = (phi[k][j+1][i] + phi[k+1][j+1][i] - phi[k][j-1][i] - phi[k+1][j-1][i]) * 0.25;
			}
			
			
			if( k==mz-2 && k_periodic) dpdz = phi[1][j][i] - phi[k][j][i];
			else if( k==mz-2 && kk_periodic) dpdz = phi[mz+1][j][i] - phi[k][j][i];
			// else if (k==mz-2 && levelset && user->bctype[5]==4) dpdz = phi[k][j][i] - phi[k-1][j][i];  // xiaolei deactivate 
			else dpdz = phi[k+1][j][i] - phi[k][j][i];
			
			if ((nvert[k][j][i] + nvert[k+1][j][i])<poisson_threshold) {
				ucont[k][j][i].z -=
						(dpdc * (kcsi[k][j][i].x * kzet[k][j][i].x + kcsi[k][j][i].y * kzet[k][j][i].y + kcsi[k][j][i].z * kzet[k][j][i].z) * kaj[k][j][i] +
						dpde * (keta[k][j][i].x * kzet[k][j][i].x + keta[k][j][i].y * kzet[k][j][i].y + keta[k][j][i].z * kzet[k][j][i].z) * kaj[k][j][i] +
						dpdz * (kzet[k][j][i].x * kzet[k][j][i].x + kzet[k][j][i].y * kzet[k][j][i].y + kzet[k][j][i].z * kzet[k][j][i].z) * kaj[k][j][i]) * user->dt *user->st / coeff / r3;
			}
		}
	}
    
	if (levelset) {
		DMDAVecRestoreArray(da, user->lDensity, &rho);
		DMDAVecRestoreArray(da, user->lLevelset, &level);
	}
  
	DMDAVecRestoreArray(fda, user->lCent, &cent);
  
	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lZet, &zet);

	DMDAVecRestoreArray(fda, user->lICsi, &icsi);
	DMDAVecRestoreArray(fda, user->lIEta, &ieta);
	DMDAVecRestoreArray(fda, user->lIZet, &izet);

	DMDAVecRestoreArray(fda, user->lJCsi, &jcsi);
	DMDAVecRestoreArray(fda, user->lJEta, &jeta);
	DMDAVecRestoreArray(fda, user->lJZet, &jzet);

	DMDAVecRestoreArray(fda, user->lKCsi, &kcsi);
	DMDAVecRestoreArray(fda, user->lKEta, &keta);
	DMDAVecRestoreArray(fda, user->lKZet, &kzet);

	DMDAVecRestoreArray(da, user->lAj, &aj);
	DMDAVecRestoreArray(da, user->lIAj, &iaj);
	DMDAVecRestoreArray(da, user->lJAj, &jaj);
	DMDAVecRestoreArray(da, user->lKAj, &kaj);
	DMDAVecRestoreArray(da, user->lPhi, &phi);
	DMDAVecRestoreArray(da, user->lP, &p);
	DMDAVecRestoreArray(fda, user->Ucont, &ucont);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);

	DMGlobalToLocalBegin(fda, user->Ucont, INSERT_VALUES, user->lUcont);
	DMGlobalToLocalEnd(fda, user->Ucont, INSERT_VALUES, user->lUcont);
	
	if(periodic) {
	  DMDAVecGetArray(user->fda, user->Ucont, &ucont);
	  DMDAVecGetArray(user->fda, user->lUcont, &lucont);
	
	  for (k=zs; k<ze; k++)
	    for (j=ys; j<ye; j++)
	      for (i=xs; i<xe; i++) {
		int i_flag=0, j_flag=0, k_flag=0;
		int a=i, b=j, c=k;
		
		if(i_periodic && i==0) a=mx-2, i_flag=1;
		else if(i_periodic && i==mx-1) a=1, i_flag=1;
		
		if(j_periodic && j==0) b=my-2, j_flag=1;
		else if(j_periodic && j==my-1) b=1, j_flag=1;
		
		if(k_periodic && k==0) c=mz-2, k_flag=1;
		else if(k_periodic && k==mz-1) c=1, k_flag=1;
		
		if(ii_periodic && i==0) a=-2, i_flag=1;
		else if(ii_periodic && i==mx-1) a=mx+1, i_flag=1;
		
		if(jj_periodic && j==0) b=-2, j_flag=1;
		else if(jj_periodic && j==my-1) b=my+1, j_flag=1;
		
		if(kk_periodic && k==0) c=-2, k_flag=1;
		else if(kk_periodic && k==mz-1) c=mz+1, k_flag=1;
		
		if(i_flag) lucont[k][j][i].x = lucont[c][b][a].x;
		if(j_flag) lucont[k][j][i].y = lucont[c][b][a].y;
		if(k_flag) lucont[k][j][i].z = lucont[c][b][a].z;
		
		ucont[k][j][i] = lucont[k][j][i];
	      }
	
	  DMDAVecRestoreArray(user->fda, user->Ucont, &ucont);
	  DMDAVecRestoreArray(user->fda, user->lUcont, &lucont);
	
	  DMGlobalToLocalBegin(user->fda, user->Ucont, INSERT_VALUES, user->lUcont); //101229
	  DMGlobalToLocalEnd(user->fda, user->Ucont, INSERT_VALUES, user->lUcont);
	}
	/*
	
	DMDAVecGetArray(fda, user->lUcont, &ucont);
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		if(i_periodic && i==0) ucont[k][j][0].x=ucont[k][j][mx-2].x;
		if(j_periodic && j==0) ucont[k][0][i].y=ucont[k][my-2][i].y;
		if(k_periodic && k==0) ucont[0][j][i].z=ucont[mz-2][j][i].z;
		
		if(ii_periodic && i==0) 	ucont[k][j][0].x=ucont[k][j][-2].x;
		if(jj_periodic && j==0) 	ucont[k][0][i].y=ucont[k][-2][i].y;
		if(kk_periodic && k==0) 	ucont[0][j][i].z=ucont[-2][j][i].z;
	}
	DMDAVecRestoreArray(fda, user->lUcont, &ucont);
	
	DMLocalToGlobal(fda, user->lUcont, INSERT_VALUES, user->Ucont);
	DMGlobalToLocalBegin(fda, user->Ucont, INSERT_VALUES, user->lUcont);
	DMGlobalToLocalEnd(fda, user->Ucont, INSERT_VALUES, user->lUcont);
  */
	
	//PetscBarrier(PETSC_NULL);

	Contra2Cart(user);
	
  //GhostNodeVelocity(user);
  //GhostNodeVelocity2(user);
  return(0);
}


PetscErrorCode UpdatePressure(UserCtx *user)
{
	DM		da = user->da, fda = user->fda;
	DMDALocalInfo	info = user->info;

	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;

	int	lxs, lys, lzs, lxe, lye, lze;
	int	i, j, k;

	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;

	Vec coords;	// seokkoo
	Cmpnts ***ucont, ***coor, ***ucat;
	Cmpnts ***csi, ***eta, ***zet;
	
	PetscReal ***p, ***phi, ***lp, ***lphi, ***aj, ***nvert, ***lnu_t;

	DMGlobalToLocalBegin(user->fda, user->Ucont, INSERT_VALUES, user->lUcont);
	DMGlobalToLocalEnd(user->fda, user->Ucont, INSERT_VALUES, user->lUcont);
    
	
	DMDAVecGetArray(da, user->P, &p);
	DMDAVecGetArray(da, user->Phi, &phi);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(fda, user->lUcont, &ucont); 
	DMDAVecGetArray(fda, user->lUcat, &ucat); 
	DMDAVecGetArray(da, user->lAj, &aj); 
   
	DMDAGetGhostedCoordinates(da, &coords);
	DMDAVecGetArray(fda, coords, &coor);

	DMDAVecGetArray(fda, user->lCsi, &csi);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lZet, &zet);
  
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		#ifdef	PRIMITIVE_PRESSURE
		p[k][j][i] = phi[k][j][i];
		
		#ifdef DIRICHLET
		double upper_y = 0.25 * ( coor[k  ][j  ][i  ].y + coor[k-1][j  ][i  ].y + coor[k  ][j  ][i-1].y +coor[k-1][j  ][i-1].y );
		double lower_y  = 0.25 * ( coor[k  ][j-1][i  ].y + coor[k-1][j-1][i  ].y + coor[k  ][j-1][i-1].y + coor[k-1][j-1][i-1].y );
		/*
			p (cent[k][j+1][i].y - upper_y ) + p_j+1 ( upper_y - cent[k][j][i].y ) = pD
			p ( lower_y - cent[k][j-1][i].y ) + p_j-1 ( cent[k][j][i].y - lower_y ) = pD
		*/
		      /*
		if( freesurface && (int)nvert[k][j][i]==5 && (int)nvert[k][j-1][i]==0 ) {
			double AB = cent[k][j][i].y - cent[k][j-1][i].y;
			double AS = free_surface_y[i][k] - cent[k][j-1][i].y;
			assert(AS>=0);
			
			double D = cent[k][j][i].y - cent[k][j-1][i].y;
			p[k][j][i] = (0 - phi[k][j-1][i]) * ( cent[k][j][i].y - lower_y ) / ( lower_y - cent[k][j-1][i].y );
		}
		*/
		#endif
		
		#else
		if( nvert[k][j][i]<0.1) p[k][j][i] += phi[k][j][i];
		//if( nvert[k][j][i]>2.1 ) p[k][j][i] = 0;
		//p[k][j][i] -= (ucont[k][j][i].x - ucont[k][j][i-1].x + ucont[k][j][i].y - ucont[k][j-1][i].y + ucont[k][j][i].z - ucont[k-1][j][i].z ) * aj[k][j][i] / user->ren; 
		/*
		double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
		double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
		
		double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
		double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
		double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
		double ajc = aj[k][j][i];
		
		Compute_du_center (i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
		Compute_du_dxyz (csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
		
		p[k][j][i] -= (du_dx + dv_dy + dw_dz) / user->ren; 
		*/
		#endif
	}
    
	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lZet, &zet);
	
	DMDAVecRestoreArray(da, user->Phi, &phi);
	DMDAVecRestoreArray(da, user->P, &p);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(fda, user->lUcont, &ucont); 
	DMDAVecRestoreArray(fda, user->lUcat, &ucat); 
	DMDAVecRestoreArray(da, user->lAj, &aj); 
	DMDAVecRestoreArray(fda, coords, &coor);
  
	DMGlobalToLocalBegin(da, user->P, INSERT_VALUES, user->lP);
	DMGlobalToLocalEnd(da, user->P, INSERT_VALUES, user->lP);
	
	DMDAVecGetArray(da, user->lP, &lp);
	DMDAVecGetArray(da, user->lPhi, &lphi);
	DMDAVecGetArray(da, user->P, &p);
	DMDAVecGetArray(da, user->Phi, &phi);
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
		int flag=0, a=i, b=j, c=k;
		
		if(i_periodic && i==0) a=mx-2, flag=1;
		else if(i_periodic && i==mx-1) a=1, flag=1;
		
		if(j_periodic && j==0) b=my-2, flag=1;
		else if(j_periodic && j==my-1) b=1, flag=1;
		
		if(k_periodic && k==0) c=mz-2, flag=1;
		else if(k_periodic && k==mz-1) c=1, flag=1;
		
		if(ii_periodic && i==0) a=-2, flag=1;
		else if(ii_periodic && i==mx-1) a=mx+1, flag=1;
		
		if(jj_periodic && j==0) b=-2, flag=1;
		else if(jj_periodic && j==my-1) b=my+1, flag=1;
		
		if(kk_periodic && k==0) c=-2, flag=1;
		else if(kk_periodic && k==mz-1) c=mz+1, flag=1;
		
		if(flag) {
			p[k][j][i] = lp[c][b][a];
			phi[k][j][i] = lphi[c][b][a];
		}
		
	}
	DMDAVecRestoreArray(da, user->lP, &lp);
	DMDAVecRestoreArray(da, user->lPhi, &lphi);
	DMDAVecRestoreArray(da, user->P, &p);
	DMDAVecRestoreArray(da, user->Phi, &phi);
	/*
	DMLocalToGlobal(da, user->lP, INSERT_VALUES, user->P);
	DMLocalToGlobal(da, user->lPhi, INSERT_VALUES, user->Phi);
	*/
	DMGlobalToLocalBegin(da, user->P, INSERT_VALUES, user->lP);
	DMGlobalToLocalEnd(da, user->P, INSERT_VALUES, user->lP);
		
	DMGlobalToLocalBegin(da, user->Phi, INSERT_VALUES, user->lPhi);
	DMGlobalToLocalEnd(da, user->Phi, INSERT_VALUES, user->lPhi);
	
	
  return 0;
}

PetscErrorCode VolumeFlux(UserCtx *user, Vec lUcor, PetscReal *ibm_Flux, PetscReal *ibm_Area)
{
  DM da = user->da, fda = user->fda;

  DMDALocalInfo	info = user->info;

  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;

  int i, j, k;
  int	lxs, lys, lzs, lxe, lye, lze;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  PetscReal ***nvert;
  Cmpnts ***ucor, ***csi, ***eta, ***zet;
  DMDAVecGetArray(fda, lUcor, &ucor);
  DMDAVecGetArray(fda, user->lCsi, &csi);
  DMDAVecGetArray(fda, user->lEta, &eta);
  DMDAVecGetArray(fda, user->lZet, &zet);
  DMDAVecGetArray(da, user->lNvert, &nvert);

  PetscReal libm_Flux, libm_area;
  
  libm_Flux = 0;
  libm_area = 0;
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	if (nvert[k][j][i] < poisson_threshold) {
	  if (nvert[k][j][i+1] > poisson_threshold && i <= mx-2) {
	    libm_Flux += ucor[k][j][i].x;
	    libm_area += sqrt(csi[k][j][i].x * csi[k][j][i].x + csi[k][j][i].y * csi[k][j][i].y + csi[k][j][i].z * csi[k][j][i].z);
	  }
	  if (nvert[k][j+1][i] > poisson_threshold && j <= my-2) {
	    libm_Flux += ucor[k][j][i].y;
	    libm_area += sqrt(eta[k][j][i].x * eta[k][j][i].x + eta[k][j][i].y * eta[k][j][i].y +  eta[k][j][i].z * eta[k][j][i].z);
	  }
	  if (nvert[k+1][j][i] > poisson_threshold && k <= mz-2) {
	    libm_Flux += ucor[k][j][i].z;
	    libm_area += sqrt(zet[k][j][i].x * zet[k][j][i].x + zet[k][j][i].y * zet[k][j][i].y + zet[k][j][i].z * zet[k][j][i].z);
	  }
	}

	if (nvert[k][j][i] > poisson_threshold ) {
	  if (nvert[k][j][i+1] < poisson_threshold && i <= mx-2) {
	    libm_Flux -= ucor[k][j][i].x;
	    libm_area += sqrt(csi[k][j][i].x * csi[k][j][i].x + csi[k][j][i].y * csi[k][j][i].y + csi[k][j][i].z * csi[k][j][i].z);
			  
	  }
	  if (nvert[k][j+1][i] < poisson_threshold && j <= my-2) {
	    libm_Flux -= ucor[k][j][i].y;
	    libm_area += sqrt(eta[k][j][i].x * eta[k][j][i].x + eta[k][j][i].y * eta[k][j][i].y + eta[k][j][i].z * eta[k][j][i].z);
	  }
	  if (nvert[k+1][j][i] < poisson_threshold && k <= mz-2) {
	    libm_Flux -= ucor[k][j][i].z;
	    libm_area += sqrt(zet[k][j][i].x * zet[k][j][i].x + zet[k][j][i].y * zet[k][j][i].y + zet[k][j][i].z * zet[k][j][i].z);
	  }
	}

      }
    }
  }
  GlobalSum_All(&libm_Flux, ibm_Flux, PETSC_COMM_WORLD);
  GlobalSum_All(&libm_area, ibm_Area, PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "IBMFlux %le %le\n", *ibm_Flux, *ibm_Area);

  PetscReal correction;

  if (*ibm_Area > 1.e-15) {
    correction = *ibm_Flux / *ibm_Area;
  }
  else {
    correction = 0;
  }

	if(immersed!=2)
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	if (nvert[k][j][i] < poisson_threshold) {
	  if (nvert[k][j][i+1] > poisson_threshold && i <= mx-2) {
	    ucor[k][j][i].x -= sqrt(csi[k][j][i].x * csi[k][j][i].x + csi[k][j][i].y * csi[k][j][i].y + csi[k][j][i].z * csi[k][j][i].z) * correction;
		if(immersed==3 || !immersed) ucor[k][j][i].x=0;
	  }
	  if (nvert[k][j+1][i] > poisson_threshold && j <= my-2) {
	    ucor[k][j][i].y -= sqrt(eta[k][j][i].x * eta[k][j][i].x + eta[k][j][i].y * eta[k][j][i].y + eta[k][j][i].z * eta[k][j][i].z) * correction;
		if(immersed==3 || !immersed) ucor[k][j][i].y=0;
	  }
	  if (nvert[k+1][j][i] > poisson_threshold && k <= mz-2) {
	    ucor[k][j][i].z -= sqrt(zet[k][j][i].x * zet[k][j][i].x + zet[k][j][i].y * zet[k][j][i].y + zet[k][j][i].z * zet[k][j][i].z) * correction;
		if(immersed==3 || !immersed) ucor[k][j][i].z=0;
	  }
	}

	if (nvert[k][j][i] > poisson_threshold ) {
	  if (nvert[k][j][i+1] < poisson_threshold && i <= mx-2) {
	    ucor[k][j][i].x += sqrt(csi[k][j][i].x * csi[k][j][i].x + csi[k][j][i].y * csi[k][j][i].y + csi[k][j][i].z * csi[k][j][i].z) * correction;
		if(immersed==3 || !immersed) ucor[k][j][i].x=0;
	  }
	  if (nvert[k][j+1][i] < poisson_threshold && j <= my-2) {
	    ucor[k][j][i].y += sqrt(eta[k][j][i].x * eta[k][j][i].x + eta[k][j][i].y * eta[k][j][i].y + eta[k][j][i].z * eta[k][j][i].z) * correction;
		if(immersed==3 || !immersed) ucor[k][j][i].y=0;
	  }
	  if (nvert[k+1][j][i] < poisson_threshold && k <= mz-2) {
	    ucor[k][j][i].z += sqrt(zet[k][j][i].x * zet[k][j][i].x + zet[k][j][i].y * zet[k][j][i].y + zet[k][j][i].z * zet[k][j][i].z) * correction;
		if(immersed==3 || !immersed) ucor[k][j][i].z=0;
	  }
	}

      }
    }
  }
  


  libm_Flux = 0;
  libm_area = 0;
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	if (nvert[k][j][i] < poisson_threshold) {
	  if (nvert[k][j][i+1] > poisson_threshold && i <= mx-2) {
	    libm_Flux += ucor[k][j][i].x;
	    libm_area += sqrt(csi[k][j][i].x * csi[k][j][i].x +
			  csi[k][j][i].y * csi[k][j][i].y +
			  csi[k][j][i].z * csi[k][j][i].z);
			  
	  }
	  if (nvert[k][j+1][i] > poisson_threshold && j <= my-2) {
	    libm_Flux += ucor[k][j][i].y;
	    libm_area += sqrt(eta[k][j][i].x * eta[k][j][i].x +
			  eta[k][j][i].y * eta[k][j][i].y +
			  eta[k][j][i].z * eta[k][j][i].z);
	  }
	  if (nvert[k+1][j][i] > poisson_threshold && k <= mz-2) {
	    libm_Flux += ucor[k][j][i].z;
	    libm_area += sqrt(zet[k][j][i].x * zet[k][j][i].x +
			  zet[k][j][i].y * zet[k][j][i].y +
			  zet[k][j][i].z * zet[k][j][i].z);
	  }
	}

	if (nvert[k][j][i] > poisson_threshold ) {
	  if (nvert[k][j][i+1] < poisson_threshold && i <= mx-2) {
	    libm_Flux -= ucor[k][j][i].x;
	    libm_area += sqrt(csi[k][j][i].x * csi[k][j][i].x +
			  csi[k][j][i].y * csi[k][j][i].y +
			  csi[k][j][i].z * csi[k][j][i].z);
			  
	  }
	  if (nvert[k][j+1][i] < poisson_threshold && j <= my-2) {
	    libm_Flux -= ucor[k][j][i].y;
	    libm_area += sqrt(eta[k][j][i].x * eta[k][j][i].x +
			  eta[k][j][i].y * eta[k][j][i].y +
			  eta[k][j][i].z * eta[k][j][i].z);
	  }
	  if (nvert[k+1][j][i] < poisson_threshold && k <= mz-2) {
	    libm_Flux -= ucor[k][j][i].z;
	    libm_area += sqrt(zet[k][j][i].x * zet[k][j][i].x +
			  zet[k][j][i].y * zet[k][j][i].y +
			  zet[k][j][i].z * zet[k][j][i].z);
	  }
	}

      }
    }
  }
  
  GlobalSum_All(&libm_Flux, ibm_Flux, PETSC_COMM_WORLD);
  GlobalSum_All(&libm_area, ibm_Area, PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "IBMFlux22 %le %le\n", *ibm_Flux, *ibm_Area);

  DMDAVecRestoreArray(da, user->lNvert, &nvert);
  DMDAVecRestoreArray(fda, user->lCsi, &csi);
  DMDAVecRestoreArray(fda, user->lEta, &eta);
  DMDAVecRestoreArray(fda, user->lZet, &zet);
  DMDAVecRestoreArray(fda, lUcor, &ucor);

  DMGlobalToLocalBegin(fda, lUcor, INSERT_VALUES, user->lUcont);
  DMGlobalToLocalEnd(fda, lUcor, INSERT_VALUES, user->lUcont);
  return 0;
}

void Add_IBMFlux_to_Outlet(UserCtx *user, PetscReal ibm_Flux)
{
	if(user->bctype[5]!=4) return;
		
	DM da = user->da, fda = user->fda;

	DMDALocalInfo	info = user->info;

	int xs = info.xs, xe = info.xs + info.xm;
	int ys = info.ys, ye = info.ys + info.ym;
	int zs = info.zs, ze = info.zs + info.zm;
	int mx = info.mx, my = info.my, mz = info.mz;

	int i, j, k;
	int lxs, lys, lzs, lxe, lye, lze;

	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
		
	PetscReal ***nvert;
	Cmpnts ***ucont, ***csi, ***eta, ***zet;
	
	DMDAVecGetArray(fda, user->Ucont, &ucont);
	DMDAVecGetArray(fda, user->lCsi, &csi);
	DMDAVecGetArray(fda, user->lEta, &eta);
	DMDAVecGetArray(fda, user->lZet, &zet);
	DMDAVecGetArray(da, user->lNvert, &nvert);
  
	double AreaSum=0, lArea=0;
	if(ze==mz) {
		k = ze-1;
		for (j=lys; j<lye; j++) 
		for (i=lxs; i<lxe; i++) {
			if (nvert[k-1][j][i] < 0.1) {
				lArea += sqrt( zet[k-1][j][i].x*zet[k-1][j][i].x + zet[k-1][j][i].y*zet[k-1][j][i].y + zet[k-1][j][i].z*zet[k-1][j][i].z );
			}
		}
	}
	GlobalSum_All(&lArea, &AreaSum, PETSC_COMM_WORLD);
	
	if(ze==mz) {
		k = ze-1;
		for (j=lys; j<lye; j++) 
		for (i=lxs; i<lxe; i++) {
			if (nvert[k-1][j][i] < 0.1) {
				double A = sqrt( zet[k-1][j][i].x*zet[k-1][j][i].x + zet[k-1][j][i].y*zet[k-1][j][i].y + zet[k-1][j][i].z*zet[k-1][j][i].z );
				ucont[k-1][j][i].z += ibm_Flux * A / AreaSum;
			}
		}
	}

	DMDAVecRestoreArray(fda, user->Ucont, &ucont);
	DMDAVecRestoreArray(fda, user->lCsi, &csi);
	DMDAVecRestoreArray(fda, user->lEta, &eta);
	DMDAVecRestoreArray(fda, user->lZet, &zet);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
}


PetscErrorCode VolumeFlux(UserCtx *user, PetscReal *ibm_Flux, PetscReal *ibm_Area, int flg)
{
  DM da = user->da, fda = user->fda;

  DMDALocalInfo	info = user->info;

  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;

  int i, j, k;
  int	lxs, lys, lzs, lxe, lye, lze;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;
  
  PetscReal epsilon=1.e-10;//1.e-8;
  PetscReal ***nvert, ibmval=1.1;
  Cmpnts ***ucor, ***csi, ***eta, ***zet;
  DMDAVecGetArray(fda, user->Ucont, &ucor);
  DMDAVecGetArray(fda, user->lCsi, &csi);
  DMDAVecGetArray(fda, user->lEta, &eta);
  DMDAVecGetArray(fda, user->lZet, &zet);
  DMDAVecGetArray(da, user->lNvert, &nvert);

  PetscReal libm_Flux, libm_area, libm_Flux_abs=0., ibm_Flux_abs;
  libm_Flux = 0;
  libm_area = 0;
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	if (nvert[k][j][i] < 0.1) {
	  if (nvert[k][j][i+1] > 0.1 && nvert[k][j][i+1] < ibmval && i < mx-2) {
	    if (fabs(ucor[k][j][i].x)>epsilon) {
	    libm_Flux += ucor[k][j][i].x;
	    if (flg==3) 
	      libm_Flux_abs += fabs(ucor[k][j][i].x)/sqrt(csi[k][j][i].x * csi[k][j][i].x +
							  csi[k][j][i].y * csi[k][j][i].y +
							  csi[k][j][i].z * csi[k][j][i].z);
	    else
	      libm_Flux_abs += fabs(ucor[k][j][i].x);
	    libm_area += sqrt(csi[k][j][i].x * csi[k][j][i].x +
			  csi[k][j][i].y * csi[k][j][i].y +
			  csi[k][j][i].z * csi[k][j][i].z);
	    } else 
	      ucor[k][j][i].x=0.;
			  
	  }
	  if (nvert[k][j+1][i] > 0.1 && nvert[k][j+1][i] < ibmval && j < my-2) {
	    if (fabs(ucor[k][j][i].y)>epsilon) {
	    libm_Flux += ucor[k][j][i].y;
	    if (flg==3) 
	      libm_Flux_abs += fabs(ucor[k][j][i].y)/sqrt(eta[k][j][i].x * eta[k][j][i].x +
							  eta[k][j][i].y * eta[k][j][i].y +
							  eta[k][j][i].z * eta[k][j][i].z);
	    else
	      libm_Flux_abs += fabs(ucor[k][j][i].y);
	    libm_area += sqrt(eta[k][j][i].x * eta[k][j][i].x +
			  eta[k][j][i].y * eta[k][j][i].y +
			  eta[k][j][i].z * eta[k][j][i].z);
	    } else 
	      ucor[k][j][i].y=0.;
	  }
	  if (nvert[k+1][j][i] > 0.1 && nvert[k+1][j][i] < ibmval && k < mz-2) {
	    if (fabs(ucor[k][j][i].z)>epsilon) {
	    libm_Flux += ucor[k][j][i].z;
	    if (flg==3)
	      libm_Flux_abs += fabs(ucor[k][j][i].z)/sqrt(zet[k][j][i].x * zet[k][j][i].x +
							  zet[k][j][i].y * zet[k][j][i].y +
							  zet[k][j][i].z * zet[k][j][i].z);
	    else
	      libm_Flux_abs += fabs(ucor[k][j][i].z);
	    libm_area += sqrt(zet[k][j][i].x * zet[k][j][i].x +
			  zet[k][j][i].y * zet[k][j][i].y +
			  zet[k][j][i].z * zet[k][j][i].z);
	    }else 
	      ucor[k][j][i].z=0.;
	  }
	}

	if (nvert[k][j][i] > 0.1 && nvert[k][j][i] < ibmval) {
	  if (nvert[k][j][i+1] < 0.1 && i < mx-2) {
	    if (fabs(ucor[k][j][i].x)>epsilon) {
	    libm_Flux -= ucor[k][j][i].x;
	    if (flg==3)
	    libm_Flux_abs += fabs(ucor[k][j][i].x)/sqrt(csi[k][j][i].x * csi[k][j][i].x +
							csi[k][j][i].y * csi[k][j][i].y +
							csi[k][j][i].z * csi[k][j][i].z);
	    else
	      libm_Flux_abs += fabs(ucor[k][j][i].x);
	    libm_area += sqrt(csi[k][j][i].x * csi[k][j][i].x +
			  csi[k][j][i].y * csi[k][j][i].y +
			  csi[k][j][i].z * csi[k][j][i].z);
			  
	    }else 
	      ucor[k][j][i].x=0.;
	  }
	  if (nvert[k][j+1][i] < 0.1 && j < my-2) {
	    if (fabs(ucor[k][j][i].y)>epsilon) {
	    libm_Flux -= ucor[k][j][i].y;
	    if (flg==3)
	      libm_Flux_abs += fabs(ucor[k][j][i].y)/ sqrt(eta[k][j][i].x * eta[k][j][i].x +
							   eta[k][j][i].y * eta[k][j][i].y +
							   eta[k][j][i].z * eta[k][j][i].z);
	    else
	      libm_Flux_abs += fabs(ucor[k][j][i].y);
	    libm_area += sqrt(eta[k][j][i].x * eta[k][j][i].x +
			  eta[k][j][i].y * eta[k][j][i].y +
			  eta[k][j][i].z * eta[k][j][i].z);
	    }else 
	      ucor[k][j][i].y=0.;
	  }
	  if (nvert[k+1][j][i] < 0.1 && k < mz-2) {
	    if (fabs(ucor[k][j][i].z)>epsilon) {
	    libm_Flux -= ucor[k][j][i].z;
	    if (flg==3)
	      libm_Flux_abs += fabs(ucor[k][j][i].z)/sqrt(zet[k][j][i].x * zet[k][j][i].x +
							  zet[k][j][i].y * zet[k][j][i].y +
							  zet[k][j][i].z * zet[k][j][i].z);
	    else
	      libm_Flux_abs += fabs(ucor[k][j][i].z);
	    libm_area += sqrt(zet[k][j][i].x * zet[k][j][i].x +
			  zet[k][j][i].y * zet[k][j][i].y +
			  zet[k][j][i].z * zet[k][j][i].z);
	    }else 
	      ucor[k][j][i].z=0.;
	  }
	}

      }
    }
  }
  GlobalSum_All(&libm_Flux, ibm_Flux, PETSC_COMM_WORLD);
  GlobalSum_All(&libm_Flux_abs, &ibm_Flux_abs, PETSC_COMM_WORLD);
  GlobalSum_All(&libm_area, ibm_Area, PETSC_COMM_WORLD);

  PetscReal correction;

  if (*ibm_Area > 1.e-15) {
    if (flg>1) 
      correction = (*ibm_Flux + user->FluxIntpSum)/ ibm_Flux_abs;
    else if (flg)
      correction = (*ibm_Flux + user->FluxIntpSum) / *ibm_Area;
    else
      correction = *ibm_Flux / *ibm_Area;
  }
  else {
    correction = 0;
  }

  if(immersed==2) correction = 0; //seokkoo

  PetscPrintf(PETSC_COMM_WORLD, "IBMFlux %le %le %le\n", *ibm_Flux, *ibm_Area, correction);

  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	if (nvert[k][j][i] < 0.1) {
	  if (nvert[k][j][i+1] > 0.1 && nvert[k][j][i+1] <ibmval && i < mx-2) {
	    if (fabs(ucor[k][j][i].x)>epsilon){
	    if (flg==3) 
	      ucor[k][j][i].x -=correction*fabs(ucor[k][j][i].x)/
		sqrt(csi[k][j][i].x * csi[k][j][i].x +
		     csi[k][j][i].y * csi[k][j][i].y +
		     csi[k][j][i].z * csi[k][j][i].z);
	    else if (flg==2) 
	      ucor[k][j][i].x -=correction*fabs(ucor[k][j][i].x);
	    else
	    ucor[k][j][i].x -= sqrt(csi[k][j][i].x * csi[k][j][i].x +
				    csi[k][j][i].y * csi[k][j][i].y +
				    csi[k][j][i].z * csi[k][j][i].z) *
				    correction;
	    }
			  
	  }
	  if (nvert[k][j+1][i] > 0.1 && nvert[k][j+1][i] < ibmval && j < my-2) {
	    if (fabs(ucor[k][j][i].y)>epsilon) {
	    if (flg==3) 
	      ucor[k][j][i].y -=correction*fabs(ucor[k][j][i].y)/
		sqrt(eta[k][j][i].x * eta[k][j][i].x + 
		     eta[k][j][i].y * eta[k][j][i].y +
		     eta[k][j][i].z * eta[k][j][i].z);
	    else if (flg==2) 
	      ucor[k][j][i].y -=correction*fabs(ucor[k][j][i].y);
	    else
	    ucor[k][j][i].y -= sqrt(eta[k][j][i].x * eta[k][j][i].x + 
				    eta[k][j][i].y * eta[k][j][i].y +
				    eta[k][j][i].z * eta[k][j][i].z) *
				    correction;
	    }
	  }
	  if (nvert[k+1][j][i] > 0.1 && nvert[k+1][j][i] < ibmval && k < mz-2) {
	    if (fabs(ucor[k][j][i].z)>epsilon) {
	    if (flg==3) 
	      ucor[k][j][i].z -= correction*fabs(ucor[k][j][i].z)/
		sqrt(zet[k][j][i].x * zet[k][j][i].x +
		     zet[k][j][i].y * zet[k][j][i].y +
		     zet[k][j][i].z * zet[k][j][i].z);
	    else if (flg==2) 
	      ucor[k][j][i].z -= correction*fabs(ucor[k][j][i].z);
	    else
	    ucor[k][j][i].z -= sqrt(zet[k][j][i].x * zet[k][j][i].x +
				    zet[k][j][i].y * zet[k][j][i].y +
				    zet[k][j][i].z * zet[k][j][i].z) *
				    correction;
	    }
	  }
	}

	if (nvert[k][j][i] > 0.1 && nvert[k][j][i] < 1.1) {
	  if (nvert[k][j][i+1] < 0.1 && i < mx-2) {
	    if (fabs(ucor[k][j][i].x)>epsilon) {
	    if (flg==3) 
	      ucor[k][j][i].x += correction*fabs(ucor[k][j][i].x)/
		sqrt(csi[k][j][i].x * csi[k][j][i].x +
		     csi[k][j][i].y * csi[k][j][i].y +
		     csi[k][j][i].z * csi[k][j][i].z);
	    else if (flg==2) 
	      ucor[k][j][i].x += correction*fabs(ucor[k][j][i].x);
	    else
	    ucor[k][j][i].x += sqrt(csi[k][j][i].x * csi[k][j][i].x +
				    csi[k][j][i].y * csi[k][j][i].y +
				    csi[k][j][i].z * csi[k][j][i].z) *
				    correction;
	    }
			  
	  }
	  if (nvert[k][j+1][i] < 0.1 && j < my-2) {
	    if (fabs(ucor[k][j][i].y)>epsilon) {
	    if (flg==3) 
	      ucor[k][j][i].y +=correction*fabs(ucor[k][j][i].y)/
		sqrt(eta[k][j][i].x * eta[k][j][i].x +
		     eta[k][j][i].y * eta[k][j][i].y +
		     eta[k][j][i].z * eta[k][j][i].z);
	    else if (flg==2) 
	      ucor[k][j][i].y +=correction*fabs(ucor[k][j][i].y);
	    else
	    ucor[k][j][i].y += sqrt(eta[k][j][i].x * eta[k][j][i].x +
				    eta[k][j][i].y * eta[k][j][i].y +
				    eta[k][j][i].z * eta[k][j][i].z) *
				    correction;
	    }
	  }
	  if (nvert[k+1][j][i] < 0.1 && k < mz-2) {
	    if (fabs(ucor[k][j][i].z)>epsilon) {
	    if (flg==3) 
	      ucor[k][j][i].z += correction*fabs(ucor[k][j][i].z)/
		sqrt(zet[k][j][i].x * zet[k][j][i].x +
		     zet[k][j][i].y * zet[k][j][i].y +
		     zet[k][j][i].z * zet[k][j][i].z);
	    else if (flg==2) 
	      ucor[k][j][i].z += correction*fabs(ucor[k][j][i].z);
	    else
	    ucor[k][j][i].z += sqrt(zet[k][j][i].x * zet[k][j][i].x +
				    zet[k][j][i].y * zet[k][j][i].y +
				    zet[k][j][i].z * zet[k][j][i].z) *
				    correction;
	    }
	  }
	}

      }
    }
  }
  


  libm_Flux = 0;
  libm_area = 0;
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	if (nvert[k][j][i] < 0.1) {
	  if (nvert[k][j][i+1] > 0.1 && nvert[k][j][i+1] < ibmval && i < mx-2) {
	    libm_Flux += ucor[k][j][i].x;
	    libm_area += sqrt(csi[k][j][i].x * csi[k][j][i].x +
			  csi[k][j][i].y * csi[k][j][i].y +
			  csi[k][j][i].z * csi[k][j][i].z);
			  
	  }
	  if (nvert[k][j+1][i] > 0.1 && nvert[k][j+1][i] < ibmval && j < my-2) {
	    libm_Flux += ucor[k][j][i].y;
	    libm_area += sqrt(eta[k][j][i].x * eta[k][j][i].x +
			  eta[k][j][i].y * eta[k][j][i].y +
			  eta[k][j][i].z * eta[k][j][i].z);
	  }
	  if (nvert[k+1][j][i] > 0.1 && nvert[k+1][j][i] < ibmval && k < mz-2) {
	    libm_Flux += ucor[k][j][i].z;
	    libm_area += sqrt(zet[k][j][i].x * zet[k][j][i].x +
			  zet[k][j][i].y * zet[k][j][i].y +
			  zet[k][j][i].z * zet[k][j][i].z);
	  }
	}

	if (nvert[k][j][i] > 0.1 && nvert[k][j][i] < ibmval) {
	  if (nvert[k][j][i+1] < 0.1 && i < mx-2) {
	    libm_Flux -= ucor[k][j][i].x;
	    libm_area += sqrt(csi[k][j][i].x * csi[k][j][i].x +
			  csi[k][j][i].y * csi[k][j][i].y +
			  csi[k][j][i].z * csi[k][j][i].z);
			  
	  }
	  if (nvert[k][j+1][i] < 0.1 && j < my-2) {
	    libm_Flux -= ucor[k][j][i].y;
	    libm_area += sqrt(eta[k][j][i].x * eta[k][j][i].x +
			  eta[k][j][i].y * eta[k][j][i].y +
			  eta[k][j][i].z * eta[k][j][i].z);
	  }
	  if (nvert[k+1][j][i] < 0.1 && k < mz-2) {
	    libm_Flux -= ucor[k][j][i].z;
	    libm_area += sqrt(zet[k][j][i].x * zet[k][j][i].x +
			  zet[k][j][i].y * zet[k][j][i].y +
			  zet[k][j][i].z * zet[k][j][i].z);
	  }
	}

      }
    }
  }
  GlobalSum_All(&libm_Flux, ibm_Flux, PETSC_COMM_WORLD);
  GlobalSum_All(&libm_area, ibm_Area, PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "IBMFlux22 %le %le\n", *ibm_Flux, *ibm_Area);

  DMDAVecRestoreArray(da, user->lNvert, &nvert);
  DMDAVecRestoreArray(fda, user->lCsi, &csi);
  DMDAVecRestoreArray(fda, user->lEta, &eta);
  DMDAVecRestoreArray(fda, user->lZet, &zet);
  DMDAVecRestoreArray(fda, user->Ucont, &ucor);

  DMGlobalToLocalBegin(fda, user->Ucont, INSERT_VALUES, user->lUcont);
  DMGlobalToLocalEnd(fda, user->Ucont, INSERT_VALUES, user->lUcont);
  return 0;
}

PetscErrorCode MaxPosition(UserCtx *user, int pos)
{
  
  DMDALocalInfo	info = user->info;
  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;

  int i, j, k;
  for (k=zs; k<ze; k++) {
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
	if (lidx(i,j,k,user) == pos) {
	  PetscPrintf(PETSC_COMM_SELF, "Position %i %i %i\n", i, j, k);
	}
      }
    }
  }

  return 0;
}
#define Epsilon_Eq 1.e-6
//#define PartFloat_Eq(a, b) (fabs(a-b)>Epislon_Eq) ? 0:1;
#define Float_Eq(a, b) (a==b) ? PETSC_TRUE : (((a-b)<Epsilon_Eq) && (a-b) > -Epsilon_Eq)

PetscErrorCode MyNFaceFine(UserCtx *user)
{
  DM		da = user->da, fda = user->fda;
  DMDALocalInfo	info = user->info;

  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;

  int	mx = info.mx, my = info.my, mz = info.mz;

  PetscReal ***nvert;
  Cmpnts ***nface;

  int      i, j, k;
  int	lxs, lxe, lys, lye, lzs, lze;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  DMDAVecGetArray(da, user->lNvert, &nvert);
  DMDAVecGetArray(fda, user->lNFace, &nface);

  VecSet(user->lNFace, 0.);
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	if (Float_Eq(nvert[k][j][i], 0)) {
	  if (Float_Eq(nvert[k][j][i+1], 1)) {
	    nface[k][j][i].x = 1.;
	  }
	  if (Float_Eq(nvert[k][j+1][i], 1)) {
	    nface[k][j][i].y = 1.;
	  }
	  if (Float_Eq(nvert[k+1][j][i], 1)) {
	    nface[k][j][i].z = 1.;
	  }
	}
	else {
	  nface[k][j][i].x = 1.;
	  nface[k][j][i].y = 1.;
	  nface[k][j][i].z = 1.;
	}
      }
    }
  }
  DMDAVecRestoreArray(da, user->lNvert, &nvert);
  DMDAVecRestoreArray(fda, user->lNFace, &nface);

  DMDALocalToLocalBegin(fda, user->lNFace, INSERT_VALUES, user->lNFace);
  DMDALocalToLocalEnd(fda, user->lNFace, INSERT_VALUES, user->lNFace);
  return 0;
}

PetscErrorCode MyNFaceRestrict(UserCtx *user)
{
  DMDALocalInfo	info = user->info;
  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;
  
  int i,j,k;
  int ih, jh, kh, ia, ja, ka;
  int	lxs, lxe, lys, lye, lzs, lze;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  Cmpnts ***nface_h, ***nface;
  PetscReal ***nvert;

  UserCtx *user_h = user->user_f;
  DM	fda_h = user_h->fda, fda = user->fda, da = user->da;
  DMDAVecGetArray(fda_h, user_h->lNFace, &nface_h);
  DMDAVecGetArray(fda,   user->lNFace, &nface);

  VecSet(user->lNFace, 0.);

  if (*(user->isc)) ia = 0;
  else ia = 1;

  if (*(user->jsc)) ja = 0;
  else ja = 1;

  if (*(user->ksc)) ka = 0;
  else ka = 1;

  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	GridRestriction(i, j, k, &ih, &jh, &kh, user);
	if (Float_Eq(nface_h[kh   ][jh   ][ih].x, 1) &&
	    Float_Eq(nface_h[kh   ][jh-ja][ih].x, 1) &&
	    Float_Eq(nface_h[kh-ka][jh   ][ih].x, 1) &&
	    Float_Eq(nface_h[kh-ka][jh-ja][ih].x, 1)) {
	  nface[k][j][i].x = 1.;
	}

	if (Float_Eq(nface_h[kh   ][jh][ih   ].y, 1) &&
	    Float_Eq(nface_h[kh   ][jh][ih-ia].y, 1) &&
	    Float_Eq(nface_h[kh-ka][jh][ih   ].y, 1) &&
	    Float_Eq(nface_h[kh-ka][jh][ih-ia].y, 1)) {
	  nface[k][j][i].y = 1.;
	}

	if (Float_Eq(nface_h[kh][jh   ][ih   ].z, 1) &&
	    Float_Eq(nface_h[kh][jh-ja][ih   ].z, 1) &&
	    Float_Eq(nface_h[kh][jh   ][ih-ia].z, 1) &&
	    Float_Eq(nface_h[kh][jh-ja][ih-ia].z, 1)) {
	  nface[k][j][i].z = 1.;
	}
      }
    }
  }

  DMDAVecGetArray(da, user->lNvert, &nvert);

  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	if (Float_Eq(nvert[k][j][i], 0)) {
	  if (Float_Eq(nvert[k][j][i+1], 1)) {
	    nface[k][j][i].x = 1.;
	  }
	  if (Float_Eq(nvert[k][j+1][i], 1)) {
	    nface[k][j][i].y = 1.;
	  }
	  if (Float_Eq(nvert[k+1][j][i], 1)) {
	    nface[k][j][i].z = 1.;
	  }
	}
	else {
	  nface[k][j][i].x = 1.;
	  nface[k][j][i].y = 1.;
	  nface[k][j][i].z = 1.;
	}
      }
    }
  }


  
  DMDAVecRestoreArray(da, user->lNvert, &nvert);

  DMDAVecRestoreArray(fda, user->lNFace, &nface);
  DMDAVecRestoreArray(fda_h, user_h->lNFace, &nface_h);

  DMDALocalToLocalBegin(fda, user->lNFace, INSERT_VALUES, user->lNFace);
  DMDALocalToLocalEnd(fda, user->lNFace, INSERT_VALUES, user->lNFace);

  //  VecSet(user->lNFace, 0.);
  return 0;
}

PetscErrorCode MyNFaceInit(UserMG *usermg)
{
  int l;

  int bi;

  MGCtx *mgctx = usermg->mgctx;

  UserCtx *user;
  for (l=usermg->mglevels-1; l>=0; l--) {
    user = mgctx[l].user;
    for (bi=0; bi<block_number; bi++) {
      VecDuplicate(user[bi].lCsi, &user[bi].lNFace);
      if (l == usermg->mglevels-1) {
	MyNFaceFine(&user[bi]);
      }
      else {
	MyNFaceRestrict(&user[bi]);
      }
    }
  }
  return 0;
}

PetscErrorCode MyNFaceFinalize(UserMG *usermg)
{
  int l;

  MGCtx *mgctx = usermg->mgctx;

  UserCtx *user;
  for (l=usermg->mglevels-1; l>=0; l--) {
    user = mgctx[l].user;
    VecDestroy(&user->lNFace);
  }
  return 0;
}

void n_phase(int *phase, int *previous_ti)
{
	int n=0, t;
	
	for(int i=ti_lastsave+1; i<=ti_lastsave+ti; i++) {
		if(i%phase_averaging==0) {
			n++;
			t = i - ti_lastsave;
		}
	}
	*phase = n;
	*previous_ti = t;
}

PetscErrorCode Do_averaging(UserCtx *user)
{
	DMDALocalInfo info = user->info;
	PetscInt xs = info.xs, xe = info.xs + info.xm;
	PetscInt ys = info.ys, ye = info.ys + info.ym;
	PetscInt zs = info.zs, ze = info.zs + info.zm;
	PetscInt mx = info.mx, my = info.my, mz = info.mz;

	PetscInt i, j, k;
	PetscInt lxs, lxe, lys, lye, lzs, lze;

	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
  
	Cmpnts ***ucat, ***csi, ***eta, ***zet;
	Cmpnts ***u2sum, ***u_cross_sum, ***vort_sum, ***vort2_sum, ***uuusum;
	Cmpnts ***u2sum_phase, ***u_cross_sum_phase;
	Cmpnts ***conc_cross_sum;
	PetscReal ***p2sum, ***p, ***k_sum, ***lnu_t, ***du2sum;
	PetscReal ***level, ***level2sum;
	PetscReal ***conc, ***conc_sum;
	PetscReal ***aj, ***nvert, ***udpsum;//, ***nut_sum, ***taussum;
	PetscReal ***p2sum_phase;
	Cmpnts2 ***ko;
	
	VecAXPY(user->Ucat_sum, 1., user->Ucat);
	VecAXPY(user->P_sum, 1., user->P);
	
	if(levelset) VecAXPY(user->Levelset_sum, 1., user->Levelset);
		
	if(phase_averaging && (ti+ti_lastsave)%phase_averaging==0) {
		VecAXPY(user->Ucat_sum_phase, 1., user->Ucat);
		VecAXPY(user->P_sum_phase, 1., user->P);
	}
	
	double max_norm;
	VecMax(user->Ucat_sum, &i, &max_norm);
	PetscPrintf(PETSC_COMM_WORLD, "\n*** Max Ucat_avg = %e \n", max_norm / (double)ti);
	
	if(levelset) {
		DMDAVecGetArray(user->da, user->Levelset, &level);
		DMDAVecGetArray(user->da, user->Levelset_square_sum, &level2sum);
	}
	
        if (conv_diff)
        {
          DMDAVecGetArray(user->da, user->Conc, &conc);
          DMDAVecGetArray(user->da, user->Conc_sum, &conc_sum);
          DMDAVecGetArray(user->fda, user->Conc_cross_sum, &conc_cross_sum);
	}
	
	DMDAVecGetArray(user->fda, user->lUcat, &ucat);
	DMDAVecGetArray(user->da, user->lAj, &aj);
	DMDAVecGetArray(user->da, user->lNvert, &nvert);
	DMDAVecGetArray(user->fda,user->lCsi, &csi);
	DMDAVecGetArray(user->fda,user->lEta, &eta);
	DMDAVecGetArray(user->fda,user->lZet, &zet);
	DMDAVecGetArray(user->da, user->lP, &p);
	
	DMDAVecGetArray(user->fda, user->Ucat_square_sum, &u2sum);
	DMDAVecGetArray(user->fda, user->Ucat_cross_sum, &u_cross_sum);
	/*if(averaging>=2)*/ DMDAVecGetArray(user->da, user->P_square_sum, &p2sum);
	//DMDAVecGetArray(user->fda, user->P_cross_sum, &p_cross_sum);
	
	if(phase_averaging) {
		DMDAVecGetArray(user->fda, user->Ucat_square_sum_phase, &u2sum_phase);
		DMDAVecGetArray(user->fda, user->Ucat_cross_sum_phase, &u_cross_sum_phase);
		DMDAVecGetArray(user->da, user->P_square_sum_phase, &p2sum_phase);
	}
	

	PetscReal ***tmprt, ***tmprt_sum, ***tt_sum, ***tp_sum, ***prt_sum, ***pr_t;
	Cmpnts ***tu_sum;
	// xyang
        if(user->temp) {
                DMDAVecGetArray(user->da, user->gTmprt, &tmprt);
                DMDAVecGetArray(user->da, user->T_sum, &tmprt_sum);
                DMDAVecGetArray(user->da, user->TT_sum, &tt_sum);
                DMDAVecGetArray(user->da, user->TP_sum, &tp_sum);
                DMDAVecGetArray(user->fda, user->TU_sum, &tu_sum);
		if (les_prt) {
			DMDAVecGetArray(user->da, user->Prt_sum, &prt_sum);
			DMDAVecGetArray(user->da, user->lPr_t, &pr_t);
		}
        }
	//


	
	if(rans) {
		DMDAVecGetArray(user->fda2, user->K_Omega, &ko);
		DMDAVecGetArray(user->da, user->K_sum, &k_sum);
	}
	
	if(les) {
		DMDAVecGetArray(user->da, user->lNu_t, &lnu_t);
		//DMDAVecGetArray(user->da, user->Nut_sum, &nut_sum);
	}
	
	if(averaging>=3) {
		if(les) {
			//DMDAVecGetArray(user->da, user->tauS_sum, &taussum);
		}
		
		DMDAVecGetArray(user->da, user->Udp_sum, &udpsum);
		DMDAVecGetArray(user->da, user->dU2_sum, &du2sum);
		DMDAVecGetArray(user->fda, user->UUU_sum, &uuusum);

		DMDAVecGetArray(user->fda, user->Vort_sum, &vort_sum);
		DMDAVecGetArray(user->fda, user->Vort_square_sum, &vort2_sum);
	}
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		double U = ucat[k][j][i].x, V = ucat[k][j][i].y, W = ucat[k][j][i].z;
		u2sum[k][j][i].x += ucat[k][j][i].x * ucat[k][j][i].x;
		u2sum[k][j][i].y += ucat[k][j][i].y * ucat[k][j][i].y;
		u2sum[k][j][i].z += ucat[k][j][i].z * ucat[k][j][i].z;
		/*if(averaging>=2) */p2sum[k][j][i] += p[k][j][i] * p[k][j][i];
		u_cross_sum[k][j][i].x += ucat[k][j][i].x * ucat[k][j][i].y;	// uv
		u_cross_sum[k][j][i].y += ucat[k][j][i].y * ucat[k][j][i].z;	// vw
		u_cross_sum[k][j][i].z += ucat[k][j][i].z * ucat[k][j][i].x;	// wu
		
		if(phase_averaging && (ti+ti_lastsave)%phase_averaging==0) {
			u2sum_phase[k][j][i].x += ucat[k][j][i].x * ucat[k][j][i].x;
			u2sum_phase[k][j][i].y += ucat[k][j][i].y * ucat[k][j][i].y;
			u2sum_phase[k][j][i].z += ucat[k][j][i].z * ucat[k][j][i].z;
			p2sum_phase[k][j][i] += p[k][j][i] * p[k][j][i];
			u_cross_sum_phase[k][j][i].x += ucat[k][j][i].x * ucat[k][j][i].y;	// uv
			u_cross_sum_phase[k][j][i].y += ucat[k][j][i].y * ucat[k][j][i].z;	// vw
			u_cross_sum_phase[k][j][i].z += ucat[k][j][i].z * ucat[k][j][i].x;	// wu
		}


		if(user->temp) {
			tmprt_sum[k][j][i] += tmprt[k][j][i];
			tt_sum[k][j][i] += tmprt[k][j][i]*tmprt[k][j][i];
			tp_sum[k][j][i] += tmprt[k][j][i]*p[k][j][i];
			tu_sum[k][j][i].x += tmprt[k][j][i]*ucat[k][j][i].x;
			tu_sum[k][j][i].y += tmprt[k][j][i]*ucat[k][j][i].y;
			tu_sum[k][j][i].z += tmprt[k][j][i]*ucat[k][j][i].z;

			if (les_prt) prt_sum[k][j][i] += pr_t[k][j][i];
		}

                if (conv_diff)
                {
                  conc_sum[k][j][i] += conc[k][j][i];
                  conc_cross_sum[k][j][i].x += ucat[k][j][i].x * conc[k][j][i];	// cu
                  conc_cross_sum[k][j][i].y += ucat[k][j][i].y * conc[k][j][i];	// cv
                  conc_cross_sum[k][j][i].z += ucat[k][j][i].z * conc[k][j][i];	// cw	
                }

		if(rans) {
			k_sum[k][j][i] += ko[k][j][i].x;
		}

		if(les) {
			//nut_sum[k][j][i] += lnu_t[k][j][i];
		}
		
		if(levelset) {
			level2sum[k][j][i] += level[k][j][i] * level[k][j][i];
		}
		//if(averaging>=2) 
		{
			
			/*
			p_cross_sum[k][j][i].x += p[k][j][i] * ucat[k][j][i].x;
			p_cross_sum[k][j][i].y += p[k][j][i] * ucat[k][j][i].y;
			p_cross_sum[k][j][i].z += p[k][j][i] * ucat[k][j][i].z;
			*/
		}
		
		if(averaging>=3) {
			double dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz;
			double du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz;
			double dpdc, dpde, dpdz;
			double dp_dx, dp_dy, dp_dz;

			double csi0 = csi[k][j][i].x, csi1 = csi[k][j][i].y, csi2 = csi[k][j][i].z;
			double eta0= eta[k][j][i].x, eta1 = eta[k][j][i].y, eta2 = eta[k][j][i].z;
			double zet0 = zet[k][j][i].x, zet1 = zet[k][j][i].y, zet2 = zet[k][j][i].z;
			double ajc = aj[k][j][i];
				
			Compute_du_center ( i, j, k, mx, my, mz, ucat, nvert, &dudc, &dvdc, &dwdc, &dude, &dvde, &dwde, &dudz, &dvdz, &dwdz);
			Compute_dscalar_center ( i, j, k, mx, my, mz, p, nvert, &dpdc, &dpde, &dpdz);
			Compute_du_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz, &du_dx, &dv_dx, &dw_dx, &du_dy, &dv_dy, &dw_dy, &du_dz, &dv_dz, &dw_dz );
			Compute_dscalar_dxyz ( csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc, dpdc, dpde, dpdz, &dp_dx, &dp_dy, &dp_dz);

			double vort_x = dw_dy - dv_dz, vort_y = du_dz - dw_dx, vort_z = dv_dx - du_dy;
			
			vort_sum[k][j][i].x += vort_x;
			vort_sum[k][j][i].y += vort_y;
			vort_sum[k][j][i].z += vort_z;
			
			vort2_sum[k][j][i].x += vort_x*vort_x;
			vort2_sum[k][j][i].y += vort_y*vort_y;
			vort2_sum[k][j][i].z += vort_z*vort_z;

			udpsum[k][j][i] += U * dp_dx + V * dp_dy + W * dp_dz;

			du2sum[k][j][i]/*.x*/ += pow(du_dx, 2.) + pow(du_dy, 2.) + pow(du_dz, 2.);
			du2sum[k][j][i]/*.y*/ += pow(dv_dx, 2.) + pow(dv_dy, 2.) + pow(dv_dz, 2.);
			du2sum[k][j][i]/*.z*/ += pow(dw_dx, 2.) + pow(dw_dy, 2.) + pow(dw_dz, 2.);
			
			uuusum[k][j][i].x += (U*U + V*V + W*W) * U;
			uuusum[k][j][i].y += (U*U + V*V + W*W) * V;
			uuusum[k][j][i].z += (U*U + V*V + W*W) * W;
			
			if(les) {
				double Sxx = 0.5*( du_dx + du_dx ), Sxy = 0.5*(du_dy + dv_dx), Sxz = 0.5*(du_dz + dw_dx);
				double Syx = Sxy, Syy = 0.5*(dv_dy + dv_dy),	Syz = 0.5*(dv_dz + dw_dy);
				double Szx = Sxz, Szy=Syz, Szz = 0.5*(dw_dz + dw_dz);
				double SS = Sxx*Sxx + Sxy*Sxy + Sxz*Sxz + Syx*Syx + Syy*Syy + Syz*Syz + Szx*Szx + Szy*Szy + Szz*Szz;
				//taussum[k][j][i] += 2. * lnu_t[k][j][i] * SS;
			}
		}
		
		
	}
	
	if(levelset) {
		DMDAVecRestoreArray(user->da, user->Levelset, &level);
		DMDAVecRestoreArray(user->da, user->Levelset_square_sum, &level2sum);
	}
	DMDAVecRestoreArray(user->fda, user->lUcat, &ucat);
	DMDAVecRestoreArray(user->da, user->lAj, &aj);
	DMDAVecRestoreArray(user->da, user->lNvert, &nvert);
	DMDAVecRestoreArray(user->fda,user->lCsi, &csi);
	DMDAVecRestoreArray(user->fda,user->lEta, &eta);
	DMDAVecRestoreArray(user->fda,user->lZet, &zet);
	DMDAVecRestoreArray(user->da, user->lP, &p);
	
	DMDAVecRestoreArray(user->fda, user->Ucat_square_sum, &u2sum);
	DMDAVecRestoreArray(user->fda, user->Ucat_cross_sum, &u_cross_sum);
	/*if(averaging>=2) */DMDAVecRestoreArray(user->da, user->P_square_sum, &p2sum);
	
	if(phase_averaging) {
		DMDAVecRestoreArray(user->fda, user->Ucat_square_sum_phase, &u2sum_phase);
		DMDAVecRestoreArray(user->fda, user->Ucat_cross_sum_phase, &u_cross_sum_phase);
		DMDAVecRestoreArray(user->da, user->P_square_sum_phase, &p2sum_phase);
	}


        // xyang
        if(user->temp) {
                DMDAVecRestoreArray(user->da, user->gTmprt, &tmprt);
                DMDAVecRestoreArray(user->da, user->T_sum, &tmprt_sum);
                DMDAVecRestoreArray(user->da, user->TT_sum, &tt_sum);
                DMDAVecRestoreArray(user->da, user->TP_sum, &tp_sum);
                DMDAVecRestoreArray(user->fda, user->TU_sum, &tu_sum);
                if (les_prt) {
			DMDAVecRestoreArray(user->da, user->lPr_t, &pr_t);
			DMDAVecRestoreArray(user->da, user->Prt_sum, &prt_sum);
		}

        }
        //
	if (conv_diff)
        {
          DMDAVecRestoreArray(user->da, user->Conc, &conc);
          DMDAVecRestoreArray(user->da, user->Conc_sum, &conc_sum);
          DMDAVecRestoreArray(user->fda, user->Conc_cross_sum, &conc_cross_sum);
	}
	
	if(rans) {
		DMDAVecRestoreArray(user->fda2, user->K_Omega, &ko);
		DMDAVecRestoreArray(user->da, user->K_sum, &k_sum);
        }
	if(les) {
		DMDAVecRestoreArray(user->da, user->lNu_t, &lnu_t);
		//DMDAVecRestoreArray(user->da, user->Nut_sum, &nut_sum);
        }

	//if(averaging>=2) 
	{
		
		//DMDAVecRestoreArray(user->fda, user->P_cross_sum, &p_cross_sum);
	}
	
	if(averaging>=3) {
		if(les) {
			//DMDAVecRestoreArray(user->da, user->tauS_sum, &taussum);
		}
		DMDAVecRestoreArray(user->da, user->Udp_sum, &udpsum);
		DMDAVecRestoreArray(user->da, user->dU2_sum, &du2sum);
		DMDAVecRestoreArray(user->fda, user->UUU_sum, &uuusum);
		DMDAVecRestoreArray(user->fda, user->Vort_sum, &vort_sum);
		DMDAVecRestoreArray(user->fda, user->Vort_square_sum, &vort2_sum);
	}
	
	
	int rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	
	if (ti!=0 && ti == (ti/tiout) * tiout && periodic && inletprofile==13 ) {
		Cmpnts ***u_sum, ***uu_sum, ***cent;
		double N=ti;
		double buffer[20][1000];	// [var][points]
		
		for(i=0; i<20; i++)
		for(j=0; j<1000; j++) buffer[i][j]=0;
		
		DMDAVecGetArray(user->fda, user->lCent, &cent);
		DMDAVecGetArray(user->fda, user->Ucat_sum, &u_sum);
		DMDAVecGetArray(user->fda, user->Ucat_square_sum, &uu_sum);
		DMDAVecGetArray(user->fda, user->Ucat_cross_sum, &u_cross_sum);
		
		std::vector<int> count (my), total_count(my);
		std::vector<double> Y_sum(my), U_sum(my), V_sum(my), W_sum(my), UU_sum(my), VV_sum(my), WW_sum(my), UV_sum(my), VW_sum(my), WU_sum(my);
		std::vector<double> Y_sum_tmp(my), U_sum_tmp(my), V_sum_tmp(my), W_sum_tmp(my), UU_sum_tmp(my), VV_sum_tmp(my), WW_sum_tmp(my), UV_sum_tmp(my), VW_sum_tmp(my), WU_sum_tmp(my);
	
		for (j=0; j<my; j++) Y_sum_tmp[j] = 0, U_sum_tmp[j]=0, V_sum_tmp[j]=0, W_sum_tmp[j]=0, UU_sum_tmp[j]=0, VV_sum_tmp[j]=0, WW_sum_tmp[j]=0, UV_sum_tmp[j]=0, VW_sum_tmp[j]=0, WU_sum_tmp[j]=0;
	
		std::fill( count.begin(), count.end(), 0 );
		
			
		for (j=ys; j<ye; j++) {
			
			for (k=lzs; k<lze; k++)
			for (i=lxs; i<lxe; i++) {
				{
					count[j] ++;
					
					double U = u_sum[k][j][i].x / N, V = u_sum[k][j][i].y / N, W = u_sum[k][j][i].z / N;
					double uu = uu_sum[k][j][i].x/N - U*U;
					double vv = uu_sum[k][j][i].y/N - V*V;
					double ww = uu_sum[k][j][i].z/N - W*W;
					double uv = u_cross_sum[k][j][i].x/N - U*V;
					double vw = u_cross_sum[k][j][i].y/N - V*W;
					double wu = u_cross_sum[k][j][i].z/N - W*U;
					
					Y_sum_tmp[j] += cent[k][j][i].y;
					
					U_sum_tmp[j] += U;
					V_sum_tmp[j] += V;
					W_sum_tmp[j] += W;
						
					UU_sum_tmp[j] += uu;
					VV_sum_tmp[j] += vv;
					WW_sum_tmp[j] += ww;
						
					UV_sum_tmp[j] += uv;
					VW_sum_tmp[j] += vw;
					WU_sum_tmp[j] += wu;
				}
			}
		}
		
		DMDAVecRestoreArray(user->fda, user->lCent, &cent);
		DMDAVecRestoreArray(user->fda, user->Ucat_sum, &u_sum);
		DMDAVecRestoreArray(user->fda, user->Ucat_square_sum, &uu_sum);
		DMDAVecRestoreArray(user->fda, user->Ucat_cross_sum, &u_cross_sum);
		
		MPI_Reduce( &count[0], &total_count[0], my, MPI_INT, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce( &Y_sum_tmp[0], &Y_sum[0], my, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce( &U_sum_tmp[0], &U_sum[0], my, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce( &V_sum_tmp[0], &V_sum[0], my, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce( &W_sum_tmp[0], &W_sum[0], my, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce( &UU_sum_tmp[0], &UU_sum[0], my, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce( &VV_sum_tmp[0], &VV_sum[0], my, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce( &WW_sum_tmp[0], &WW_sum[0], my, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce( &UV_sum_tmp[0], &UV_sum[0], my, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce( &VW_sum_tmp[0], &VW_sum[0], my, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		MPI_Reduce( &WU_sum_tmp[0], &WU_sum[0], my, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		
		if(!rank) {
			char filen[80];
			
			sprintf(filen, "%s/Channel_Profile_%06d.dat", path, (int)ti);
			FILE *fp = fopen(filen, "w");
			if(fp) {
				fprintf(fp, "VARIABLES = \"y\" \"y+\" \"w\" \"w+\" \"uu+\" \"vv+\" \"ww+\" \"urms+\" \"vrms+\" \"wrms+\" \"vw+\" \"log\"\n");
				fprintf(fp, "ZONE T=\"Channel\"\n\n");
				for (j=1; j<my-1; j++) {
					
					double Y = Y_sum[j] / total_count[j];
					double Yp = user->Re_tau_avg*Y;
					double ustar = user->ustar_avg;
					double ustar2 = ustar * ustar;
					double W = W_sum[j] / total_count[j];
					double uu = UU_sum[j] / total_count[j];
					double vv = VV_sum[j] / total_count[j];
					double ww = WW_sum[j] / total_count[j];
					double vw = VW_sum[j] / total_count[j];
					
					if(Y<0) continue;
					
					fprintf(fp, "%e ", Y);	// y
					fprintf(fp, "%e ", Yp);	// y+
					fprintf(fp, "%e ", W);	// w
					fprintf(fp, "%e ", W/ustar);	// w+
					fprintf(fp, "%e ", uu/ustar2);	// uu+
					fprintf(fp, "%e ", vv/ustar2);	// vv+
					fprintf(fp, "%e ", ww/ustar2);	// ww+
					fprintf(fp, "%e ", sqrt(uu/ustar2));	// urms
					fprintf(fp, "%e ", sqrt(vv/ustar2));	// vrms
					fprintf(fp, "%e ", sqrt(ww/ustar2));	// wrms
					fprintf(fp, "%e ", vw/ustar2);	// vw+
					fprintf(fp, "%e ", 2.5*log(Yp)+5.5);
					fprintf(fp, "\n");
				}
				fclose(fp);
			}
			
		}
	}
	
	PetscBarrier(PETSC_NULL);
	
	return 0;
}

PetscErrorCode KE_Output(UserCtx *user)
{
	DMDALocalInfo info = user->info;
	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;

	int i, j, k;
	int	lxs, lys, lzs, lxe, lye, lze;
  
	Cmpnts	***ucat;
	PetscReal ***aj;
	
	lxs = xs; lxe = xe;
	lys = ys; lye = ye;
	lzs = zs; lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
  
	DMDAVecGetArray(user->fda, user->lUcat, &ucat);
	DMDAVecGetArray(user->da, user->lAj, &aj);
		
	double local_sum=0, sum=0;
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		local_sum += 0.5 * ucat[k][j][i].x * ucat[k][j][i].x / aj[k][j][i];
		local_sum += 0.5 * ucat[k][j][i].y * ucat[k][j][i].y / aj[k][j][i];
		local_sum += 0.5 * ucat[k][j][i].z * ucat[k][j][i].z / aj[k][j][i];
	}
	GlobalSum_All(&local_sum, &sum, PETSC_COMM_WORLD);
	
	DMDAVecRestoreArray(user->fda, user->lUcat, &ucat);
	DMDAVecRestoreArray(user->da, user->lAj, &aj);
	
	int rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	if (!rank) {	
		char filen[80];
		sprintf(filen, "%s/Kinetic_Energy.dat", path);
		FILE *f = fopen(filen, "a");
		PetscFPrintf(PETSC_COMM_WORLD, f, "%d\t%.7e\n", ti, sum);
		fclose(f);
	}
	
	return 0;
}


int setup_lidx3(UserCtx *user)	// with component, 1,2,3, for momentum
{
	DMDALocalInfo	info = user->info;
	DM		da = user->da, fda = user->fda;
	int	gxs, gxe, gys, gye, gzs, gze;
	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
	int i, j, k;
	PetscReal ***nvert, ***lidm;
	Cmpnts ***gidm;

	Vec	Lid;
	VecDuplicate(user->lUcont, &user->Gidm);
	VecDuplicate(user->lNvert, &Lid);
	
	DMDAVecGetArray(fda, user->Gidm, &gidm);
	DMDAVecGetArray(da, Lid, &lidm);
	DMDAVecGetArray(da, user->lNvert, &nvert);

	gxs = info.gxs; gxe = gxs + info.gxm;
	gys = info.gys; gye = gys + info.gym;
	gzs = info.gzs; gze = gzs + info.gzm;
		
	//int nblank_node[4096], nblank_node_tmp[4096];	// # of blank nodes for processors
	int ndof_node[4096], ndof_node_tmp[4096];	// # of pressure dof for processors
	int ndof_node_accu;
	
	int r, myrank, size;
	
		MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
		MPI_Comm_size(PETSC_COMM_WORLD,&size);
		
		for(r=0; r<size; r++) {
			ndof_node_tmp[r] = 0;
			//nblank_node_tmp[r] = 0;
		}
	
		for(k=zs; k<ze; k++)
		for(j=ys; j<ye; j++)
		for(i=xs; i<xe; i++) {
			if(! (i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1) &&
					(nvert[k][j][i]+nvert[k][j][i+1]>1.1 || nvert[k][j][i]+nvert[k][j+1][i]>1.1 || nvert[k][j][i]+nvert[k+1][j][i]>1.1) ) {
				//nblank_node_tmp[myrank]+=3;
				lidm[k][j][i]=0;
			}
			else {
				lidm[k][j][i] = (PetscReal)ndof_node_tmp[myrank];
				ndof_node_tmp[myrank] += 3;	// vector size
			}
		}
	
		//MPI_Allreduce( &nblank_node_tmp, &nblank_node, size, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);
		MPI_Allreduce( &ndof_node_tmp, &ndof_node, size, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);
		
		ndof_node_accu = 0;
		for(r=0; r<myrank; r++) ndof_node_accu += ndof_node[r];
		
		PetscInt n;
		VecGetSize(user->Ucont,&n);
		if(myrank==size-1) {
			printf("\n\n********* momentum: %d %d ***********\n\n", ndof_node_accu + ndof_node[myrank], (int)n);
		}
		
		//MPI_Bcast(&user->reduced_p_size, 1, MPI_INT, size-1, PETSC_COMM_WORLD);
		//PetscPrintf(PETSC_COMM_WORLD, "%d***********************\n", reduced_p_size);
		
		
		PetscBarrier(PETSC_NULL);
		
		for(k=zs; k<ze; k++)
		for(j=ys; j<ye; j++)
		for(i=xs; i<xe; i++) {
			gidm[k][j][i].x = lidm[k][j][i] + ndof_node_accu;	// gidm is double, be careful
			gidm[k][j][i].y = gidm[k][j][i].x + 1;
			gidm[k][j][i].z = gidm[k][j][i].x + 2;
		}
		
		
	VecCreateMPI(PETSC_COMM_WORLD, ndof_node[myrank], PETSC_DETERMINE, &user->Ucont2);
	
		
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(fda, user->Gidm, &gidm);
	DMDAVecRestoreArray(da, Lid, &lidm);
	
	
	VecDestroy(&Lid);
	
	DMDALocalToLocalBegin(da, user->Gidm, INSERT_VALUES, user->Gidm);
	DMDALocalToLocalEnd(da, user->Gidm, INSERT_VALUES, user->Gidm);
	return 0;
}


void Convert_Ucont2_Ucont(UserCtx *user)
{
	DMDALocalInfo	info = user->info;
	DM		da = user->da, fda = user->fda;
	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
	int i, j, k;
	PetscReal ***nvert, *ucont2;
	Cmpnts ***ucont;
	
	DMDAVecGetArray(fda, user->Ucont, &ucont);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	VecGetArray(user->Ucont2, &ucont2);
	
	int pos=0;
	for(k=zs; k<ze; k++)
	for(j=ys; j<ye; j++)
	for(i=xs; i<xe; i++) {
		if(! (i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1) &&
					(nvert[k][j][i]+nvert[k][j][i+1]>1.1 || nvert[k][j][i]+nvert[k][j+1][i]>1.1 || nvert[k][j][i]+nvert[k+1][j][i]>1.1) ) {
			// do nothing
		}
		else {
			ucont[k][j][i].x = ucont2[pos++];
			ucont[k][j][i].y = ucont2[pos++];
			ucont[k][j][i].z = ucont2[pos++];
		}
	}
	
	DMDAVecRestoreArray(fda, user->Ucont, &ucont);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	VecRestoreArray(user->Ucont2, &ucont2);
	
	DMGlobalToLocalBegin(fda, user->Ucont, INSERT_VALUES, user->lUcont);
	DMGlobalToLocalEnd(fda, user->Ucont, INSERT_VALUES, user->lUcont);
}

void Convert_Ucont_Ucont2(UserCtx *user)
{
	DMDALocalInfo	info = user->info;
	DM		da = user->da, fda = user->fda;
	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
	int i, j, k;
	PetscReal ***nvert, *ucont2;
	Cmpnts ***ucont;
	
	DMDAVecGetArray(fda, user->Ucont, &ucont);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	VecGetArray(user->Ucont2, &ucont2);
	
	int pos=0;
	for(k=zs; k<ze; k++)
	for(j=ys; j<ye; j++)
	for(i=xs; i<xe; i++) {
		if(! (i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1) &&
					(nvert[k][j][i]+nvert[k][j][i+1]>1.1 || nvert[k][j][i]+nvert[k][j+1][i]>1.1 || nvert[k][j][i]+nvert[k+1][j][i]>1.1) ) {
			// do nothing
		}
		else {
			ucont2[pos++] = ucont[k][j][i].x;
			ucont2[pos++] = ucont[k][j][i].y;
			ucont2[pos++] = ucont[k][j][i].z;
		}
	}
	
	DMDAVecRestoreArray(fda, user->Ucont, &ucont);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	VecRestoreArray(user->Ucont2, &ucont2);
	
	DMGlobalToLocalBegin(fda, user->Ucont, INSERT_VALUES, user->lUcont);
	DMGlobalToLocalEnd(fda, user->Ucont, INSERT_VALUES, user->lUcont);
}

void Convert_RHS_RHS2(UserCtx *user, Vec RHS, Vec RHS2)
{
	DMDALocalInfo	info = user->info;
	DM		da = user->da, fda = user->fda;
	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
	int i, j, k;
	PetscReal ***nvert, *rhs2;
	Cmpnts ***rhs;
	
	DMDAVecGetArray(fda, RHS, &rhs);
	DMDAVecGetArray(da, user->lNvert, &nvert);
	VecGetArray(RHS2, &rhs2);
	
	int pos=0;
	for(k=zs; k<ze; k++)
	for(j=ys; j<ye; j++)
	for(i=xs; i<xe; i++) {
		if(! (i==0 || i==mx-1 || j==0 || j==my-1 || k==0 || k==mz-1) &&
					(nvert[k][j][i]+nvert[k][j][i+1]>1.1 || nvert[k][j][i]+nvert[k][j+1][i]>1.1 || nvert[k][j][i]+nvert[k+1][j][i]>1.1) ) {
			// do nothing
		}
		else {
			rhs2[pos++] = rhs[k][j][i].x;
			rhs2[pos++] = rhs[k][j][i].y;
			rhs2[pos++] = rhs[k][j][i].z;
		}
	}
	
	DMDAVecRestoreArray(fda, RHS, &rhs);
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	VecRestoreArray(RHS2, &rhs2);
}
