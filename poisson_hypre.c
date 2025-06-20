#include "variables.h"
#include "petscksp.h"
#include "petscpc.h"
#include "petscpcmg.h"
#include <stdlib.h>

//#define PCG_POISSON

/*
extern int i_periodic, j_periodic, k_periodic, pseudo_periodic;
extern int block_number, freesurface, immersed, ti, tistart, les, tiout;
extern int tistart;
extern double poisson_tol;
extern char path[256];
extern double mean_pressure_gradient;
*/
HYPRE_Solver pcg_solver_p, precon_p;
HYPRE_IJMatrix Ap;
HYPRE_ParCSRMatrix par_Ap;
HYPRE_IJVector Vec_p, Vec_p_rhs;
HYPRE_ParVector par_Vec_p, par_Vec_p_rhs;

void Destroy_Hypre_Solver()
{
	HYPRE_BoomerAMGDestroy(precon_p);
	#ifdef PCG_POISSON
	HYPRE_ParCSRPCGDestroy (pcg_solver_p);
	#else
	HYPRE_ParCSRGMRESDestroy (pcg_solver_p);
	#endif
}

void Create_Hypre_Solver()
{
	/*
	0  CLJP-coarsening (a parallel coarsening algorithm using independent sets.
	1  classical Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!)
	3  classical Ruge-Stueben coarsening on each processor, followed by a third pass, which adds coarse
	   points on the boundaries
	6  Falgout coarsening (uses 1 first, followed by CLJP using the interior coarse points
	   generated by 1 as its first independent set)
	7  CLJP-coarsening (using a fixed random vector, for debugging purposes only)
	8  PMIS-coarsening (a parallel coarsening algorithm using independent sets, generating
	   lower complexities than CLJP, might also lead to slower convergence)
	9  PMIS-coarsening (using a fixed random vector, for debugging purposes only)
	10 HMIS-coarsening (uses one pass Ruge-Stueben on each processor independently, followed
	   by PMIS using the interior C-points generated as its first independent set)
	11 one-pass Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!)
	21 CGC coarsening by M. Griebel, B. Metsch and A. Schweitzer
	22 CGC-E coarsening by M. Griebel, B. Metsch and A.Schweitzer
	*/
	
	
	HYPRE_BoomerAMGCreate (&precon_p);
	HYPRE_BoomerAMGSetInterpType(precon_p, 13);	// 6:ext, 13: FF1
	
	if(amg_agg) {
	  HYPRE_BoomerAMGSetAggNumLevels(precon_p, amg_agg);	// FF1+aggresive coarsening is good for > 50mil grids
	}
	
	HYPRE_BoomerAMGSetStrongThreshold(precon_p, amg_thresh);// 0.5 : Cartesian, 0.6 : Distorted
	
	//HYPRE_BoomerAMGSetTol (precon_p, 1.e-6);
	HYPRE_BoomerAMGSetTol (precon_p, poisson_tol);
	HYPRE_BoomerAMGSetPrintLevel(precon_p,0);//1
	HYPRE_BoomerAMGSetCoarsenType(precon_p,amg_coarsentype);	// 0:CLJP, 6:Falgout, 8:PMIS, 10:HMIS, 
	HYPRE_BoomerAMGSetCycleType(precon_p,1);// 1
	HYPRE_BoomerAMGSetMaxIter(precon_p,1);
       	
	//HYPRE_BoomerAMGSetRelaxType (precon_p, 6);
	//HYPRE_BoomerAMGSetRelaxWt(precon_p,0.8);
	
	//HYPRE_BoomerAMGSetCycleNumSweeps(precon_p,5,3);	// 5 sweep at coarsest level
	//HYPRE_BoomerAMGSetLevelOuterWt(precon_p,0.5,0);
	//HYPRE_BoomerAMGSetSmoothType(precon_p, 7);	// more complex smoother
	//HYPRE_BoomerAMGSetSmoothNumSweeps(precon_p,2);
	
	// Pressure solver
	
	
#ifdef PCG_POISSON
	HYPRE_ParCSRPCGCreate ( PETSC_COMM_WORLD, &pcg_solver_p );
	HYPRE_PCGSetTwoNorm( pcg_solver_p, 1);
	HYPRE_PCGSetLogging ( pcg_solver_p, 1);
	HYPRE_PCGSetPrecond ( pcg_solver_p, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precon_p );
	HYPRE_PCGSetMaxIter( pcg_solver_p, poisson_it );
	HYPRE_PCGSetTol( pcg_solver_p, poisson_tol );
	HYPRE_PCGSetPrintLevel( pcg_solver_p, 3 ); 
#else	
	HYPRE_ParCSRGMRESCreate ( PETSC_COMM_WORLD, &pcg_solver_p );
	HYPRE_ParCSRGMRESSetKDim (pcg_solver_p, 51);
	HYPRE_GMRESSetPrecond ( pcg_solver_p, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precon_p );
	HYPRE_GMRESSetMaxIter( pcg_solver_p, poisson_it);
	HYPRE_GMRESSetTol( pcg_solver_p, poisson_tol );
	HYPRE_GMRESSetPrintLevel( pcg_solver_p, 3 ); 
#endif
};

/* deactivate (xiaolei)
void MatHYPRE_IJMatrixCopy(Mat v,HYPRE_IJMatrix &ij)
 {
	PetscInt i, rstart,rend;//
	const PetscInt *cols;
	PetscInt ncols;
	const PetscScalar *values;

	//HYPRE_IJMatrixInitialize(ij);	//->> Call just once when initialize matrix. memory leaks
	MatGetOwnershipRange(v,&rstart,&rend);

	for (i=rstart; i<rend; i++) {
		 MatGetRow(v,i,&ncols,&cols,&values);
		// MatGetRow(Mat mat,PetscInt row,PetscInt *ncols,const PetscInt *cols[],const PetscScalar *vals[])

		 HYPRE_IJMatrixSetValues(ij,1,&ncols,&i,cols,values);
		 MatRestoreRow(v,i,&ncols,&cols,&values);
	}

	HYPRE_IJMatrixAssemble(ij);
 };
*/ 


/* deactivate (xiaolei)
 void Remove_Nullspace_Scale(UserCtx *user, HYPRE_IJVector &B, PetscInt i_lower)
 {
	if( user->bctype[3] == -10 ) return;
	 
	DMDALocalInfo info = user->info;
	int	xs = info.xs, xe = info.xs + info.xm;
	int ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;

	int i, j, k;
	PetscReal	***nvert, ***gid, ***aj;


	int lxs = xs, lxe = xe;
	int lys = ys, lye = ye;
	int lzs = zs, lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
  	
	DMDAVecGetArray(user->da, user->lAj, &aj);
	DMDAVecGetArray(user->da, user->Gid, &gid);
	DMDAVecGetArray(user->da, user->lNvert, &nvert);
	
	int lcount=0;
	double sum, sum_aj;
	double localsum=0, localsum_aj=0;
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		double val;
		if (nvert[k][j][i] < 0.1) {
			PetscInt idx=(PetscInt)gid[k][j][i];
			HYPRE_IJVectorGetValues(B, 1, &idx, &val);
			localsum += val;
			localsum_aj += aj[k][j][i];
			lcount++;
		}
	}
	
	GlobalSum_All(&localsum, &sum, PETSC_COMM_WORLD);
	GlobalSum_All(&localsum_aj, &sum_aj, PETSC_COMM_WORLD);
	MPI_Allreduce( &lcount, &user->rhs_count, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
	
	
	//double val = -sum / sum_aj;
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		if(nvert[k][j][i]<0.1) {
			PetscInt idx=(PetscInt)gid[k][j][i];
			double val = -sum * aj[k][j][i] / sum_aj;
			HYPRE_IJVectorAddToValues(B, 1, &idx, &val);
		}
	}
	HYPRE_IJVectorAssemble(B);
	
	// scale by aj
	//for (k=lzs; k<lze; k++)
	//for (j=lys; j<lye; j++)
	//for (i=lxs; i<lxe; i++) {
	//	double val;
	//	if (nvert[k][j][i] < 0.1) {
	//		int idx=(int)gid[k][j][i];
	//		HYPRE_IJVectorGetValues(B, 1, &idx, &val);
	//		val /= sum_aj;
	//		HYPRE_IJVectorSetValues(B, 1, &idx, &val);
	//	}
	//}
	//HYPRE_IJVectorAssemble(B);
	
	DMDAVecRestoreArray(user->da, user->lAj, &aj);
	DMDAVecRestoreArray(user->da, user->Gid, &gid);
	DMDAVecRestoreArray(user->da, user->lNvert, &nvert);
 }
 */
 void Remove_Nullspace(UserCtx *user, HYPRE_IJVector &B, int i_lower)
 {
	if ( user->bctype[3] == -10 ) return;
	 	 
	DMDALocalInfo info = user->info;
	int	xs = info.xs, xe = info.xs + info.xm;
	int ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;

	int i, j, k;
	PetscReal	***nvert, ***gid;


	int lxs = xs, lxe = xe;
	int lys = ys, lye = ye;
	int lzs = zs, lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
  	
	DMDAVecGetArray(user->da, user->Gid, &gid);
	DMDAVecGetArray(user->da, user->lNvert, &nvert);
	
	int lcount=0;
	double localsum=0;
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		double val;
		if (nvert[k][j][i] < 0.1) {
			// PetscInt idx=(PetscInt)gid[k][j][i]; (xiaolei)
			int idx=(PetscInt)gid[k][j][i]; // (xiaolei)
			HYPRE_IJVectorGetValues(B, 1, &idx, &val);
			localsum += val;
			lcount++;
		}
	}

	double sum;
	GlobalSum_All(&localsum, &sum, PETSC_COMM_WORLD);
	MPI_Allreduce(&lcount, &user->rhs_count, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
	
	double val = -sum/(double) (user->rhs_count);	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		if(nvert[k][j][i]<0.1) {
			// PetscInt idx=(PetscInt)gid[k][j][i]; (xiaolei)
			int idx=(PetscInt)gid[k][j][i]; // (xiaolei)
			HYPRE_IJVectorAddToValues(B, 1, &idx, &val);
		}
	}
	HYPRE_IJVectorAssemble(B);
	
	DMDAVecRestoreArray(user->da, user->Gid, &gid);
	DMDAVecRestoreArray(user->da, user->lNvert, &nvert);
 }
 
  void PoissonRHS2_hypre(UserCtx *user, HYPRE_IJVector &B, PetscInt i_lower)
{
	DMDALocalInfo info = user->info;
	int	xs = info.xs, xe = info.xs + info.xm;
	int ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;

	int i, j, k;
	PetscReal	***nvert, ***aj, ***gid, dt = user->dt;
	Cmpnts	***ucont, ***cent;
	Cmpnts	***icsi, ***ieta, ***izet;
	Cmpnts	***jcsi, ***jeta, ***jzet;
	Cmpnts	***kcsi, ***keta, ***kzet;
	PetscReal	***iaj, ***jaj, ***kaj, ***rho, ***level, ***p;
	
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
	DMDAVecGetArray(user->fda, user->lCent, &cent);
	
	DMDAVecGetArray(user->fda, user->lICsi, &icsi);
	DMDAVecGetArray(user->fda, user->lIEta, &ieta);
	DMDAVecGetArray(user->fda, user->lIZet, &izet);

	DMDAVecGetArray(user->fda, user->lJCsi, &jcsi);
	DMDAVecGetArray(user->fda, user->lJEta, &jeta);
	DMDAVecGetArray(user->fda, user->lJZet, &jzet);

	DMDAVecGetArray(user->fda, user->lKCsi, &kcsi);
	DMDAVecGetArray(user->fda, user->lKEta, &keta);
	DMDAVecGetArray(user->fda, user->lKZet, &kzet);
	
	DMDAVecGetArray(user->da, user->lIAj, &iaj);
	DMDAVecGetArray(user->da, user->lJAj, &jaj);
	DMDAVecGetArray(user->da, user->lKAj, &kaj);
	
	DMDAVecGetArray(user->da, user->lP, &p);
	
	if(levelset) {
		DMDAVecGetArray(user->da, user->lDensity, &rho);
		DMDAVecGetArray(user->da, user->lLevelset, &level);
	}
		
	int lcount=0;
	double localsum=0;

	/*
	// xiaolei add
        Vec RHS;
        VecDuplicate(user->lP, &RHS);
        VecSet(RHS,0.0);

	PetscReal ***rhs;
	DMDAVecGetArray(user->da, RHS, &rhs);
	*/
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		double val;

		if (nvert[k][j][i] >= 0.1) {
			/*if( (int) (gid[k][j][i]) >=0 ) val = 0;	// for fsi
			else */continue;
		}
		else {
			double coeff=time_coeff();

			val=0;
			
			val -= ucont[k][j][i].x;

			if (i==1 && i_periodic)
                          val += ucont[k][j][mx-2].x;
			else if (i==1 && ii_periodic)
                          val += ucont[k][j][-2].x;
			else
                          val += ucont[k][j][i-1].x;

			val -= ucont[k][j][i].y;

			if (j==1 && j_periodic)
                          val += ucont[k][my-2][i].y;
			else if (j==1 && jj_periodic)
                          val += ucont[k][-2][i].y;
			else
                          val += ucont[k][j-1][i].y;

			val -= ucont[k][j][i].z;

			if (k==1 && k_periodic)
                          val += ucont[mz-2][j][i].z;
			else if (k==1 && kk_periodic)
                          val += ucont[-2][j][i].z;
			else
                          val += ucont[k-1][j][i].z;
		
			val *=  -1.0 / dt * user->st * coeff;
			/*
			if(levelset && user->bctype[5]==4 && k==mz-2) {
				double g33 = (kzet[k][j][i].x * kzet[k][j][i].x + kzet[k][j][i].y * kzet[k][j][i].y + kzet[k][j][i].z * kzet[k][j][i].z) * kaj[k][j][i];
				double pBC_old = 1.5 * p[k][j][i] - 0.5 * p[k-1][j][i];
				double pBC_new = p[k+1][j][i];
				double phiBC_new = pBC_new - pBC_old;
				val -= g33 * phiBC_new / rho[k][j][i];
				
			}*/
			if(levelset==1 && user->bctype[5]==4 && k==mz-2 && nvert[k][j][i]<0.1) {
				/* 
					RHS = - 1/rho * g33 * phi_BC / 0.5
					where
						phi_BC = p_(n+1)_BC - p_(n)_BC
						p_(n+1)_BC = rho[k][j][i] * gravity * level[k][j][i];
						p_(n)_BC = 1.5 * p[k][j][i] - 0.5 * p[k-1][j][i];
				*/

				// xiaolei deactivate
				/*							
				double pBC;
				if(inlet_y_flag)  pBC = - rho[k][j][i] * gravity_y * level[k][j][i];
				else if(inlet_z_flag) pBC = - rho[k][j][i] * gravity_z * level[k][j][i]; 
				double phi_BC = pBC - (1.5 * p[k][j][i] - 0.5 * p[k-1][j][i]);
				
				double g33 = (kzet[k][j][i].x * kzet[k][j][i].x + kzet[k][j][i].y * kzet[k][j][i].y + kzet[k][j][i].z * kzet[k][j][i].z) * kaj[k][j][i];
				val -= g33 / rho[k][j][i] * phi_BC / 0.5;	// assume grid is orthogonal in the k-direction at the outlet.
				*/
	
			}

			/* xiaolei add*/
			/*
			if(inflow_levelset && levelset==1 && user->bctype[4]==5 && k==1 && nvert[k][j][i]<0.1) {

				double pBC;
				if(inlet_y_flag)  pBC = - rho[k-1][j][i] * gravity_y * level[k-1][j][i];
				else if(inlet_z_flag) pBC = - rho[k-1][j][i] * gravity_z * level[k-1][j][i]; 
	
				double phi_BC = pBC - (1.5 * p[k][j][i] - 0.5 * p[k-1][j][i]);
				
				double g33 = (kzet[k][j][i].x * kzet[k][j][i].x + kzet[k][j][i].y * kzet[k][j][i].y + kzet[k][j][i].z * kzet[k][j][i].z) * kaj[k][j][i];
				val -= g33 / rho[k-1][j][i] * phi_BC / 0.5;
			}
			*/
			lcount++;
		}
		// PetscInt idx=(PetscInt)gid[k][j][i]; // (xiaolei)
		int idx=(PetscInt)gid[k][j][i];
	
		//rhs[k][j][i]=val; // xiaolei add
	
		HYPRE_IJVectorSetValues(B, 1, &idx, &val);
		localsum += val;
	}
	
	HYPRE_IJVectorAssemble(B);
	
	//if(levelset && user->bctype[5]==4) { }
	//else 
	//Remove_Nullspace_Scale(user, B, i_lower); //101103
  
	DMDAVecRestoreArray(user->da, user->lP, &p);
	
	if(levelset) {
		DMDAVecRestoreArray(user->da, user->lDensity, &rho);
		DMDAVecRestoreArray(user->da, user->lLevelset, &level);
	}

	DMDAVecRestoreArray(user->fda, user->lCent, &cent);
	DMDAVecRestoreArray(user->da, user->Gid, &gid);
	DMDAVecRestoreArray(user->fda, user->lUcont, &ucont);
	DMDAVecRestoreArray(user->da, user->lNvert, &nvert);
	DMDAVecRestoreArray(user->da, user->lAj, &aj);

	DMDAVecRestoreArray(user->fda, user->lICsi, &icsi);
	DMDAVecRestoreArray(user->fda, user->lIEta, &ieta);
	DMDAVecRestoreArray(user->fda, user->lIZet, &izet);

	DMDAVecRestoreArray(user->fda, user->lJCsi, &jcsi);
	DMDAVecRestoreArray(user->fda, user->lJEta, &jeta);
	DMDAVecRestoreArray(user->fda, user->lJZet, &jzet);

	DMDAVecRestoreArray(user->fda, user->lKCsi, &kcsi);
	DMDAVecRestoreArray(user->fda, user->lKEta, &keta);
	DMDAVecRestoreArray(user->fda, user->lKZet, &kzet);
	
	DMDAVecRestoreArray(user->da, user->lIAj, &iaj);
	DMDAVecRestoreArray(user->da, user->lJAj, &jaj);
	DMDAVecRestoreArray(user->da, user->lKAj, &kaj);

	// xiaolei add
	//DMDAVecRestoreArray(user->da, RHS, &rhs);

	/*
	char fname[80];
	sprintf(fname,"RHS_Poisson");
	TECIOOut_rhs_da(user, RHS, fname);
	

        VecDestroy(&RHS);
	*/
}

void Petsc_to_Hypre_Vector(Vec A, HYPRE_IJVector &B, PetscInt i_lower)
{
	PetscInt localsize;
	PetscReal *a;
	
	VecGetLocalSize(A, &localsize);
	// std::vector<PetscInt> indices (localsize); // (xiaolei)
	std::vector<int> indices (localsize); // (xiaolei)
	
	VecGetArray(A, &a);
	for(PetscInt i=0; i<localsize; i++) /*values[i] = a[i],*/ indices[i] = i_lower + i;
	
	HYPRE_IJVectorSetValues(B, localsize, &indices[0], &a[0]);
	HYPRE_IJVectorAssemble(B);
	
	VecRestoreArray(A, &a);
	
//	std::vector<PetscInt> ().swap(indices); // (xiaolei)
	std::vector<int> ().swap(indices); // (xiaolei)
};

void Hypre_to_Petsc_Vector(HYPRE_IJVector &B, Vec A, PetscInt i_lower)
{
	PetscInt localsize;
	VecGetLocalSize(A, &localsize);
	
	std::vector<double> values (localsize);
//	std::vector<PetscInt> indices (localsize); // (xiaolei)

	std::vector<int> indices (localsize); // (xiaolei)
	
	for(PetscInt i=0; i<localsize; i++) indices[i] = i_lower + i;
	HYPRE_IJVectorGetValues(B, localsize, &indices[0], &values[0]);
	
	PetscReal *a;
	VecGetArray(A, &a);
	for(PetscInt i=0; i<localsize; i++) a[i] = values[i];
	VecRestoreArray(A, &a);
};



void Destroy_Hypre_Matrix(UserCtx *user)
{
	HYPRE_IJMatrixDestroy(Ap);
}

void Destroy_Hypre_Vector(UserCtx *user)
{
	HYPRE_IJVectorDestroy(Vec_p);
	HYPRE_IJVectorDestroy(Vec_p_rhs);
}

void Create_Hypre_Matrix(UserCtx *user)
{
	PetscInt localsize_p;
	VecGetLocalSize(user->Phi2, &localsize_p);
	
	PetscInt p_lower = user->p_global_begin;
	PetscInt p_upper = p_lower + localsize_p - 1;
	
//	std::vector<PetscInt> nz_p (localsize_p); (xiaolei)
	std::vector<int> nz_p (localsize_p); // xiaolei
	std::fill ( nz_p.begin(), nz_p.end(), 19);

	PetscPrintf(PETSC_COMM_WORLD, "\nbegin HYPRE_IJMatrixCreate\n");
	HYPRE_IJMatrixCreate(PETSC_COMM_WORLD, p_lower, p_upper, p_lower, p_upper, &Ap);
	HYPRE_IJMatrixSetObjectType(Ap, HYPRE_PARCSR);
	HYPRE_IJMatrixSetRowSizes(Ap, &nz_p[0]);
	//HYPRE_IJMatrixSetMaxOffProcElmts (Ap, 10);
	HYPRE_IJMatrixInitialize(Ap);
	PetscPrintf(PETSC_COMM_WORLD, "end HYPRE_IJMatrixCreate\n\n");
}

void Create_Hypre_Vector(UserCtx *user)
{
	PetscInt localsize_p;
	
	VecGetLocalSize(user->Phi2, &localsize_p);
	
	PetscInt p_lower = user->p_global_begin;
	PetscInt p_upper = p_lower + localsize_p - 1;
	
	PetscPrintf(PETSC_COMM_WORLD, "\nbegin HYPRE_IJVectorCreate\n");
	
	// p vector
	HYPRE_IJVectorCreate(PETSC_COMM_WORLD, p_lower, p_upper, &Vec_p);
	HYPRE_IJVectorSetObjectType(Vec_p, HYPRE_PARCSR);
	//HYPRE_IJVectorSetMaxOffProcElmts (Vec_p, 10);//?
		
	HYPRE_IJVectorCreate(PETSC_COMM_WORLD, p_lower, p_upper, &Vec_p_rhs);
	HYPRE_IJVectorSetObjectType(Vec_p_rhs, HYPRE_PARCSR);
	//HYPRE_IJVectorSetMaxOffProcElmts (Vec_p_rhs, 10);//?
	
	HYPRE_IJVectorInitialize(Vec_p);
	HYPRE_IJVectorInitialize(Vec_p_rhs);
	
	PetscPrintf(PETSC_COMM_WORLD, "end HYPRE_IJVectorCreate\n\n");
};


void PoissonSolver_Hypre(UserCtx *user, IBMNodes *ibm, IBMInfo *ibminfo)
{
	DMDALocalInfo info = user->info;
	int	xs = info.xs, xe = info.xs + info.xm;
	int ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
	
	int lxs = xs, lxe = xe;
	int lys = ys, lye = ye;
	int lzs = zs, lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
	
	int l;
	const int bi=0;

	PetscReal ts,te,cput;
	PetscGetTime(&ts);
	
	PetscReal _ts,_te,_cput; // xiaolei add
//	extern int movefsi, rotatefsi;
	
	PetscGetTime(&_ts);
	if(ti==tistart /*&& !levelset*/) {
		
		// 07.09.2009 Seokkoo
	  setup_lidx2(user);

		Create_Hypre_Matrix(user);
		Create_Hypre_Vector(user);

		
		PoissonLHSNew(user, Ap);
		//MatHYPRE_IJMatrixCopy(user->A, Ap);
		//MatDestroy(&user->A);	
		user->assignedA=PETSC_FALSE;
		
		HYPRE_IJMatrixGetObject(Ap, (void**) &par_Ap);
		
		Create_Hypre_Solver();
	}
	else if(( sediment && ti == (ti/bed_avg)*bed_avg ) || movefsi || rotatefsi || (levelset && !fix_level) ) {
		/*VecDestroy(&user->Gid); destroyed inside lidx2 func*/
		//PoissonLHSNew(user, ibm, user->ibm_intp);
	  setup_lidx2(user);

		Destroy_Hypre_Matrix(user);
		Destroy_Hypre_Vector(user);
		
		Create_Hypre_Matrix(user);
		Create_Hypre_Vector(user);
		
		PoissonLHSNew(user, Ap);
		//MatHYPRE_IJMatrixCopy(user->A, Ap);
		//MatDestroy(&user->A);	
		user->assignedA=PETSC_FALSE;
		
		HYPRE_IJMatrixGetObject(Ap, (void**) &par_Ap);
		
		Destroy_Hypre_Solver();
		Create_Hypre_Solver();
	}
	

	PetscGetTime(&_te);

	_cput=_te-_ts;
	PetscPrintf(PETSC_COMM_WORLD, "%d(poisson_part1)  %.2e(s)\n", ti, _cput);


	
	PetscGetTime(&_ts);
	PetscReal ibm_Flux, ibm_Area;

	
	if(immersed==1) VolumeFlux(&user[bi], &ibm_Flux, &ibm_Area, 0);
	else if(immersed==2) {
	  VolumeFlux(&user[bi], &ibm_Flux, &ibm_Area, 0);
	  Add_IBMFlux_to_Outlet(&user[bi], ibm_Flux);
	}
	else VolumeFlux(&user[bi], user[bi].Ucont, &ibm_Flux, &ibm_Area);
			
	PoissonRHS2_hypre(user, Vec_p_rhs, user->p_global_begin);
	
	PetscPrintf(PETSC_COMM_WORLD, "Petsc_to_Hypre_Vector...Phi2 begin\n");
	// Petsc_to_Hypre_Vector(user->Phi2, Vec_p, user->p_global_begin); // xiaolei deactivate
	PetscPrintf(PETSC_COMM_WORLD, "Petsc_to_Hypre_Vector...Phi2 end\n");
	
	PetscGetTime(&_te);

	_cput=_te-_ts;
	PetscPrintf(PETSC_COMM_WORLD, "%d(poisson_part2)  %.2e(s)\n", ti, _cput);



//	HYPRE_IJVectorGetObject(Vec_p, (void **) &par_Vec_p);	// sudden crash?
//	HYPRE_IJVectorGetObject(Vec_p_rhs, (void **) &par_Vec_p_rhs);

	PetscGetTime(&_ts);
	if(ti==tistart) {
		HYPRE_IJVectorGetObject(Vec_p, (void **) &par_Vec_p);
		HYPRE_IJVectorGetObject(Vec_p_rhs, (void **) &par_Vec_p_rhs);
		
		#ifdef PCG_POISSON
		HYPRE_ParCSRPCGSetup (pcg_solver_p, par_Ap, par_Vec_p_rhs, par_Vec_p);
		#else
		HYPRE_ParCSRGMRESSetup (pcg_solver_p, par_Ap, par_Vec_p_rhs, par_Vec_p);
		#endif
	}
	else if(( sediment && ti == (ti/bed_avg)*bed_avg) || movefsi || rotatefsi || (levelset && !fix_level) ) {
		HYPRE_IJVectorGetObject(Vec_p, (void **) &par_Vec_p);
		HYPRE_IJVectorGetObject(Vec_p_rhs, (void **) &par_Vec_p_rhs);

		#ifdef PCG_POISSON
		HYPRE_ParCSRPCGSetup (pcg_solver_p, par_Ap, par_Vec_p_rhs, par_Vec_p);
		#else
		HYPRE_ParCSRGMRESSetup (pcg_solver_p, par_Ap, par_Vec_p_rhs, par_Vec_p);
		#endif
	}
	
	PetscGetTime(&_te);

	_cput=_te-_ts;
	PetscPrintf(PETSC_COMM_WORLD, "%d(poisson_part3)  %.2e(s)\n", ti, _cput);



	
	
	PetscGetTime(&_ts);
	PetscPrintf(PETSC_COMM_WORLD, "Solving Poisson ...\n");
	
	MPI_Barrier(PETSC_COMM_WORLD);
	
	//PetscGetTime(&ts);
	#ifdef PCG_POISSON
	HYPRE_ParCSRPCGSolve (pcg_solver_p, par_Ap, par_Vec_p_rhs, par_Vec_p);
	#else
	HYPRE_ParCSRGMRESSolve (pcg_solver_p, par_Ap, par_Vec_p_rhs, par_Vec_p);
	#endif

	PetscGetTime(&_te);

	_cput=_te-_ts;
	PetscPrintf(PETSC_COMM_WORLD, "%d(poisson_part4)  %.2e(s)\n", ti, _cput);





	PetscGetTime(&te);

	// xiaolei deactivate
	//if(levelset && user->bctype[5]==4) { }
	/*else if ( movefsi || rotatefsi ) { }*/
	//else 
	  Remove_Nullspace(user, Vec_p, user->p_global_begin);
	
	Hypre_to_Petsc_Vector(Vec_p, user->Phi2, user->p_global_begin);
	
	
	
	MPI_Barrier(PETSC_COMM_WORLD);
	
	Convert_Phi2_Phi(user);
	
	DMGlobalToLocalBegin(user->da, user->Phi, INSERT_VALUES, user->lPhi);
	DMGlobalToLocalEnd(user->da, user->Phi, INSERT_VALUES, user->lPhi);
		
      
	cput=te-ts;
	int rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	
	if (!rank) {
		FILE *f;
		char filen[80];
		sprintf(filen, "%s/Converge_dU", path);
		f = fopen(filen, "a");
		PetscFPrintf(PETSC_COMM_WORLD, f, "%d(poisson)  %.2e(s)", ti, cput);
		fclose(f);
	}

	MPI_Barrier(PETSC_COMM_WORLD);
}



