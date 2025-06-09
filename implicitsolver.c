#include "variables.h"
#include "petscksp.h"
#include "petscpc.h"
/*
extern int block_number, NumberOfBodies;
extern int immersed;
extern int ti,tistart;
extern int imp_MAX_IT;
extern PetscReal imp_atol, imp_rtol, imp_stol;
extern int mg_idx, mg_preItr, mg_poItr, mg_MAX_IT;
extern char path[256];
extern IBMNodes	*ibm_ptr;
extern  FSInfo        *fsi_ptr;


extern double time_coeff();

extern void IB_BC(UserCtx *user);
extern int les;
*/

template < typename T > T abs(T a) { return (a >= 0) ? a : -a; }

PetscErrorCode CalcRHS(UserCtx *user, int dudt)
{
  int      i, j, k;
  DMDALocalInfo	info ;
  int	xs, xe, ys, ye, zs, ze;
  int  	mx,my,mz;	
  int	lxs, lxe, lys, lye, lzs, lze;
  Cmpnts        ***rhs;
  PetscReal     ***nvert;
  Vec           dUcont;

  DM            da = user->da,fda = user->fda;
  info = user->info;
  
  xs = info.xs; xe = info.xs + info.xm;
  ys = info.ys; ye = info.ys + info.ym;
  zs = info.zs; ze = info.zs + info.zm;
  mx = info.mx; my = info.my; mz = info.mz;
  
  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;
  
  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;
  
  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;
  
  int     mz_end;
  if (user->bctype[5]==8) 
    mz_end=mz-2;
  else
    mz_end=mz-3;
   
  
	double threshold=0.1;   //03.29
	
	
  
	VecDuplicate(user->Ucont, &dUcont);

  // Calculate du/dt
  // du = 1.5 u - 2 u_o + 0.5 u_rm1
  
  double coeff = time_coeff();
  
  if (coeff>1.4 && coeff<1.6) {
	VecCopy(user->Ucont, dUcont);
	VecScale(dUcont, 1.5);
	VecAXPY(dUcont, -2.,user->Ucont_o );
	VecAXPY(dUcont, 0.5, user->Ucont_rm1);
  }
  /*
  else if (COEF_TIME_ACCURACY>1.8 && COEF_TIME_ACCURACY<1.9) {
		VecCopy(user->Ucont, dUcont);
		VecScale(dUcont, 11./6.);
		VecAXPY(dUcont, -3.,user->Ucont_o );
		VecAXPY(dUcont, 1.5, user->Ucont_rm1);
		VecAXPY(dUcont, -1./3., user->Ucont_rm2);
  }*/

  else {
    VecWAXPY(dUcont, -1., user->Ucont_o, user->Ucont);
  }

      // Calc the RHS
	VecSet(user->Rhs,0);
	Formfunction_2(user, user->Rhs, 1.0);
  //FormFunction1(user->Ucont, user->Rhs, user);
  //extern PetscErrorCode Formfunction_2(UserCtx *user, Vec Rhs);
  //Formfunction_2(user, user->Rhs);
    
  // Add -du/dt to right hand side
  VecAXPY(user->Rhs, -1./user->dt, dUcont);

  // set rhs BC
  DMDAVecGetArray(fda, user->Rhs, &rhs);//
  DMDAVecGetArray(da, user->lNvert, &nvert);//
  
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
		
		if ( (i==mx-2 && !i_periodic && !ii_periodic) || (nvert[k][j][i]+nvert[k][j][i+1])>threshold ) {
			rhs[k][j][i].x=0.;
		} 
		
		if ( (j==my-2 && !j_periodic && !jj_periodic) || (nvert[k][j][i]+nvert[k][j+1][i])>threshold ) {
			rhs[k][j][i].y=0.;
		} 
		
		if ( (k==mz-2 && !k_periodic && !kk_periodic) || (nvert[k][j][i]+nvert[k+1][j][i])>threshold ) {
			rhs[k][j][i].z=0.;
		}
      }
    }
  }  

  
  /* i direction boundary conditions*/
  if (xs ==0) {
    i = 0;
    for (k=zs; k<ze; k++) {
      for (j=ys; j<ye; j++) {
	rhs[k][j][i].x = 0;
	rhs[k][j][i].y = 0;
	rhs[k][j][i].z = 0;
      }
    }
  }

  if (xe == mx) {
    for (k=zs; k<ze; k++) {
      for (j=ys; j<ye; j++) {
	if(!i_periodic && !ii_periodic) {
		i = mx-2;
		rhs[k][j][i].x = 0;
	}
	i = mx-1;
	rhs[k][j][i].x = 0;
	rhs[k][j][i].y = 0;
	rhs[k][j][i].z = 0;
      }
    }
  }

  /* j direction boundary conditions */
  if (ys == 0) {
    for (k=zs; k<ze; k++) {
      for (i=xs; i<xe; i++) {
	j=0;
	rhs[k][j][i].x = 0;
	rhs[k][j][i].y = 0;
	rhs[k][j][i].z = 0;
      }
    }
  }

  if (ye == my) {
    for (k=zs; k<ze; k++) {
      for (i=xs; i<xe; i++) {
	if(!j_periodic && !jj_periodic) {
		j=my-2;
		rhs[k][j][i].y = 0;
	}
	j=my-1;
	rhs[k][j][i].x = 0;
	rhs[k][j][i].y = 0;
	rhs[k][j][i].z = 0;
      }
    }
  }
  /* k direction boundary conditions */
  if (zs == 0) {
    k=0;
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
	rhs[k][j][i].x = 0;
	rhs[k][j][i].y = 0;
	rhs[k][j][i].z = 0;
      }
    }
  }

  if (ze == mz) {
    for (j=ys; j<ye; j++) {
      for (i=xs; i<xe; i++) {
	if(!k_periodic && !kk_periodic) {
		k=mz-2;
		rhs[k][j][i].z = 0;
	}
	k=mz-1;
	rhs[k][j][i].x = 0;
	rhs[k][j][i].y = 0;
	rhs[k][j][i].z = 0;
      }
    }
  }

  DMDAVecRestoreArray(fda, user->Rhs, &rhs);//
  DMDAVecRestoreArray(da, user->lNvert, &nvert);//
  

  VecDestroy(&dUcont);

  return(0);
}

PetscErrorCode MySNESMonitor(SNES snes, PetscInt n,PetscReal rnorm,void *dummy)
{
	PetscPrintf(PETSC_COMM_WORLD,"     (%D) SNES Residual norm %14.12e \n",n,rnorm);
	return 0;
}

PetscErrorCode FormFunction_seokkoo(SNES snes, Vec Ucont, Vec Rhs, void *ptr) //UserCtx *user)
{
	UserCtx *user = (UserCtx*)ptr;
	VecCopy(Ucont, user->Ucont);
	
	DMDALocalInfo	info = user->info;
	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
	
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
	
	
	DMGlobalToLocalBegin(user->fda, user->Ucont, INSERT_VALUES, user->lUcont);
	DMGlobalToLocalEnd(user->fda, user->Ucont, INSERT_VALUES, user->lUcont);
	
	Contra2Cart(user);
	IB_BC(user);
	
	/*
			0      1      2                     mx-2  mx-1
		|	|   1	|   2	|   3	|  ...	| mx-2| mx-1 |
	*/
			
	CalcRHS(user, 1);
	VecCopy(user->Rhs, Rhs);
	
	//VecScale(Rhs,user->dt);	// 11.9.2008
	//VecScale(Rhs,user->ren);	// 12.9.2008
	return(0);
}

PetscErrorCode FormFunction_seokkoo2(SNES snes, Vec Ucont2, Vec Rhs2, void *ptr)
{
	UserCtx *user = (UserCtx*)ptr;
	
	VecCopy(Ucont2, user->Ucont2);
	Convert_Ucont2_Ucont(user);
	
	CalcRHS(user, 1);
	Convert_RHS_RHS2(user, user->Rhs, Rhs2);
	
	return(0);
}


int snes_created=0;

PetscErrorCode Implicit_MatrixFree(UserCtx *user, IBMNodes *ibm, FSInfo *fsi)
{

	DMDALocalInfo	info = user->info;
	int	xs = info.xs, xe = info.xs + info.xm;
	int  	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;
	
	int	bi;
	Vec U;
	double norm;
	Vec Coor;
	Cmpnts ***ucont, ***lucont, ***coor, ***csi, ***eta, ***ucat, ***lucat, ***zet;
	PetscReal ***level;
	PetscInt i, j, k;
	

	int rank;
	PetscReal ts,te,cput;
	PetscGetTime(&ts);
	
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	
	
	for (bi=0; bi<block_number; bi++) {
		
		KSP ksp;
		PC pc;
		
		VecDuplicate(user[bi].Ucont, &user[bi].rhs);
		if(!snes_created) {
			SNESCreate(PETSC_COMM_WORLD,&user[bi].snes);
			SNESMonitorSet (user[bi].snes, MySNESMonitor, PETSC_NULL, PETSC_NULL);
		}
		SNESSetFunction(user[bi].snes,user[bi].rhs,FormFunction_SNES,(void *)&user[bi]);
		
		#if defined(PETSC_HAVE_ADIC___)
		DAGetMatrix(user->da,MATAIJ,&user[bi].J);
		ISColoring             iscoloring;
		DAGetColoring(user->da,IS_COLORING_GHOSTED,&iscoloring);
		MatSetColoring(J[bi],iscoloring);
		ISColoringDestroy(iscoloring);
		SNESSetJacobian(user[bi].snes,user[bi].J,user[bi].J,SNESDAComputeJacobianWithAdic,(void *)&user[bi]);
		#else
		MatCreateSNESMF(user[bi].snes, &user[bi].J);
		SNESSetJacobian(user[bi].snes,user[bi].J,user[bi].J,MatMFFDComputeJacobian,(void *)&user[bi]);
		#endif
		
		extern double imp_free_tol;
		SNESSetType(user[bi].snes, SNESTR);			//SNESTR,SNESLS	: SNESLS is better for stiff PDEs such as the one including IB but slower

		SNESSetMaxLinearSolveFailures(user[bi].snes,10000);
		SNESSetMaxNonlinearStepFailures(user[bi].snes,10000);		
		SNESKSPSetUseEW(user[bi].snes, PETSC_TRUE);
		SNESKSPSetParametersEW(user[bi].snes,3,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
		//SNESSetTolerances(user[bi].snes,imp_free_tol,PETSC_DEFAULT,PETSC_DEFAULT,50,50000);
		SNESSetTolerances(user[bi].snes, /*5.e-4*/PETSC_DEFAULT,imp_free_tol,PETSC_DEFAULT,50,50000);
		
		SNESGetKSP(user[bi].snes, &ksp);
		KSPGetPC(ksp,&pc);
		
		if(!snes_created) {
			KSPSetType(ksp, KSPGMRES);
			//KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);	//2009.09.22 poor performance
			//KSPSetInitialGuessKnoll(ksp, PETSC_TRUE);	//2009.09.22
			
			//KSPFischerGuess itg;
			//KSPFischerGuessCreate(ksp,1,100,&itg);
			//KSPSetFischerGuess(ksp, itg);	//2009.09.22
			
			//KSPGMRESSetPreAllocateVectors(ksp);	--> crazy thing consumes memory 
		}
		#if defined(PETSC_HAVE_ADIC___)
		PCSetType(pc,PCBJACOBI);
		#else
		PCSetType(pc,PCNONE);
		#endif
		int maxits=1000;	
		//double rtol=PETSC_DEFAULT, atol=imp_free_tol, dtol=PETSC_DEFAULT;
		double rtol=imp_free_tol, atol=PETSC_DEFAULT, dtol=PETSC_DEFAULT;

		KSPSetTolerances(ksp,rtol,atol,dtol,maxits);
		
		snes_created=1;
	}
	
	bi =0;
	//int it=0;
		
	
	VecDuplicate(user[bi].Ucont, &U);
	VecDuplicate(user[bi].Ucont, &(user[bi].Rhs));
	//extern int outflow_scale;
		
		
	InflowFlux(&(user[bi]));
	outflow_scale=0;
	FormBCS(&(user[bi]),&fsi[0]);
		
	VecCopy(user[bi].Ucont, U);
		

	SNESSolve(user[bi].snes, PETSC_NULL, U);
		
	PetscPrintf(PETSC_COMM_WORLD, "\nMomentum eqs computed ...\n");
	
	
	
	SNESGetFunctionNorm(user[bi].snes, &norm);
	PetscPrintf(PETSC_COMM_WORLD, "\nSNES residual norm=%.5e\n\n", norm);
		
	VecCopy(U, user[bi].Ucont);
		
	DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucont, INSERT_VALUES, user[bi].lUcont);
	DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucont, INSERT_VALUES, user[bi].lUcont);
	
	
	
	DMDAGetGhostedCoordinates(user[0].da, &Coor);
	if(levelset) DMDAVecGetArray(user[0].da, user[0].lLevelset, &level);
	DMDAVecGetArray(user[0].fda, Coor, &coor);
	DMDAVecGetArray(user[0].fda, user[0].Ucont, &ucont);
	DMDAVecGetArray(user[0].fda, user[0].Ucat, &ucat);
	DMDAVecGetArray(user[0].fda, user[0].lUcont, &lucont);
	DMDAVecGetArray(user[0].fda, user[0].lUcat, &lucat);
	DMDAVecGetArray(user[0].fda, user[0].lCsi, &csi);
	DMDAVecGetArray(user[0].fda, user[0].lEta, &eta);
	DMDAVecGetArray(user[0].fda, user[0].lZet, &zet);
	for (k=zs; k<ze; k++)
	for (j=ys; j<ye; j++)
	for (i=xs; i<xe; i++) {
	  
		if ( (user[0].bctype[0]==1 || user[0].bctype[0]==-1 || user[0].bctype[0]==-2 || user[0].bctype[0]==10) && i==0) ucont[k][j][i].x = 0;
		if ( (user[0].bctype[1]==1 || user[0].bctype[1]==-1 || user[0].bctype[1]==-2 || user[0].bctype[1]==10) && i==mx-2) ucont[k][j][i].x = 0;
		if ( (user[0].bctype[2]==1 || user[0].bctype[2]==-1 || user[0].bctype[2]==-2 || user[0].bctype[2]==10) && j==0) ucont[k][j][i].y = 0;
		if ( (user[0].bctype[3]==1 || user[0].bctype[3]==-1 || user[0].bctype[3]==-2 || user[0].bctype[3]==10 || user[0].bctype[3]==2) && j==my-2) ucont[k][j][i].y = 0;
		if ( (user[0].bctype[4]==1 || user[0].bctype[4]==-1 || user[0].bctype[4]==-2 || user[0].bctype[4]==10) && k==0) ucont[k][j][i].z = 0;
		if ( (user[0].bctype[5]==1 || user[0].bctype[5]==-1 || user[0].bctype[5]==-2 || user[0].bctype[5]==10) && k==mz-2) ucont[k][j][i].z = 0;
	  	/*
		if ( (user[0].bctype[0]==-1 || user[0].bctype[0]==-2) && i==1) ucont[k][j][i].x = 0;
		if ( (user[0].bctype[1]==-1 || user[0].bctype[1]==-2) && i==mx-3) ucont[k][j][i].x = 0;
		if ( (user[0].bctype[2]==-1 || user[0].bctype[2]==-2) && j==1) ucont[k][j][i].y = 0;
		if ( (user[0].bctype[3]==-1 || user[0].bctype[3]==-2) && j==my-3) ucont[k][j][i].y = 0;
		if ( (user[0].bctype[4]==-1 || user[0].bctype[4]==-2) && k==1) ucont[k][j][i].z = 0;
		if ( (user[0].bctype[5]==-1 || user[0].bctype[5]==-2) && k==mz-3) ucont[k][j][i].z = 0;
		*/
		/*
		if ( user[0].bctype[4]==5 && k==0 ) {
		  if(levelset) {
		    //if(level[k][j][i]>0) ucont[k][j][i].y = 0;
		    ucat[k][j][i].x =  lucat[k+1][j][i].x;
		    ucat[k][j][i].y =  lucat[k+1][j][i].y;
                    ucat[k][j][i].z =  lucat[k+1][j][i].z;
		    lucat[k][j][i] = ucat[k][j][i];
		    if(ucont[k][j][i].z<0) {
		      ucont[k][j][i].z = 0;
		      ucat[k][j][i].x = ucat[k][j][i].y = ucat[k][j][i].z = 0;
		      ucat[k+1][j][i] = ucat[k][j][i];
		    }
		  }
		}
		*/

		if ( user[0].bctype[3]==4 && j==my-2 ) {
		  ucat[k][j+1][i] = ucat[k][j][i];
		  lucat[k][j+1][i] = ucat[k][j+1][i];
		  ucont[k][j][i].y = 0.5*(lucat[k][j][i].x+lucat[k][j+1][i].x)*eta[k][j][i].x + 0.5*(lucat[k][j][i].y+lucat[k][j+1][i].y)*eta[k][j][i].y + 0.5*(lucat[k][j][i].z+lucat[k][j+1][i].z)*eta[k][j][i].z;
                }
		
		if ( ti && user[0].bctype[4]==4 && k==0 ) {
		  /*
		  ucat[k][j][i] = ucat[k+1][j][i];
                  lucat[k][j][i] = ucat[k][j][i];
                  ucont[k][j][i].z = 0.5*(lucat[k][j][i].x+lucat[k+1][j][i].x)*zet[k][j][i].x + 0.5*(lucat[k][j][i].y+lucat[k+1][j][i].y)*zet[k][j][i].y + 0.5*(lucat[k][j][i].z+lucat[k+1][j][i].z)*zet[k][j][i].z;
		  */
                  if(ucont[k][j][i].z<0) {
                    ucont[k][j][i].z = 0;
                    ucat[k][j][i].x = ucat[k][j][i].y = ucat[k][j][i].z = 0;
                    ucat[k+1][j][i] = ucat[k][j][i];
                  }
		}
		
		if ( ti && (user[0].bctype[5]==4 || user[0].bctype[5]==5) && k==mz-2 ) { //[5]==5(inlet) is for subcritical levelset flow with fixed outlet
		  ucat[k+1][j][i] = ucat[k][j][i]; 
		  lucat[k+1][j][i] = ucat[k+1][j][i];
		  ucont[k][j][i].z = 0.5*(lucat[k][j][i].x+lucat[k+1][j][i].x)*zet[k][j][i].x + 0.5*(lucat[k][j][i].y+lucat[k+1][j][i].y)*zet[k][j][i].y + 0.5*(lucat[k][j][i].z+lucat[k+1][j][i].z)*zet[k][j][i].z;
		  
		  if(ucont[k][j][i].z<0) {
		    /*if(!levelset)*/ {
		      ucont[k][j][i].z = 0;
		      ucat[k][j][i].x = ucat[k][j][i].y = ucat[k][j][i].z = 0;
		      ucat[k+1][j][i] = ucat[k][j][i];
		    }
		  }
		  
		}
		
		if (user->bctype[0]==11 && i==0 && (j!=0 && j!=my-1 && k!=0 && k!=mz-1) ) {
			double zc = ( coor[k][j][i+1].z + coor[k-1][j][i+1].z + coor[k][j-1][i+1].z + coor[k-1][j-1][i+1].z ) * 0.25;
			if( zc > 0 ) {
				ucont[k][j][i].x = 0 * csi[k][j][i].x + 0 * csi[k][j][i].y + lucat[k][j][i].z * csi[k][j][i].z;
			}
		}
	}
	if(levelset) DMDAVecRestoreArray(user[0].da, user[0].lLevelset, &level);
	DMDAVecRestoreArray(user[0].fda, Coor, &coor);
	DMDAVecRestoreArray(user[0].fda, user[0].Ucont, &ucont);
	DMDAVecRestoreArray(user[0].fda, user[0].Ucat, &ucat);
	DMDAVecRestoreArray(user[0].fda, user[0].lUcont, &lucont);
	DMDAVecRestoreArray(user[0].fda, user[0].lUcat, &lucat);
	DMDAVecRestoreArray(user[0].fda, user[0].lCsi, &csi);
	DMDAVecRestoreArray(user[0].fda, user[0].lEta, &eta);
	DMDAVecRestoreArray(user[0].fda, user[0].lZet, &zet);
	
	outflow_scale=1;
	FormBCS(&(user[bi]),&fsi[0]);

	if (block_number>1) Block_Interface_U(user);

	PetscGetTime(&te);
	cput=te-ts;
	if (!rank) {
		FILE *f;
		char filen[80];
  		sprintf(filen, "%s/Converge_dU", path);
		f = fopen(filen, "a");
		PetscFPrintf(PETSC_COMM_WORLD, f, "%d(momentum) %.2e(s) %le\n", ti, cput, norm);
		fclose(f);
	}
	VecDestroy(&U);
	VecDestroy(&user[0].Rhs);
	
	
	
	for (bi=0; bi<block_number; bi++) {
		DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucont, INSERT_VALUES, user[bi].lUcont);
		DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucont, INSERT_VALUES, user[bi].lUcont);
		VecWAXPY(user[bi].DUold, -1., user[bi].Ucont_o, user[bi].Ucont);
		/*
		
		SNESDestroy(&user[bi].snes);
		*/
		VecDestroy(&user[bi].rhs);
		MatDestroy(&user[bi].J);
	}
	
	double max_norm;
	VecMax(user->Ucat, &i, &max_norm);
	PetscPrintf(PETSC_COMM_WORLD, "\n*** Max Ucat = %e \n", max_norm);


        // Calculate Courant number. Currently requires that k is x direction,
        // i is y direction, and j is z direction.
        {
          Cmpnts ***ucat, ***cent;
          DMDAVecGetArray(user->fda, user->lUcat, &ucat);
          DMDAVecGetArray(user->fda, user->lCent, &cent);
          double courant_max = 0.0;

          for (k=zs; k<ze-1; k++)
          for (j=ys; j<ye-1; j++)
          for (i=xs; i<xe-1; i++) {

            // Calculate for x direction.
            double dist = cent[k][j][i].x - cent[k+1][j][i].x;
            if (dist != 0.0)
            {
              double courant = abs(ucat[k][j][i].x * user->dt / dist);
              courant_max = (courant_max > courant) ? courant_max : courant;
            }

            // Calculate for y direction.
            dist = cent[k][j][i].y - cent[k][j][i+1].y;
            if (dist != 0.0)
            {
              double courant = abs(ucat[k][j][i].y * user->dt / dist);
              courant_max = (courant_max > courant) ? courant_max : courant;
            }

            // Calculate for z direction.
            dist = cent[k][j][i].z - cent[k][j+1][i].z;
            if (dist != 0.0)
            {
              double courant = abs(ucat[k][j][i].z * user->dt / dist);
              courant_max = (courant_max > courant) ? courant_max : courant;
            }
          }

          double global_courant_max;
          MPI_Reduce(&courant_max, &global_courant_max, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);

          if (!rank) {
            PetscPrintf(PETSC_COMM_WORLD, "\nMaximum Courant number is %e.\n\n", global_courant_max);
            FILE *f;
            char filen[80];
            sprintf(filen, "Converge_dU");
            f = fopen(filen, "a");
            PetscFPrintf(PETSC_COMM_WORLD, f, "%d Courant %.5e\n", ti, global_courant_max);
            fclose(f);
          }

          DMDAVecRestoreArray(user->fda, user->lCent, &cent);
          DMDAVecRestoreArray(user->fda, user->lUcat, &ucat);
        }


	
        // Calculate energy equation stability criteria. Currently requires that k is x direction,
        // i is y direction, and j is z direction.
        if (user->temp)
        {
          Cmpnts ***ucat, ***cent;
          DMDAVecGetArray(user->fda, user->lUcat, &ucat);
          DMDAVecGetArray(user->fda, user->lCent, &cent);
          double h_max = 0.0, ts_max = 0.0;
          double alpha = 1.0 / (user->ren * user->Pr);

          for (k=zs; k<ze-1; k++)
          for (j=ys; j<ye-1; j++)
          for (i=xs; i<xe-1; i++) {

            // Calculate for x direction.
            double dist = abs(cent[k][j][i].x - cent[k+1][j][i].x);
            if (dist != 0.0)
            {
              double h = 2 * alpha / abs(ucat[k][j][i].x);
              double ts = dist * dist / (2 * alpha);
              h_max = (h_max > h) ? h_max : h;
              ts_max = (ts_max > ts) ? ts_max : ts;
            }

            // Calculate for y direction.
            dist = abs(cent[k][j][i].y - cent[k][j][i+1].y);
            if (dist != 0.0)
            {
              double h = 2 * alpha / abs(ucat[k][j][i].y);
              double ts = dist * dist / (2 * alpha);
              h_max = (h_max > h) ? h_max : h;
              ts_max = (ts_max > ts) ? ts_max : ts;
            }

            // Calculate for z direction.
            dist = abs(cent[k][j][i].z - cent[k][j+1][i].z);
            if (dist != 0.0)
            {
              double h = 2 * alpha / abs(ucat[k][j][i].z);
              double ts = dist * dist / (2 * alpha);
              h_max = (h_max > h) ? h_max : h;
              ts_max = (ts_max > ts) ? ts_max : ts;
            }
          }

          double global_h_max, global_ts_max;
          MPI_Reduce(&h_max, &global_h_max, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);
          MPI_Reduce(&ts_max, &global_ts_max, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);

          if (!rank)
            PetscPrintf(PETSC_COMM_WORLD, "\nTemperature stablity: h < %f and ts < %f.\n\n", global_h_max, global_ts_max);

          DMDAVecRestoreArray(user->fda, user->lCent, &cent);
          DMDAVecRestoreArray(user->fda, user->lUcat, &ucat);
        }


	
	Contra2Cart(user);
	return(0);
}

PetscReal LinearInterpolation(PetscReal host[2], PetscReal p, PetscReal val[2])
{
  if (fabs(host[0]-host[1])>1e-9) {

  return(val[0]*(p-host[1])/(host[0]-host[1])
       + val[1]*(p-host[0])/(host[1]-host[0]));
  } else {
    PetscPrintf(PETSC_COMM_SELF, "Linear intrp Failed!!!!!!!!!!!!!\n"); 
  }
  return(0.);
}

  
PetscReal BilinearInterpolation(Cpt2D host[4], Cpt2D p, PetscReal val[4])
{
  /* Needs special ordering of host
             vkji 
     host[0]=v?00
     host[1]=v?10
     host[2]=v?01
     host[3]=v?11

  */
  PetscReal  v,x23,x01,v01,v23;

  // bilinear interpolation
  // in y
  if (fabs(host[0].y-host[1].y)>1e-9) {
  v01=val[0]*(p.y-host[1].y)/(host[0].y-host[1].y)
    + val[1]*(p.y-host[0].y)/(host[1].y-host[0].y);

  x01=host[0].x*(p.y-host[1].y)/(host[0].y-host[1].y)
    + host[1].x*(p.y-host[0].y)/(host[1].y-host[0].y);

  v23=val[2]*(p.y-host[3].y)/(host[2].y-host[3].y)
    + val[3]*(p.y-host[2].y)/(host[3].y-host[2].y);

  x23=host[2].x*(p.y-host[3].y)/(host[2].y-host[3].y)
    + host[3].x*(p.y-host[2].y)/(host[3].y-host[2].y);
  
  // in x
  if (fabs(x01-x23)>1e-9) {
  v = v01*(p.x-x23)/(x01-x23) +
      v23*(p.x-x01)/(x23-x01);
  return(v);
  } else {
   PetscPrintf(PETSC_COMM_SELF, "Bilinear intrp Failed x!!!!!!!!!!!!!\n"); 
  }

  } else {
    PetscPrintf(PETSC_COMM_SELF, "Bilinear intrp Failed y!!!!!!!!!!!!!\n"); 
  }
  return(0.);
}

PetscReal TrilinearInterpolation(Cmpnts host[8], Cmpnts p, PetscReal val[8])
{
  /* Needs special ordering of host
             vkji 
     host[0]=v000
     host[1]=v100
     host[2]=v010
     host[3]=v110
     host[4]=v001
     host[5]=v101
     host[6]=v011
     host[7]=v111
  */

  Cpt2D      bih[4], bip ;
  PetscReal  v, bval[4] ;

  bip.x=p.x;
  bip.y=p.y;
  // in z
  if (fabs(host[0].z-host[1].z)>1e-9) {
    bval[0]=val[0]*(p.z-host[1].z)/(host[0].z-host[1].z)
          + val[1]*(p.z-host[0].z)/(host[1].z-host[0].z);    
    bih[0].x=host[0].x*(p.z-host[1].z)/(host[0].z-host[1].z)
           + host[1].x*(p.z-host[0].z)/(host[1].z-host[0].z);
    bih[0].y=host[0].y*(p.z-host[1].z)/(host[0].z-host[1].z)
           + host[1].y*(p.z-host[0].z)/(host[1].z-host[0].z);
    
    bval[1]=val[2]*(p.z-host[3].z)/(host[2].z-host[3].z)
          + val[3]*(p.z-host[2].z)/(host[3].z-host[2].z);    
    bih[1].x=host[2].x*(p.z-host[3].z)/(host[2].z-host[3].z)
           + host[3].x*(p.z-host[2].z)/(host[3].z-host[2].z);
    bih[1].y=host[2].y*(p.z-host[3].z)/(host[2].z-host[3].z)
           + host[3].y*(p.z-host[2].z)/(host[3].z-host[2].z);

    bval[2]=val[4]*(p.z-host[5].z)/(host[4].z-host[5].z)
          + val[5]*(p.z-host[4].z)/(host[5].z-host[4].z);    
    bih[2].x=host[4].x*(p.z-host[5].z)/(host[4].z-host[5].z)
           + host[5].x*(p.z-host[4].z)/(host[5].z-host[4].z);
    bih[2].y=host[4].y*(p.z-host[5].z)/(host[4].z-host[5].z)
           + host[5].y*(p.z-host[4].z)/(host[5].z-host[4].z);

    bval[3]=val[6]*(p.z-host[7].z)/(host[6].z-host[7].z)
          + val[7]*(p.z-host[6].z)/(host[7].z-host[6].z);    
    bih[3].x=host[6].x*(p.z-host[7].z)/(host[6].z-host[7].z)
          + host[7].x*(p.z-host[6].z)/(host[7].z-host[6].z);
    bih[3].y=host[6].y*(p.z-host[7].z)/(host[6].z-host[7].z)
          + host[7].y*(p.z-host[6].z)/(host[7].z-host[6].z);
    
    // in y,x
    v = BilinearInterpolation(bih,bip,bval);
    return(v);
  } else {
    PetscPrintf(PETSC_COMM_SELF, "Trilinear intrp Failed!!!!!!!!!!!!!\n"); 
  }
 return(0.);
}
 
PetscErrorCode MgRestrictionP(UserCtx *user,int flevel)
{ 
  /* 6/28/06 Iman
     Restrics Pressure from the fine grid to coarse grid
     flevel   finer grid level
     clevel   coarser grid level
  */

  DM da = user->da, fda = user->fda;

  UserCtx *user_f = user->user_f;
  DM da_f =  user_f->da, fda_f = user_f->fda;  	
  DMDALocalInfo	info;
  DMDAGetLocalInfo(da, &info);

  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;
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

  PetscReal	***p_c, ***p_f, pRhost[8];
  Cmpnts        ***coor_c, ***coor_f, Rhost[8], R;

  int      i,j,k;

  // Get the vectors
  DMDAVecGetArray(fda, user->lCent,&coor_f);
  DMDAVecGetArray(fda_f, user_f->lCent,&coor_c);
  DMDAVecGetArray(da, user->lP, &p_f);
  DMDAVecGetArray(da_f, user_f->lP, &p_c);

  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	R=coor_c[k][j][i];

      /*  trilinear interpolation
	Vxyz = 	V000 (1 - x) (1 - y) (1 - z) +
	V100 x (1 - y) (1 - z) +
	V010 (1 - x) y (1 - z) +
	V001 (1 - x) (1 - y) z +
	V101 x (1 - y) z +
	V011 (1 - x) y z +
	V110 x y (1 - z) +
	V111 x y z
      */ 
       	
	Rhost[0]=coor_f[2*k-1][2*j-1][2*i-1];
	Rhost[1]=coor_f[2*k  ][2*j-1][2*i-1];
	Rhost[2]=coor_f[2*k-1][2*j  ][2*i-1];
	Rhost[3]=coor_f[2*k  ][2*j  ][2*i-1];
	Rhost[4]=coor_f[2*k-1][2*j-1][2*i  ];
	Rhost[5]=coor_f[2*k  ][2*j-1][2*i  ];
	Rhost[6]=coor_f[2*k-1][2*j  ][2*i  ];
	Rhost[7]=coor_f[2*k  ][2*j  ][2*i  ];

	pRhost[0]=p_f[2*k-1][2*j-1][2*i-1];
	pRhost[1]=p_f[2*k  ][2*j-1][2*i-1];
	pRhost[2]=p_f[2*k-1][2*j  ][2*i-1];
	pRhost[3]=p_f[2*k  ][2*j  ][2*i-1];
	pRhost[4]=p_f[2*k-1][2*j-1][2*i  ];
	pRhost[5]=p_f[2*k  ][2*j-1][2*i  ];
	pRhost[6]=p_f[2*k-1][2*j  ][2*i  ];
	pRhost[7]=p_f[2*k  ][2*j  ][2*i  ];

	p_c[k][j][i]= TrilinearInterpolation(Rhost,R,pRhost);

/* 	// Calc Coeff using lagrange formula */
/* 	for (n=0;n<8;n++) { */
/* 	  Rcoeff[n]=1.; */
/* 	  for (l=0;l<8;l++) { */
/* 	    if (l!=n) { */
/* 	      if (fabs(Rhost[n].x-Rhost[l].x)>1e-8)  */
/* 		Rcoeff[n] *=(R.x-Rhost[l].x)/(Rhost[n].x-Rhost[l].x); */
/* 	      if (fabs(Rhost[n].y-Rhost[l].y)>1e-8)  */
/* 		Rcoeff[n] *=(R.y-Rhost[l].y)/(Rhost[n].y-Rhost[l].y); */
/* 	      if (fabs(Rhost[n].z-Rhost[l].z)>1e-8)  */
/* 		Rcoeff[n] *=(R.z-Rhost[l].z)/(Rhost[n].z-Rhost[l].z); */
/* 	    }//if */
/* 	  }//l */
/* 	}//n */	     
/* 	p_c[k][j][i]=0.; */
/* 	for (n=0;n<8;n++) { */
/* 	  Rcoeff[n]=1/8.; */
/* 	  p_c[k][j][i]+=Rcoeff[n]*pRhost[n]; */
/* 	} */

      }
    }
  }

  // Restore vectors
  DMDAVecRestoreArray(fda, user->lCent,&coor_f);
  DMDAVecRestoreArray(fda_f, user_f->lCent,&coor_c);
  DMDAVecRestoreArray(da, user->lP, &p_f);
  DMDAVecRestoreArray(da_f, user_f->lP, &p_c);

  return(0);
}

PetscErrorCode MgGridInterpolation(int i, int j, int k,
				   int *ih, int *jh, int *kh,
				   int dir, UserCtx *user)
{
  //PetscPrintf(PETSC_COMM_WORLD, "grid intp/n");
  if (*(user->isc)) {
    *ih = i;
  }
  else if (dir==0 || i%2==0){
    *ih = i/2;
  }
  else {
    *ih = i/2 + 1;
  }

  if (*(user->jsc)) {
    *jh = j;
  }
  else if (dir==1 || j%2==0){
    *jh = j/2;
  }
  else {
    *jh = j/2 + 1;
  }

  if (*(user->ksc)) {
    *kh = k;
  }
  else if (dir==2 || k%2==0){
    *kh = k/2;
  }
  else {
    *kh = k/2 + 1;
  }

  return 0;
}

PetscErrorCode MgInterpolationdU_old(UserCtx *user)
{
/*  Input: Coarsegrid dU (user) */
/*  Output: Finegrid dU   */
  DM da = user->da, fda = user->fda;

  UserCtx *user_f = user->user_f;
  DM	fda_f = user_f->fda;  	
  DMDALocalInfo	info;
  DMDAGetLocalInfo(da, &info);
  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;
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

  int i, j, k, ih, jh, kh, ia, ja, ka, dir;
  Cmpnts ***dU, ***dU_f;

  if (*(user->isc)) ia = 1;
  else ia = 0;

  if (*(user->jsc)) ja = 1;
  else ja = 0;

  if (*(user->ksc)) ka = 1;
  else ka = 0;

  DMDAVecGetArray(fda, user->dUcont, &dU);
  DMDAVecGetArray(fda_f, user_f->dUcont, &dU_f);

  // x-dir
  dir=0;
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=xs; i<lxe; i++) {
	MgGridInterpolation(i, j, k, &ih, &jh, &kh,dir, user);

	if (i%2==0) {
	dU_f[k][j][i].x = (dU[kh   ][jh   ][ih  ].x )
		            / (2.-ka) / (2.-ja);
	} else {
	dU_f[k][j][i].x = (dU[kh   ][jh   ][ih     ].x +
			   dU[kh   ][jh   ][ih+1-ia].x )
		            / (2.-ka) / (2.-ja) / 2.;
	}
      }
    }
  }

  // y-dir
  dir=1;
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=xs; i<lxe; i++) {
	MgGridInterpolation(i, j, k, &ih, &jh, &kh,dir, user);
	
	if (j%2==0) {
	dU_f[k][j][i].y = (dU[kh   ][jh   ][ih  ].y )
		            / (2.-ka) / (2.-ia);
	} else {
	dU_f[k][j][i].y = (dU[kh   ][jh     ][ih  ].y +
			   dU[kh   ][jh+1-ja][ih  ].y )
		            / (2.-ka) / (2.-ia) / 2.;
	}
      }
    }
  }

  // z-dir
  dir=2;
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=xs; i<lxe; i++) {
	MgGridInterpolation(i, j, k, &ih, &jh, &kh,dir, user);
	
	if (k%2==0) {
	dU_f[k][j][i].z = (dU[kh   ][jh   ][ih  ].z )
		            / (2.-ia) / (2.-ja);
	} else {
	dU_f[k][j][i].z = (dU[kh     ][jh   ][ih  ].z +
			   dU[kh+1-ka][jh   ][ih  ].z )
		            / (2.-ia) / (2.-ja) / 2.;
	}
      }
    }
  }

  DMDAVecRestoreArray(fda, user->dUcont, &dU);
  DMDAVecRestoreArray(fda_f, user_f->dUcont, &dU_f);

  return(0);
}
/*
PetscErrorCode MgInterpolationdU(UserCtx *user)
{
  DM	fda = user->fda;

  UserCtx *user_f = user->user_f;
  DM da_f =  user_f->da, fda_f = user_f->fda;  	
  DMDALocalInfo	info;
  DMDAGetLocalInfo(da_f, &info);
  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;
  int	lxs, lxe, lys, lye, lzs, lze, llze,llye,llxe;
  lxs = xs; lxe = xe; llxe=xe;
  lys = ys; lye = ye; llye=ye;
  lzs = zs; lze = ze; llze=ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  if (xe==mx) llxe = xe-2;
  if (ye==my) llye = ye-2;
  if (ze==mz) llze = ze-2;

  int i, j, k, ih, jh, kh, ia, ja, ka, dir;
  Cmpnts ***dU, ***dU_f;
  Cmpnts ***dA, ***dA_f;

  if (*(user->isc)) ia = 1;
  else ia = 0;

  if (*(user->jsc)) ja = 1;
  else ja = 0;

  if (*(user->ksc)) ka = 1;
  else ka = 0;

  DMDAVecGetArray(fda, user->dUcont, &dU);
  DMDAVecGetArray(fda_f, user_f->dUcont, &dU_f);
  DMDAVecGetArray(fda, user->lArea, &dA);
  DMDAVecGetArray(fda_f, user_f->lArea, &dA_f);

  // x-dir
  dir=0;
  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=xs; i<llxe; i++) {
	MgGridInterpolation(i, j, k, &ih, &jh, &kh,dir, user);

	if (i%2==0) {
	dU_f[k][j][i].x = (dU[kh   ][jh   ][ih  ].x /
			   dA[kh   ][jh   ][ih  ].x)
		         * dA_f[k][j][i].x;
	} else {
	dU_f[k][j][i].x = (dU[kh   ][jh   ][ih     ].x /
			   dA[kh   ][jh   ][ih     ].x +
			   dU[kh   ][jh   ][ih+1-ia].x /
			   dA[kh   ][jh   ][ih+1-ia].x)
		            * dA_f[k][j][i].x / 2.;
	}
       
	if (fabs(dA_f[k][j][i].x)<1e-6)
	  PetscPrintf(PETSC_COMM_SELF, "dA_x is zero!!!! %d %d %d \n", i, j, k);

      }
    }
  }

  // y-dir
  dir=1;
  for (k=lzs; k<lze; k++) {
    for (j=ys; j<llye; j++) {
      for (i=lxs; i<lxe; i++) {
	MgGridInterpolation(i, j, k, &ih, &jh, &kh,dir, user);
	
	if (j%2==0) {
	dU_f[k][j][i].y = (dU[kh   ][jh   ][ih  ].y /
			   dA[kh   ][jh   ][ih  ].y)
		         * dA_f[k][j][i].y;
	} else {
	dU_f[k][j][i].y = (dU[kh   ][jh     ][ih  ].y /
			   dA[kh   ][jh     ][ih  ].y +
			   dU[kh   ][jh+1-ja][ih  ].y /
			   dA[kh   ][jh+1-ja][ih  ].y)
		         * dA_f[k][j][i].y   / 2.;
	}

	if (fabs(dA_f[k][j][i].y)<1e-6)
	  PetscPrintf(PETSC_COMM_SELF, "dA_y is zero!!!! %d %d %d \n", i, j, k);
      }
    }
  }

  // z-dir
  dir=2;
  for (k=zs; k<llze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	MgGridInterpolation(i, j, k, &ih, &jh, &kh,dir, user);
	
	if (k%2==0) {
	dU_f[k][j][i].z = (dU[kh   ][jh   ][ih  ].z /
			   dA[kh   ][jh   ][ih  ].z)
		          * dA_f[k][j][i].z;
	} else {
	dU_f[k][j][i].z = (dU[kh     ][jh   ][ih  ].z /
			   dA[kh     ][jh   ][ih  ].z +
			   dU[kh+1-ka][jh   ][ih  ].z /
			   dA[kh+1-ka][jh   ][ih  ].z)
		           * dA_f[k][j][i].z / 2.;
	}

	if (fabs(dA_f[k][j][i].z)<1e-6)
	  PetscPrintf(PETSC_COMM_SELF, "dA_z is zero!!!! %d %d %d \n", i, j, k);

      }
    }
  }

  DMDAVecRestoreArray(fda, user->dUcont, &dU);
  DMDAVecRestoreArray(fda_f, user_f->dUcont, &dU_f);
  DMDAVecRestoreArray(fda, user->lArea, &dA);
  DMDAVecRestoreArray(fda_f, user_f->lArea, &dA_f);

  return(0);
}
*/
/*
PetscErrorCode MgFieldRestriction(UserCtx *user)
{
  DM da = user->da, fda = user->fda;

  DM da_f = *user->da_f;

  UserCtx *user_f = user->user_f;
  DM	fda_f = user_f->fda;

  DMDALocalInfo	info;
  DMDAGetLocalInfo(da, &info);
  int	xs = info.xs, xe = info.xs + info.xm;
  int  	ys = info.ys, ye = info.ys + info.ym;
  int	zs = info.zs, ze = info.zs + info.zm;
  int	mx = info.mx, my = info.my, mz = info.mz;
  int	lxs, lxe, lys, lye, lzs, lze;

  Cmpnts ***ucont, ***ucont_f;
  Cmpnts ***ucont_o, ***ucont_o_f;
  Cmpnts ***ucont_rm1, ***ucont_rm1_f;

  int i, j, k, ih, jh, kh, ia, ja, ka;

  lxs = xs; lxe = xe;
  lys = ys; lye = ye;
  lzs = zs; lze = ze;

  if (xs==0) lxs = xs+1;
  if (ys==0) lys = ys+1;
  if (zs==0) lzs = zs+1;

  if (xe==mx) lxe = xe-1;
  if (ye==my) lye = ye-1;
  if (ze==mz) lze = ze-1;

  DMDAVecGetArray(fda, user->Ucont, &ucont);
  DMDAVecGetArray(fda_f, user_f->lUcont, &ucont_f);
  DMDAVecGetArray(fda, user->Ucont_o, &ucont_o);
  DMDAVecGetArray(fda_f, user_f->lUcont_o, &ucont_o_f);
  DMDAVecGetArray(fda, user->Ucont_rm1, &ucont_rm1);
  DMDAVecGetArray(fda_f, user_f->lUcont_rm1, &ucont_rm1_f);

  if (*(user->isc)) ia = 0;
  else ia = 1;

  if (*(user->jsc)) ja = 0;
  else ja = 1;

  if (*(user->ksc)) ka = 0;
  else ka = 1;

  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=xs; i<lxe; i++) {
	GridRestriction(i, j, k, &ih, &jh, &kh, user);
	
	ucont[k][j][i].x = (ucont_f[kh   ][jh   ][ih  ].x +
			    ucont_f[kh-ka][jh   ][ih  ].x +
			    ucont_f[kh   ][jh-ja][ih  ].x +
			    ucont_f[kh-ka][jh-ja][ih  ].x) / (2.-ka) / (2.-ja);

	ucont_o[k][j][i].x = (ucont_o_f[kh   ][jh   ][ih  ].x +
			      ucont_o_f[kh-ka][jh   ][ih  ].x +
			      ucont_o_f[kh   ][jh-ja][ih  ].x +
			      ucont_o_f[kh-ka][jh-ja][ih  ].x) / (2.-ka) / (2.-ja);

	ucont_rm1[k][j][i].x = (ucont_rm1_f[kh   ][jh   ][ih  ].x +
				ucont_rm1_f[kh-ka][jh   ][ih  ].x +
				ucont_rm1_f[kh   ][jh-ja][ih  ].x +
				ucont_rm1_f[kh-ka][jh-ja][ih  ].x) / (2.-ka) / (2.-ja);
      }
    }
  }

  for (k=lzs; k<lze; k++) {
    for (j=ys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	GridRestriction(i, j, k, &ih, &jh, &kh, user);

	ucont[k][j][i].y = (ucont_f[kh   ][jh  ][ih   ].y +
			    ucont_f[kh-ka][jh  ][ih   ].y +
			    ucont_f[kh   ][jh  ][ih-ia].y +
			    ucont_f[kh-ka][jh  ][ih-ia].y) / (2.-ka) / (2.-ia);

	ucont_o[k][j][i].y = (ucont_o_f[kh   ][jh  ][ih   ].y +
			      ucont_o_f[kh-ka][jh  ][ih   ].y +
			      ucont_o_f[kh   ][jh  ][ih-ia].y +
			      ucont_o_f[kh-ka][jh  ][ih-ia].y) / (2.-ka) / (2.-ia);

	ucont_rm1[k][j][i].y = (ucont_rm1_f[kh   ][jh  ][ih   ].y +
				ucont_rm1_f[kh-ka][jh  ][ih   ].y +
				ucont_rm1_f[kh   ][jh  ][ih-ia].y +
				ucont_rm1_f[kh-ka][jh  ][ih-ia].y) / (2.-ka) / (2.-ia);
      }
    }
  }

  for (k=zs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	GridRestriction(i, j, k, &ih, &jh, &kh, user);

	ucont[k][j][i].z = (ucont_f[kh  ][jh   ][ih   ].z +
			    ucont_f[kh  ][jh   ][ih-ia].z +
			    ucont_f[kh  ][jh-ja][ih   ].z +
			    ucont_f[kh  ][jh-ja][ih-ia].z) / (2.-ja) / (2.-ia);

	ucont_o[k][j][i].z = (ucont_o_f[kh  ][jh   ][ih   ].z +
			      ucont_o_f[kh  ][jh   ][ih-ia].z +
			      ucont_o_f[kh  ][jh-ja][ih   ].z +
			      ucont_o_f[kh  ][jh-ja][ih-ia].z) / (2.-ja) / (2.-ia);

	ucont_rm1[k][j][i].z = (ucont_rm1_f[kh  ][jh   ][ih   ].z +
				ucont_rm1_f[kh  ][jh   ][ih-ia].z +
				ucont_rm1_f[kh  ][jh-ja][ih   ].z +
				ucont_rm1_f[kh  ][jh-ja][ih-ia].z) / (2.-ja) / (2.-ia);
      }
    }
  }

  DMDAVecRestoreArray(fda, user->Ucont, &ucont);
  DMDAVecRestoreArray(fda_f, user_f->lUcont, &ucont_f);
  DMDAVecRestoreArray(fda, user->Ucont_o, &ucont_o);
  DMDAVecRestoreArray(fda_f, user_f->lUcont_o, &ucont_o_f);
  DMDAVecRestoreArray(fda, user->Ucont_rm1, &ucont_rm1);
  DMDAVecRestoreArray(fda_f, user_f->lUcont_rm1, &ucont_rm1_f);
  
  DMGlobalToLocalBegin(fda, user->Ucont, INSERT_VALUES, user->lUcont);
  DMGlobalToLocalEnd(fda, user->Ucont, INSERT_VALUES, user->lUcont);

  DMGlobalToLocalBegin(fda, user->Ucont_o, INSERT_VALUES, user->lUcont_o);
  DMGlobalToLocalEnd(fda, user->Ucont_o, INSERT_VALUES, user->lUcont_o);

  DMGlobalToLocalBegin(fda, user->Ucont_rm1, INSERT_VALUES, user->lUcont_rm1);
  DMGlobalToLocalEnd(fda, user->Ucont_rm1, INSERT_VALUES, user->lUcont_rm1);

  VecCopy(user->Ucont, user->Ucont_MG);

  PetscReal ***p, ***p_f, ***v_f, v_tot;

  DMDAVecGetArray(da, user->P, &p);
  DMDAVecGetArray(da_f, user_f->lP, &p_f);
  DMDAVecGetArray(da_f, user_f->lVolume, &v_f);


  for (k=lzs; k<lze; k++) {
    for (j=lys; j<lye; j++) {
      for (i=lxs; i<lxe; i++) {
	GridRestriction(i, j, k, &ih, &jh, &kh, user);

	v_tot =(      v_f[kh   ][jh   ][ih   ] +		
		      v_f[kh   ][jh   ][ih-ia] +		
		      v_f[kh   ][jh-ja][ih   ] +		
		      v_f[kh   ][jh-ja][ih-ia] +		
		      v_f[kh-ka][jh   ][ih   ] +		
		      v_f[kh-ka][jh   ][ih-ia] +		
		      v_f[kh-ka][jh-ja][ih   ] +		
		      v_f[kh-ka][jh-ja][ih-ia]    );

	p[k][j][i] = (p_f[kh   ][jh   ][ih   ] *
		      v_f[kh   ][jh   ][ih   ] +
		      p_f[kh   ][jh   ][ih-ia] *
		      v_f[kh   ][jh   ][ih-ia] +
		      p_f[kh   ][jh-ja][ih   ] *
		      v_f[kh   ][jh-ja][ih   ] +
		      p_f[kh   ][jh-ja][ih-ia] *
		      v_f[kh   ][jh-ja][ih-ia] +
		      p_f[kh-ka][jh   ][ih   ] *
		      v_f[kh-ka][jh   ][ih   ] +
		      p_f[kh-ka][jh   ][ih-ia] *
		      v_f[kh-ka][jh   ][ih-ia] +
		      p_f[kh-ka][jh-ja][ih   ] *
		      v_f[kh-ka][jh-ja][ih   ] +
		      p_f[kh-ka][jh-ja][ih-ia] *
		      v_f[kh-ka][jh-ja][ih-ia])/v_tot;

      }
    }
  }

  
  DMDAVecRestoreArray(da, user->P, &p);
  DMDAVecRestoreArray(da_f, user_f->lP, &p_f);
  DMDAVecRestoreArray(da_f, user_f->lVolume, &v_f);
  
  DMGlobalToLocalBegin(da, user->P, INSERT_VALUES, user->lP);
  DMGlobalToLocalEnd(da, user->P, INSERT_VALUES, user->lP);
		      
  return 0;
}
*/

/* ==================================================================================             */
/*      Multi-Grid Cycle  */
/*
PetscErrorCode MGCYC(UserMG *usermg,IBMNodes *ibm, FSInfo *fsi,int level, int cycl_idx, int preItr, int postItr,int CGItr)
{

  int i;
  Vec      pUcont, dUcont;

  UserCtx *user, *user_c;

  user   = usermg->mgctx[level].user;
  user_c = usermg->mgctx[level-1].user;

  DM    fda=user->fda;

//     Presmoothing fG 
  ImplicitSmoother(user,ibm, fsi,preItr);
  PetscPrintf(PETSC_COMM_WORLD, "PRE SMOOTHING!\n");


//    CGC: 
//    1) Restrict U,p to CG
    
    //MgRestrictionP(user, level, isc, jsc, ksc);
    MgFieldRestriction(user_c);
    //MyNvertRestriction(user, user_c);
    PetscPrintf(PETSC_COMM_WORLD, "RESTRICTION!!!\n");

//      save Ucont  
    VecDuplicate(user_c->lUcont, &pUcont);   
    VecDuplicate(user_c->lUcont, &dUcont);   
    VecCopy(user_c->lUcont, pUcont);

//      2) Solve Ucont on CG 
    if (level-1==0) { // Coarsest level: Solve to machine zero!
      //ImplicitMomentumSolver(user_c, ibm, fsi);
      ImplicitSmoother(user_c,ibm, fsi, CGItr);
      PetscPrintf(PETSC_COMM_WORLD, "CG Solver!!!\n");

    } else { // not the coarsest level: Solve by MGM
      for (i=0; i<cycl_idx; i++) {
	MGCYC(usermg, ibm, fsi,level-1,cycl_idx, preItr, postItr, CGItr);
      }
    }
  

//      calc dUcont  
    VecWAXPY(dUcont, -1., pUcont, user_c->lUcont);
    VecDuplicate(user_c->lUcont, &(user_c->dUcont));
    VecCopy(dUcont, user_c->dUcont);
    VecDuplicate(user->Ucont, &(user->dUcont));


      //3) Interpolate dU from CG to finer Grid
    MgInterpolationdU(user_c);  
    // Ucont=Ucont+dU on fine Grid
    VecDuplicate(user->Ucont, &(user->pUcont));
    VecCopy(user->Ucont, user->pUcont);
    VecWAXPY(user->Ucont, +1., user->dUcont, user->pUcont);  
    
    DMGlobalToLocalBegin(fda, user->Ucont, INSERT_VALUES, user->lUcont);
    DMGlobalToLocalEnd(fda, user->Ucont, INSERT_VALUES, user->lUcont);

    PetscPrintf(PETSC_COMM_WORLD, "INTERPOLATION!!!\n");
    

//      Destroy 
    VecDestroy(&dUcont);
    VecDestroy(&pUcont);
    VecDestroy(&user->pUcont);
    VecDestroy(&user_c->dUcont);
    VecDestroy(&user->dUcont);

//    Postsmoothing fG 
  ImplicitSmoother(user,ibm,fsi, postItr);
  PetscPrintf(PETSC_COMM_WORLD, "POST SMOOTHING!\n");

  
  return(0);
}
*/
/* ==================================================================================             */
/* ==================================================================================             */
/*
PetscErrorCode MGMomentumSolver(UserMG *usermg,IBMNodes *ibm, FSInfo *fsi)
{
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  int finest_level=usermg->mglevels - 1;
  UserCtx *user, *user_l;
  user = usermg->mgctx[finest_level].user;
  
  Vec pUcont, dUcont;
  
  int bi, level;
  int pseudot;
  PetscReal normdU=10.,normdU1=1.,reldU=1.,normdT, cput,ts,te;
  pseudot=0;
  
  for (bi=0; bi<block_number; bi++) { 

    InflowFlux(&(user[bi]));
    OutflowFlux(&(user[bi]));
    FormBCS(&(user[bi]),&fsi[0]);

    if (immersed)
	{
      ibm_interpolation_advanced(&user[bi],1);
    }

    for (level=finest_level-1; level>-1; level--) {
      user_l=usermg->mgctx[level].user;
      MyFieldRestriction(user_l);
    }

    VecDuplicate(user[bi].Ucont, &pUcont);   
    VecDuplicate(user[bi].Ucont, &dUcont);

    PetscGetTime(&ts);
    
    while (( (normdU>1e-7 && reldU>1e-3) || pseudot<3) && pseudot<mg_MAX_IT) {
      pseudot++;
      VecCopy(user[bi].Ucont, pUcont);

      MGCYC(usermg, ibm, fsi, finest_level, mg_idx, mg_preItr,mg_poItr,imp_MAX_IT);
      
      VecWAXPY(dUcont, -1., pUcont, user[bi].Ucont);


      VecNorm(dUcont, NORM_INFINITY, &normdU);
      if (pseudot==1) normdU1=normdU;
      if (pseudot>1) reldU=normdU/normdU1;
      VecNorm(user[bi].psuedot, NORM_INFINITY, &normdT);
      PetscGetTime(&te);
      cput=te-ts;
      PetscPrintf(PETSC_COMM_WORLD, "!!norm of dU MG %d  %le %le %le %le\n", pseudot, normdU, reldU, normdT, cput);

      DMGlobalToLocalBegin(user[bi].fda, user[bi].Ucont, INSERT_VALUES, user[bi].lUcont);
      DMGlobalToLocalEnd(user[bi].fda, user[bi].Ucont, INSERT_VALUES, user[bi].lUcont);
      
      InflowFlux(&(user[bi]));
      OutflowFlux(&(user[bi]));            

      FormBCS(&(user[bi]),&fsi[0]);

      if (immersed) 
	{
	  ibm_interpolation_advanced(&user[bi],1);
	}

      if (!rank) {
	FILE *f;
	char filen[80];
	sprintf(filen, "Converge_dU_MG");
	f = fopen(filen, "a");
	PetscFPrintf(PETSC_COMM_WORLD, f, "%d %d %le %le %le %le\n",ti,pseudot, normdU, reldU, normdT, cput);
	fclose(f);
      }
    }

    VecDestroy(&pUcont);
    VecDestroy(&dUcont);
  }
  return(0);
}

*/

void initial_guess_for_snes(UserCtx *user, Vec U)
{
	DM da = user->da, fda = user->fda;
	
	DMDALocalInfo	info = user->info;
	int	xs = info.xs, xe = info.xs + info.xm;
	int	ys = info.ys, ye = info.ys + info.ym;
	int	zs = info.zs, ze = info.zs + info.zm;
	int	mx = info.mx, my = info.my, mz = info.mz;

	int  i,j,k;
	PetscReal	***nvert;
	Cmpnts ***u, ***rhs, ***ucont;

	int lxs = xs, lxe = xe;
	int lys = ys, lye = ye;
	int lzs = zs, lze = ze;

	if (xs==0) lxs = xs+1;
	if (ys==0) lys = ys+1;
	if (zs==0) lzs = zs+1;

	if (xe==mx) lxe = xe-1;
	if (ye==my) lye = ye-1;
	if (ze==mz) lze = ze-1;
  
	VecCopy(user->Ucont, U);
	
	PetscPrintf (PETSC_COMM_WORLD, "Generaing initial guess ...\n");
	VecSet(user->Rhs,0);
	Formfunction_2(user, user->Rhs, 1.0);
	
	DMDAVecGetArray(da, user->lNvert, &nvert);
	DMDAVecGetArray(fda, U, &u);
	DMDAVecGetArray(fda, user->Rhs, &rhs);
	DMDAVecGetArray(fda, user->Ucont, &ucont);
	
	for (k=lzs; k<lze; k++)
	for (j=lys; j<lye; j++)
	for (i=lxs; i<lxe; i++) {
		if((int)nvert[k][j][i]==0 && (int)nvert[k][j][i+1]==0) u[k][j][i].x = ucont[k][j][i].x + user->dt * rhs[k][j][i].x;
		else u[k][j][i].x = ucont[k][j][i].x;
		
		if((int)nvert[k][j][i]==0 && (int)nvert[k][j+1][i]==0) u[k][j][i].y = ucont[k][j][i].y + user->dt * rhs[k][j][i].y;
		else u[k][j][i].y = ucont[k][j][i].y;
		
		if((int)nvert[k][j][i]==0 && (int)nvert[k+1][j][i]==0) u[k][j][i].z = ucont[k][j][i].z + user->dt * rhs[k][j][i].z;
		else u[k][j][i].z = ucont[k][j][i].z;
	}
   	
	DMDAVecRestoreArray(da, user->lNvert, &nvert);
	DMDAVecRestoreArray(fda, U, &u);
	DMDAVecRestoreArray(fda, user->Rhs, &rhs);
	DMDAVecRestoreArray(fda, user->Ucont, &ucont);
}
